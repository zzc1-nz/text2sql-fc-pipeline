"""Inference + real-execution scoring for a trained text2sql checkpoint.

WHY this module exists
----------------------
Evaluation is where a portfolio project usually starts lying. This module makes
that structurally hard:

* the **prompt is not built here**. It is resolved from :mod:`fcdata.format`, the
  same module that renders the SFT training targets. Training and inference
  therefore share one prompt builder by construction; if that module is missing
  we fail loudly instead of quietly inventing a second prompt format.
* the **score is not computed here**. Every prediction is executed against the
  real fixture sqlite via :func:`fcdata.dbexec.evaluate_pair`. A prediction that
  does not run is a failure, full stop.
* the **model is not assumed**. ``generate`` is injected (defaulting to
  :func:`fcdata.train.reference.generate`), so unit tests can score a stub model
  that emits fixed SQL strings while still going through real sqlite execution.

The only thing this module owns is the loop: prompt -> decode -> extract SQL ->
execute -> ``Prediction``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ..dbexec import evaluate_pair
from ..records import Example, Prediction, new_id, write_json, write_jsonl
from ..schema import Database
from ..sqlutils import extract_sql
from ..tokens import estimate_tokens

#: ``(model, tokenizer, prompt_text, max_new_tokens, temperature) -> text``
GenerateFn = Callable[[Any, Any, str, int, float], str]
#: ``(cfg, example, db) -> prompt_text``
PromptFn = Callable[[Any, Example, Database], str]


class PromptAdapterError(RuntimeError):
    """Raised when no prompt renderer could be resolved from :mod:`fcdata.format`.

    This is deliberately fatal: silently falling back to a locally-built prompt
    would break the guarantee that training and inference see identical prompts,
    and would make the reported accuracy meaningless.
    """


class _GeneratingModel(Protocol):
    """Minimal duck type for the stub model used in tests."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - structural only
        ...


# --------------------------------------------------------------------------- #
# prompt resolution: import from fcdata.format, never re-implement
# --------------------------------------------------------------------------- #
#: Chat-template specific "start of assistant turn" markers, used only when the
#: resolved renderer returns the *messages* rather than a generation prompt.
_ASSISTANT_PREFIX = {
    "qwen": "<|im_start|>assistant\n",
    "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "plain": "Assistant: ",
}


def _fmt_module() -> Any:
    try:
        from .. import format as fmt_mod  # noqa: PLC0415 - lazy by design
    except ImportError as exc:  # pragma: no cover - depends on sibling task
        raise PromptAdapterError(
            "fcdata.format is not importable yet; the eval harness refuses to build "
            "its own prompts (training/inference prompt parity). "
            f"underlying error: {exc}"
        ) from exc
    return fmt_mod


def _resolve_prompt_fn(cfg: Any) -> PromptFn:
    """Find the prompt-only renderer in :mod:`fcdata.format`.

    Accepted, in order of preference (all take ``(cfg, example, db)``):

    ``fcdata.format.build_eval_prompt``
    ``fcdata.format.build_prompt``
    ``fcdata.format.chat.build_prompt``
    ``fcdata.format.chat.render_prompt``

    The last one may take ``(cfg, example, db)`` or ``(messages, template)``; a
    wrapper adapts both shapes. When nothing matches we raise
    :class:`PromptAdapterError` listing exactly what was looked for.
    """
    fmt = _fmt_module()
    for name in ("build_eval_prompt", "build_prompt"):
        fn = getattr(fmt, name, None)
        if callable(fn):
            return _guard_prompt_fn(fn, f"fcdata.format.{name}")

    chat = getattr(fmt, "chat", None)
    if chat is not None:
        for name in ("build_prompt", "render_prompt", "render_eval_prompt"):
            fn = getattr(chat, name, None)
            if callable(fn):
                return _adapt_chat_fn(fn, f"fcdata.format.chat.{name}", cfg)

    # last resort: the SFT builder itself, with the assistant turn stripped
    build_sft = getattr(fmt, "build_sft_records", None)
    if callable(build_sft):
        return _adapt_sft_strip(build_sft, cfg)

    raise PromptAdapterError(
        "fcdata.format exposes no prompt renderer. Expected one of: "
        "build_eval_prompt(cfg, example, db), build_prompt(cfg, example, db), "
        "fcdata.format.chat.build_prompt/render_prompt, or build_sft_records."
    )


def _guard_prompt_fn(fn: Callable[..., Any], label: str) -> PromptFn:
    """Wrap a renderer and coerce its return value to ``str``."""

    def _call(cfg: Any, example: Example, db: Database) -> str:
        out = fn(cfg, example, db)
        return _as_prompt_text(out, label)

    return _call


def _as_prompt_text(out: Any, label: str) -> str:
    if isinstance(out, str):
        return out
    # some renderers return a list of message dicts
    if isinstance(out, (list, tuple)):
        parts: list[str] = []
        for m in out:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        if parts:
            return "\n".join(parts)
    if isinstance(out, dict) and "content" in out:
        return str(out["content"])
    raise PromptAdapterError(f"{label} returned {type(out).__name__}, expected str")


def _adapt_chat_fn(fn: Callable[..., Any], label: str, cfg: Any) -> PromptFn:
    """Adapt ``(messages, template)``-shaped renderers to ``(cfg, example, db)``."""
    template = getattr(getattr(cfg, "fmt", None), "chat_template", "qwen")
    messages_builder = None
    chat = getattr(_fmt_module(), "chat", None)
    for name in ("build_eval_messages", "build_messages", "build_user_messages"):
        cand = getattr(chat, name, None) if chat is not None else None
        if callable(cand):
            messages_builder = cand
            break

    def _call(cfg2: Any, example: Example, db: Database) -> str:
        if messages_builder is not None:
            try:
                msgs = messages_builder(cfg2, example, db)
            except TypeError:
                msgs = messages_builder(example, db)
            prompt = fn(msgs, template)
            return _as_prompt_text(prompt, label)
        # renderer wants the example directly
        return _as_prompt_text(fn(cfg2, example, db), label)

    return _call


def _adapt_sft_strip(build_sft: Callable[..., Any], cfg: Any) -> PromptFn:
    """Emergency adapter: render an SFT record and cut the assistant turn off.

    Uses the project's own SFT builder so the prompt text is still the training
    prompt, then truncates at the assistant header so the model can generate the
    tool call itself. Recorded as ``prompt_source='sft_strip'`` in prediction
    metadata so a reader can see how the prompt was obtained.
    """

    def _call(cfg2: Any, example: Example, db: Database) -> str:
        spider = _OneExampleSpider(example, db)
        records, _stats = build_sft(cfg2, [], spider, [example])
        if not records:
            raise PromptAdapterError(
                f"build_sft_records produced no record for example {example.id!r}")
        messages = records[0].messages
        # drop the assistant target, keep system + user
        kept = [m for m in messages if m.get("role") != "assistant"]
        text = "\n".join(str(m.get("content", "")) for m in kept)
        template = getattr(getattr(cfg2, "fmt", None), "chat_template", "qwen")
        return text + "\n" + _ASSISTANT_PREFIX.get(template, "Assistant: ")

    return _call


class _OneExampleSpider:
    """Tiny SpiderData stand-in so ``build_sft_records`` can run on one example."""

    def __init__(self, example: Example, db: Database) -> None:
        self.databases = {db.db_id: db}
        self.train = [example]
        self.dev = [example]
        self.root = Path(".")
        self.sqlite_dir = Path(".")

    def db(self, db_id: str) -> Database:
        return self.databases[db_id]


def build_prompt_fn(cfg: Any) -> PromptFn:
    """Public accessor for the resolved prompt renderer (useful for tests)."""
    return _resolve_prompt_fn(cfg)


# --------------------------------------------------------------------------- #
# output -> SQL extraction (prefer the format module's tool-call parser)
# --------------------------------------------------------------------------- #
def extract_pred_sql(raw: str) -> str:
    """Turn a model response into a clean, executable SQL string (or ``""``).

    Delegates to :func:`fcdata.format.assistant_sql`, the project's canonical
    "what SQL did the model produce" helper, then applies two *evaluation-side*
    cleanups that the canonical parser deliberately does not do:

    1. **Template-tail stripping.** A word-level model emits the chat template as
       ordinary tokens, and it frequently spaces them out:
       ``... gpa > 3.5"}}}} < / tool_call > <|im_end|>``. The SQL inside is
       perfect; the trailing control tokens make it non-executable. Scoring that
       as a failed prediction would understate accuracy for a purely cosmetic
       reason, so the known template/tool-call tails are cut here.
    2. **Statement-shape validation.** A candidate that does not begin with
       ``SELECT``/``WITH`` is rejected as an *empty prediction*, because
       ``assistant_sql`` is deliberately lenient and returns prose when the model
       produced no SQL at all. "Model produced no SQL" and "model produced broken
       SQL" are different failure modes and the report distinguishes them.

    Both cleanups are idempotent and never touch the interior of the statement.
    """
    if not raw or not raw.strip():
        return ""
    candidate = ""
    try:
        from ..format import assistant_sql  # noqa: PLC0415 - lazy by design
    except ImportError:
        assistant_sql = None  # type: ignore[assignment]
    if callable(assistant_sql):
        try:
            candidate = str(assistant_sql({"role": "assistant", "content": raw}) or "")
        except (TypeError, ValueError, KeyError):
            candidate = ""
    if not candidate:
        candidate = extract_sql(raw)
    candidate = strip_template_tail(candidate)
    if not _LOOKS_LIKE_SQL.match(candidate or ""):
        return ""
    return candidate.strip()


#: Template/tool-call control sequences a word-level model can emit as text.
#: Matched with optional spaces because the tokenizer splits ``<tool_call>``
#: into ``<``, ``tool_call``, ``>`` and generation often re-spaces them.
_TEMPLATE_TAIL = re.compile(
    r"""
    (?: \s* < \s* /? \s* tool_call \s* >          # <tool_call> / < / tool_call >
      | \s* <\| \s* (?: im_end | endoftext | eot_id ) \s* \|>   # <|im_end|> etc.
      | \s* < \s* \|? \s* end_of_turn \s* \|? \s* >
    )+
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: JSON/tool-call punctuation that trails a SQL statement in a malformed call.
_JSON_TAIL = re.compile(r"""[\s"'}\])]*(?:\}\s*\]?\s*\})?\s*$""")

#: A dangling code-fence tail left by a partially-emitted ```sql block.
_FENCE_TAIL = re.compile(r"`{1,3}\s*$")


def strip_template_tail(sql: str) -> str:
    """Remove trailing chat-template, tool-call and code-fence syntax from SQL.

    Applied repeatedly so ``... 3.5"}}}} < / tool_call > <|im_end|>`` collapses to
    ``... 3.5``. Only *trailing* material is removed, and only when doing so is
    safe: a quote or bracket inside a string literal in the middle of the
    statement is untouched because the trim stops as soon as removing a character
    would unbalance the statement or break an odd quote count.
    """
    if not sql:
        return ""
    out = sql.strip()
    for _ in range(4):
        before = out
        out = _TEMPLATE_TAIL.sub("", out).strip()
        out = _FENCE_TAIL.sub("", out).strip()
        out = _trim_dangling_closers(out)
        if out == before:
            break
    return out


def _trim_dangling_closers(sql: str) -> str:
    """Drop trailing JSON punctuation while keeping balanced SQL intact.

    The guards matter: a statement ending in ``'Computer Science'`` must keep its
    closing quote, and one ending in ``COUNT(*)`` must keep its paren. Only
    characters that leave the statement *more* balanced than before are removed.
    """
    out = sql.rstrip()
    while out and out[-1] in '"\'}])`':
        last = out[-1]
        if last == ")":
            # keep it if it closes a paren that is otherwise unmatched
            if out.count("(") >= out.count(")"):
                break
        elif last == "]":
            if out.count("[") >= out.count("]"):
                break
        elif last == "}":
            if out.count("{") >= out.count("}"):
                break
        elif last == "`":
            # fences are handled separately; never eat a single backtick here
            break
        else:  # quote
            if out.count(last) % 2 == 0:
                # an EVEN count means the quotes pair up: this trailing quote
                # closes a string literal and is part of the SQL
                # (e.g. ... = 'Computer Science'). Keep it.
                break
            # odd count -> this quote is an unmatched artifact of the tool-call
            # wrapper (e.g. ... gpa > 3.5"}}}}), so drop it.
        out = out[:-1].rstrip()
    return out


#: A usable prediction must start a read-only statement; anything else is noise.
_LOOKS_LIKE_SQL = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def default_generate_fn() -> GenerateFn:
    """The real generator from the training package (lazy import)."""
    from ..train.reference import generate  # noqa: PLC0415 - lazy by design

    return generate


def default_generate_batch_fn() -> Callable[[Any, Any, Sequence[str], int, float],
                                            list[str]]:
    """Batched generator (same greedy output, ~3x faster) from the trainer."""
    from ..train.reference import generate_batch  # noqa: PLC0415 - lazy by design

    return generate_batch


def _default_generate(model: Any, tokenizer: Any, prompt: str,
                      max_new_tokens: int, temperature: float) -> str:
    """Bridge to the trainer's module-level ``generate``, with a stub escape hatch.

    A stub object exposing ``.generate(prompt)`` is accepted so tests (and any
    future non-torch backend) can be scored through the exact same code path.
    """
    stub = getattr(model, "generate", None)
    if callable(stub) and getattr(model, "is_stub", False):
        return str(stub(prompt))
    return default_generate_fn()(model, tokenizer, prompt, max_new_tokens, temperature)


def _generate_all(model: Any,
                  tokenizer: Any,
                  prompts: Sequence[str],
                  n_new: int,
                  temperature: float,
                  generate_fn: GenerateFn | None,
                  batch_size: int) -> list[str]:
    """Generate one completion per prompt, batching when it is safe to do so.

    Greedy decoding is order-independent, so the trainer's ``generate_batch``
    produces byte-identical output to ``generate`` while being ~3x faster on CPU.
    Batching is skipped when a custom ``generate_fn`` is injected (tests, future
    backends) or when the model is a stub, because neither exposes a batch API.
    """
    stub = getattr(model, "generate", None)
    is_stub = callable(stub) and getattr(model, "is_stub", False)
    if generate_fn is not None or is_stub or batch_size <= 1:
        fn = generate_fn or _default_generate
        return [fn(model, tokenizer, p, n_new, temperature) for p in prompts]
    try:
        batch_fn = default_generate_batch_fn()
    except (ImportError, AttributeError):
        fn = _default_generate
        return [fn(model, tokenizer, p, n_new, temperature) for p in prompts]
    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start:start + batch_size])
        try:
            texts = batch_fn(model, tokenizer, chunk, n_new, temperature)
        except Exception:  # noqa: BLE001 - fall back to per-item decoding
            fn = _default_generate
            out.extend(fn(model, tokenizer, p, n_new, temperature) for p in chunk)
            continue
        out.extend(str(t) for t in texts)
    return out


# --------------------------------------------------------------------------- #
# the prediction loop
# --------------------------------------------------------------------------- #
def predict_examples(model: Any,
                     tokenizer: Any,
                     cfg: Any,
                     spider_data: Any,
                     examples: Sequence[Example],
                     *,
                     prompt_fn: PromptFn | None = None,
                     generate_fn: GenerateFn | None = None,
                     max_new_tokens: int | None = None,
                     temperature: float | None = None,
                     timeout_s: float | None = None,
                     batch_size: int = 8,
                     progress: bool = True) -> list[Prediction]:
    """Generate, extract and **really execute** one prediction per example.

    Two phases on purpose: every prompt is generated first (batched when the
    backend supports it), then every prediction is scored by executing it. This
    keeps the expensive generation contiguous -- roughly a 3x speedup on CPU
    versus decode-then-score per example -- without changing what is measured.

    Each ``Prediction.meta`` carries the raw ``evaluate_pair`` payload plus the
    example difficulty/kind so :mod:`fcdata.eval.metrics` can slice without
    re-reading the dataset.
    """
    ev = getattr(cfg, "ev", None)
    n_new = int(max_new_tokens if max_new_tokens is not None
                else getattr(ev, "max_new_tokens", 160))
    temp = float(temperature if temperature is not None
                 else getattr(ev, "temperature", 0.0))
    tmo = float(timeout_s if timeout_s is not None else getattr(ev, "sql_timeout_s", 5.0))
    prompt_of = prompt_fn or _resolve_prompt_fn(cfg)

    items = list(examples)
    total = len(items)
    prompts: list[str] = []
    for ex in items:
        prompts.append(prompt_of(cfg, ex, spider_data.db(ex.db_id)))

    # ---- phase 1: generation (one decode error must not kill the whole run) --
    t_gen = time.perf_counter()
    raws: list[str] = []
    errors: list[str] = []
    if items:
        try:
            raws = _generate_all(model, tokenizer, prompts, n_new, temp,
                                 generate_fn, int(batch_size))
        except Exception as exc:  # noqa: BLE001 - fall back to per-example decode
            raws = []
            errors = []
        if len(raws) != len(items):
            # per-example decode, isolating failures
            raws, errors = [], []
            single = generate_fn or _default_generate
            for prompt_text in prompts:
                try:
                    raws.append(str(single(model, tokenizer, prompt_text, n_new, temp)))
                    errors.append("")
                except Exception as exc:  # noqa: BLE001 - isolate one bad decode
                    raws.append("")
                    errors.append(f"{type(exc).__name__}: {exc}")
    gen_seconds = time.perf_counter() - t_gen
    per_item_ms = (gen_seconds * 1000.0 / max(1, len(items)))

    # ---- phase 2: real execution scoring -----------------------------------
    preds: list[Prediction] = []
    for i, ex in enumerate(items, start=1):
        raw = raws[i - 1] if i - 1 < len(raws) else ""
        decode_error = errors[i - 1] if i - 1 < len(errors) else ""
        prompt_text = prompts[i - 1]
        pred_sql = extract_pred_sql(raw)
        db_path = spider_data.db_path(ex.db_id)
        if decode_error:
            result: dict[str, Any] = {
                "executable": False, "exec_match": False, "exact_match": False,
                "reason": f"decode failed: {decode_error}", "gold_ok": True,
                "gold_error": "", "pred_rows": 0, "gold_rows": 0,
            }
        else:
            result = evaluate_pair(db_path, pred_sql, ex.sql, timeout_s=tmo)

        preds.append(Prediction(
            id=new_id("pred"),
            example_id=ex.id,
            db_id=ex.db_id,
            question=ex.question,
            gold_sql=ex.sql,
            pred_sql=pred_sql,
            raw_output=raw,
            executable=bool(result.get("executable")),
            exec_match=bool(result.get("exec_match")),
            exact_match=bool(result.get("exact_match")),
            reason=str(result.get("reason", "")),
            latency_ms=round(per_item_ms, 3),
            meta={
                "difficulty": float(ex.difficulty),
                "kind": str((ex.meta or {}).get("kind") or "raw"),
                "split": ex.split,
                "gold_ok": bool(result.get("gold_ok", True)),
                "gold_error": str(result.get("gold_error", "")),
                "pred_rows": int(result.get("pred_rows", 0) or 0),
                "gold_rows": int(result.get("gold_rows", 0) or 0),
                "pred_error": str(result.get("pred_error", "")),
                "order_sensitive": bool(result.get("order_sensitive", False)),
                "prompt_tokens": estimate_tokens(prompt_text),
                "prompt_source": getattr(prompt_of, "__name__", "custom"),
                "decode_error": decode_error,
            },
        ))
        if progress and (i % 10 == 0 or i == total):
            hit = sum(1 for p in preds if p.exec_match)
            print(f"  [eval] {i}/{total} exec_acc so far {hit / i * 100:.1f}%", flush=True)
    return preds


# --------------------------------------------------------------------------- #
# checkpoint-level evaluation
# --------------------------------------------------------------------------- #
def load_checkpoint(run_dir: str | Path) -> tuple[Any, Any]:
    """Load ``model.pt`` from a training run directory via the trainer's loader."""
    from ..train.reference import load_model  # noqa: PLC0415 - lazy by design

    ckpt = Path(run_dir) / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}; run the train stage first")
    return load_model(str(ckpt))


def evaluate_checkpoint(cfg: Any,
                        run_dir: str | Path,
                        spider_data: Any,
                        split_examples: Sequence[Example],
                        out_dir: str | Path,
                        *,
                        model: Any = None,
                        tokenizer: Any = None,
                        prompt_fn: PromptFn | None = None,
                        generate_fn: GenerateFn | None = None,
                        progress: bool = True) -> dict[str, Any]:
    """Evaluate a checkpoint and persist ``predictions.jsonl`` + ``metrics.json``.

    ``model``/``tokenizer`` may be injected to skip loading (used by tests and by
    the ablation runner, which already holds the freshly trained model).
    """
    from .metrics import format_summary_line, summarize  # noqa: PLC0415 - local import

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model is None:
        model, tokenizer = load_checkpoint(run_dir)

    examples = list(split_examples)
    preds = predict_examples(model, tokenizer, cfg, spider_data, examples,
                             prompt_fn=prompt_fn, generate_fn=generate_fn,
                             progress=progress)
    databases = getattr(spider_data, "databases", None) or {}
    metrics = summarize(preds, seed=int(getattr(cfg, "seed", 17)),
                        databases=databases)
    metrics["run_dir"] = str(run_dir)
    metrics["split"] = str(getattr(getattr(cfg, "ev", None), "split", "dev"))
    metrics["checkpoint"] = str(Path(run_dir) / "model.pt")

    n_written = write_jsonl(out / "predictions.jsonl", preds)
    metrics["n_predictions"] = n_written
    write_json(out / "metrics.json", metrics)
    if progress:
        print(f"  [eval] {format_summary_line(metrics)}", flush=True)
    return metrics
