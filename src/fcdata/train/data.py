"""Turn ``SFTRecord.messages`` into supervised token batches.

Why a separate module
---------------------
The trainer must not depend on the *format* module's internals -- only on the
frozen ``SFTRecord`` contract (``messages`` is a list of
``{"role": ..., "content": ...}`` chat turns, optionally with
``tool_calls`` on the assistant turn). Everything about *how* a chat template
renders those turns belongs to ``fcdata.format``; everything about *which tokens
get a loss* belongs here.

The supervision rule is the heart of SFT:

* the **prompt** is every message before the last assistant turn,
* the **target** is the last assistant turn (for a function-calling record that
  is the serialized ``tool_calls`` JSON, which is where the SQL lives),
* the prompt tokens are labelled ``-100`` so they are excluded from the loss,
* exactly **one** ``<eos>`` is appended so the model learns to stop, and it *is*
  supervised (a model that never emits EOS is unusable at inference).

Two renderings are supported for the assistant turn: if ``fcdata.format`` is
importable and exposes ``render_target_text(record, template)``, that is used
(it knows the chat template's assistant framing); otherwise the record's raw
``content``/``tool_calls`` are serialised here with a plain canonical format.
The fallback exists so the trainer is testable in isolation and so the pipeline
degrades gracefully instead of ImportError-ing.

Batching: sequences are padded to the longest row in the batch. ``labels`` uses
``-100`` for both prompt tokens *and* the padding, so neither contributes to the
loss or to token accuracy. The attention mask is not passed to the model because
the reference model's loss already ignores padding; the causal mask only needs
positions, and a pad token after the real tokens cannot influence earlier
positions under a causal mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from ..records import SFTRecord
from .vocab import EOS_ID, PAD_ID, WordTokenizer

IGNORE_INDEX = -100

#: Fallback assistant framing when ``fcdata.format`` cannot be imported.
_DEFAULT_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
_DEFAULT_TURN_END = "<|im_end|>"


# --------------------------------------------------------------------------- #
# message helpers
# --------------------------------------------------------------------------- #
def _message_text(message: dict[str, Any]) -> str:
    """Render one chat message as plain text, including any tool calls.

    The assistant function call is what the model must learn to emit, so it is
    serialised *deterministically* here (``json.dumps`` with sorted keys) -- a
    stable target is a learnable target.
    """
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args: Any = json.loads(raw_args)
            except json.JSONDecodeError:
                args = raw_args
        else:
            args = raw_args if raw_args is not None else {}
        parts.append(json.dumps(
            {"tool_calls": [{"type": "function",
                             "function": {"name": name, "arguments": args}}]},
            ensure_ascii=False, sort_keys=True, separators=(",", ": "),
        ))
    return "\n".join(parts)


def split_prompt_target(messages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]],
                                                                    dict[str, Any] | None]:
    """Split a chat into ``(prompt_messages, last_assistant_message)``.

    Raises ``ValueError`` when there is no assistant turn: such a record teaches
    nothing (there is no label), and silently skipping it would hide a data bug.
    """
    if not messages:
        raise ValueError("SFTRecord.messages is empty")
    last_assistant = -1
    for i, msg in enumerate(messages):
        if str(msg.get("role", "")) == "assistant":
            last_assistant = i
    if last_assistant < 0:
        raise ValueError("SFTRecord.messages contains no assistant turn to learn from")
    return list(messages[:last_assistant]), dict(messages[last_assistant])


def _try_format_api() -> dict[str, Any] | None:
    """Return the parts of ``fcdata.format`` the trainer needs, or ``None``.

    The trainer must not hard-depend on the format module (it may not exist yet,
    and unit tests construct ``SFTRecord`` objects by hand), so the import is
    guarded and the whole rendering contract is probed once. When present, the
    format module is the single source of truth for chat markup -- in particular
    ``render_supervised_pair`` guarantees ``prompt + target`` is exactly
    ``render_messages(...)``, which is what makes the label mask exact.
    """
    try:
        from .. import format as fmt  # type: ignore[attr-defined]
    except ImportError:
        return None
    if not hasattr(fmt, "render_supervised_pair"):
        return None
    return {
        "pair": getattr(fmt, "render_supervised_pair", None),
        "target": getattr(fmt, "render_target_text", None),
        "prompt": getattr(fmt, "render_record_prompt", None),
        "kind_weight": getattr(fmt, "kind_weight", None),
    }


_FORMAT_API = _try_format_api()


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_prompt_text(record: SFTRecord, *, template: str = "qwen") -> str:
    """Render the prompt side of a record (everything before the answer).

    Prefers ``fcdata.format.render_record_prompt`` so the chat markup is
    identical to what the evaluator builds at inference time -- a train/eval
    prompt mismatch is the most common cause of a model that "learns" but scores
    zero. Falls back to a plain rendering when the format module is absent.
    """
    if _FORMAT_API and _FORMAT_API["prompt"] is not None:
        try:
            rendered = _FORMAT_API["prompt"](record, template)
            if isinstance(rendered, str) and rendered:
                return rendered
        except (TypeError, ValueError, KeyError):
            pass
    prompt_msgs, _ = split_prompt_target(record.messages)
    body = "\n".join(
        f"<|im_start|>{m.get('role', 'user')}\n{_message_text(m)}{_DEFAULT_TURN_END}"
        for m in prompt_msgs
    )
    return f"{body}\n{_DEFAULT_ASSISTANT_PREFIX}"


def render_target_text(record: SFTRecord, template: str = "qwen") -> str:
    """Render the assistant answer that receives the loss.

    Prefers ``fcdata.format.render_target_text`` (which appends the template's
    end-of-turn token, so the model learns to stop). Falls back to a canonical
    tool-call serialisation when the format module is absent.
    """
    if _FORMAT_API and _FORMAT_API["target"] is not None:
        try:
            rendered = _FORMAT_API["target"](record, template)
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except (TypeError, ValueError, KeyError):
            pass
    _, target = split_prompt_target(record.messages)
    text = _message_text(target or {})
    if not text:
        raise ValueError(f"SFTRecord {record.id!r} has an empty assistant target")
    return text + _DEFAULT_TURN_END


def render_pair(record: SFTRecord, *, template: str = "qwen") -> tuple[str, str]:
    """Return ``(prompt_text, target_text)`` from one consistent rendering.

    Uses ``fcdata.format.render_supervised_pair`` when available: it guarantees
    ``prompt + target == render_messages(record.messages)`` and that ``target`` is
    a literal suffix of the full render, so encoding the pair cannot drift from
    what the evaluator will feed the model. Without the format module the two
    halves are rendered independently (still consistent, because the fallback
    prompt ends with the assistant header and the target carries the body).
    """
    if _FORMAT_API and _FORMAT_API["pair"] is not None:
        try:
            prompt, target = _FORMAT_API["pair"](record, template)
            if isinstance(prompt, str) and isinstance(target, str) and target.strip():
                return prompt, target
        except (TypeError, ValueError, KeyError):
            pass
    return render_prompt_text(record, template=template), render_target_text(
        record, template=template)


def record_weight(record: SFTRecord) -> float:
    """Sampling weight for a record, defaulting to its ``SFTRecord.weight``.

    ``fcdata.format`` may compute the weight from the record's kind
    (``cfg.fmt.hard_negative_weight`` etc.). The record field is authoritative
    when it is not the default 1.0, because the format stage already resolved it;
    otherwise the format module's kind-weight helper is consulted.
    """
    weight = float(getattr(record, "weight", 1.0) or 1.0)
    if weight != 1.0:
        return weight
    kind_weight = _FORMAT_API.get("kind_weight") if _FORMAT_API else None
    if callable(kind_weight):
        try:
            resolved = float(kind_weight(getattr(record, "kind", "")))
            if resolved > 0:
                return resolved
        except (TypeError, ValueError):
            pass
    return weight


@dataclass
class EncodedRecord:
    """One supervised example: ``input_ids`` plus labels aligned to them."""
    input_ids: list[int]
    labels: list[int]
    prompt_len: int
    target_len: int
    record_id: str = ""
    kind: str = ""
    target_truncated: bool = False

    @property
    def n_supervised(self) -> int:
        return sum(1 for lab in self.labels if lab != IGNORE_INDEX)


def encode_record(record: SFTRecord, tokenizer: WordTokenizer, *,
                  max_seq_len: int, template: str = "qwen") -> EncodedRecord:
    """Tokenize one record into ``(input_ids, labels)``.

    Layout: ``<bos> prompt_tokens target_tokens <eos>``. Labels are ``-100`` over
    ``<bos>`` and the prompt, and the real ids over the target plus the final
    ``<eos>``.

    The prompt is tokenized separately from the target and the two id lists are
    concatenated, rather than tokenizing ``prompt + target`` once and splitting
    by character offset. That is deliberate: word-level tokenization is *not*
    concatenation-invariant (``"assistant\\n{\\"`` splits differently from
    ``"assistant\\n"`` + ``"{\\"``), so a character-offset split could silently
    shift the mask by a token and supervise the wrong span. Tokenizing the two
    halves independently makes the boundary exact by construction.

    If the sequence is too long the **prompt** is truncated from the left
    (keeping ``<bos>``) -- never the target, because the target is the only thing
    being learned. A target that alone exceeds the budget is a data problem, and
    is reported by ``EncodedRecord.target_truncated`` instead of being silently
    chopped mid-JSON.

    The final length is **guaranteed** to be ``<= max_seq_len``: ``<bos>`` is
    reserved *before* the target is truncated, so a target that fills the whole
    budget still leaves room for it. Getting this wrong produced a 257-token
    sequence for ``max_seq_len=256`` (the model's ``forward`` then rejected it),
    which is why the length is asserted at the end rather than assumed.
    """
    prompt_text, target_text = render_pair(record, template=template)

    budget = int(max_seq_len)
    if budget < 2:
        raise ValueError(f"max_seq_len must be >= 2 (need <bos> + one target token), "
                         f"got {budget}")

    prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    target_ids = tokenizer.encode(target_text, add_eos=True)
    target_truncated = False

    # Reserve one slot for <bos> *before* truncating the target: the target must
    # never be the reason the sequence overflows, but it also must not squeeze out
    # the BOS token.
    max_target = budget - 1
    if len(target_ids) > max_target:
        # pathological target: keep the head (it names the tool) and one EOS
        target_ids = target_ids[: max_target - 1] + [EOS_ID]
        target_truncated = True
    room = budget - len(target_ids)          # >= 1 by construction
    if len(prompt_ids) > room:
        if room <= 1:
            prompt_ids = prompt_ids[:1]
        else:
            prompt_ids = prompt_ids[:1] + prompt_ids[1:][-(room - 1):]

    input_ids = prompt_ids + target_ids
    if len(input_ids) > budget:              # pragma: no cover - defensive
        raise AssertionError(
            f"internal truncation bug: produced {len(input_ids)} tokens for "
            f"max_seq_len={budget} (prompt {len(prompt_ids)} + target {len(target_ids)})")
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
    return EncodedRecord(
        input_ids=input_ids, labels=labels,
        prompt_len=len(prompt_ids), target_len=len(target_ids),
        record_id=getattr(record, "id", ""), kind=getattr(record, "kind", ""),
        target_truncated=target_truncated,
    )


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    """A padded batch of supervised examples."""
    input_ids: torch.Tensor   # (B, T) long
    labels: torch.Tensor      # (B, T) long, -100 where unsupervised
    weights: torch.Tensor     # (B,) float, per-record sampling weight
    n_supervised: int

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(self.input_ids.to(device), self.labels.to(device),
                     self.weights.to(device), self.n_supervised)


def collate(encoded: Sequence[EncodedRecord], weights: Sequence[float] | None = None,
            *, pad_id: int = PAD_ID, max_seq_len: int = 0) -> Batch:
    """Pad a list of encoded records into a batch.

    Padding tokens get label ``-100`` (never ``pad_id``) -- otherwise the model
    would be trained to predict padding, which is the second classic masking bug.

    ``max_seq_len``, when given, is a hard guard: a row longer than the model's
    capacity is a *data pipeline* bug, and it is much easier to diagnose here
    (with the offending record's id and length) than as a shape error inside the
    model's ``forward``. Pass the model's ``max_seq_len`` to enable it.
    """
    if not encoded:
        raise ValueError("cannot collate an empty list of encoded records")
    width = max(len(e.input_ids) for e in encoded)
    if max_seq_len and width > max_seq_len:
        offenders = [(e.record_id, len(e.input_ids)) for e in encoded
                     if len(e.input_ids) > max_seq_len][:5]
        raise ValueError(
            f"encoded sequence length {width} exceeds max_seq_len={max_seq_len}; "
            f"offending records (id, len): {offenders}. "
            "encode_record() must clamp to max_seq_len -- this indicates a bug in "
            "the truncation path, not a data problem.")
    b = len(encoded)
    input_ids = torch.full((b, width), pad_id, dtype=torch.long)
    labels = torch.full((b, width), IGNORE_INDEX, dtype=torch.long)
    for row, enc in enumerate(encoded):
        n = len(enc.input_ids)
        input_ids[row, :n] = torch.tensor(enc.input_ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(enc.labels, dtype=torch.long)
    w = torch.ones(b, dtype=torch.float32)
    if weights is not None:
        if len(weights) != b:
            raise ValueError(f"got {len(weights)} weights for {b} records")
        w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    return Batch(input_ids, labels, w, sum(e.n_supervised for e in encoded))


def iter_batches(items: Sequence[Any], batch_size: int, *, shuffle: bool = False,
                 seed: int = 0) -> Iterable[list[Any]]:
    """Yield lists of ``batch_size`` items, optionally shuffled deterministically."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(range(len(items)))
    if shuffle:
        import random
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[i] for i in order[start:start + batch_size]]


def build_encoded_dataset(records: Sequence[SFTRecord], tokenizer: WordTokenizer, *,
                          max_seq_len: int, template: str = "qwen",
                          max_records: int = 0) -> tuple[list[EncodedRecord], list[float]]:
    """Encode a split once, up front (cheap: pure Python, no torch ops).

    Records whose target cannot be rendered are skipped with a counted warning
    rather than aborting a long training run -- but at least one record must
    survive, otherwise the caller gets a clear ``ValueError``.
    """
    subset = list(records[:max_records]) if max_records and max_records > 0 else list(records)
    encoded: list[EncodedRecord] = []
    weights: list[float] = []
    skipped = 0
    truncated = 0
    for rec in subset:
        try:
            enc = encode_record(rec, tokenizer, max_seq_len=max_seq_len, template=template)
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        if enc.n_supervised <= 0:
            skipped += 1
            continue
        if enc.target_truncated:
            truncated += 1
        encoded.append(enc)
        weights.append(record_weight(rec))
    if not encoded:
        raise ValueError(
            f"no usable SFT records out of {len(subset)} (skipped {skipped}); "
            "check that messages contain an assistant turn with non-empty content")
    if skipped:
        print(f"[train] warning: skipped {skipped} unrenderable record(s) of {len(subset)}")
    if truncated:
        # Loud on purpose: a truncated target makes the task partly unlearnable and
        # would show up later as a flat accuracy curve with no obvious cause.
        print(f"[train] WARNING: {truncated} record(s) had a target longer than "
              f"max_seq_len={max_seq_len} and were truncated; "
              "reduce cfg.fmt.max_schema_tables or raise cfg.train.max_seq_len")
    return encoded, weights


def corpus_texts(records: Sequence[SFTRecord], *, template: str = "qwen") -> list[str]:
    """All prompt+target text of a split -- the tokenizer's fitting corpus.

    Uses :func:`render_pair` so the corpus is byte-identical to what will later
    be encoded. Fitting on differently-rendered text would create vocabulary
    holes (e.g. a template marker that only appears at encode time).
    """
    texts: list[str] = []
    for rec in records:
        try:
            prompt, target = render_pair(rec, template=template)
        except (ValueError, KeyError, TypeError):
            continue
        texts.append(prompt)
        texts.append(target)
    return texts
