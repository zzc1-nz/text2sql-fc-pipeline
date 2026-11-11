"""Shared plumbing for the ablation and data-scale runners.

WHY this module exists
----------------------
The ablation ("does CoT / hard-negative synthesis help?") and the scale curve
("how much synthetic data do we need?") are the same experiment with a different
independent variable. Rather than duplicating the load/train/eval/account dance
twice -- and risking the two runners reporting token costs differently -- both
call the helpers here.

Design rules encoded below
--------------------------
* **The SFT corpus is the single source of truth.** Both runners slice
  ``artifacts/04_sft/train.jsonl``; if that file does not exist yet they rebuild
  it through ``fcdata.format.build_sft_records`` so the runner is usable during
  development without a full pipeline run.
* **Token investment is attributable.** A config's synthesis cost is the sum of
  ``SFTRecord.synth_tokens`` over exactly the records that config trains on; the
  USD figure comes from the real synth ledger when present, otherwise from the
  configured per-million-token prices applied to the same token counts.
* **Training and evaluation go through the public APIs** (``fcdata.train.train``,
  ``fcdata.train.reference.load_model``, ``fcdata.eval.infer.evaluate_checkpoint``)
  so the runners exercise the same code path as the CLI.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..records import Example, SFTRecord, read_json, read_sft, write_json
from ..tokens import TokenLedger, load_ledger_jsonl

#: Kinds a config may include; ``raw`` records are plain Spider examples.
ALL_KINDS: tuple[str, ...] = ("raw", "cot", "paraphrase", "hard_negative")

#: Records are duplicated at most this many times by difficulty resampling, so
#: one very hard example cannot swamp the batch budget.
MAX_RESAMPLE_FACTOR = 3


# --------------------------------------------------------------------------- #
# SFT corpus loading
# --------------------------------------------------------------------------- #
@dataclass
class SFTBundle:
    """The SFT corpus both runners slice, plus provenance for the report."""

    train: list[SFTRecord] = field(default_factory=list)
    dev: list[SFTRecord] = field(default_factory=list)
    source: str = ""
    eval_examples: list[Example] = field(default_factory=list)

    def kind_counts(self) -> dict[str, int]:
        counts = {k: 0 for k in ALL_KINDS}
        for r in self.train:
            counts[r.kind] = counts.get(r.kind, 0) + 1
        return counts

    @property
    def synth_tokens(self) -> int:
        return sum(int(r.synth_tokens) for r in self.train)


def load_sft_bundle(cfg: Any, spider_data: Any) -> SFTBundle:
    """Load (or rebuild) the SFT train/dev corpora.

    Preference: on-disk ``04_sft/train.jsonl`` (produced by the format stage).
    Fallback: rebuild from ``03_validated/validated.jsonl`` +
    ``01_raw/*.jsonl`` through ``fcdata.format.build_sft_records``.
    """
    sft_dir = Path(cfg.paths.sft_dir)
    train_path = sft_dir / "train.jsonl"
    dev_path = sft_dir / "dev.jsonl"
    if train_path.exists():
        train = read_sft(train_path)
        dev = read_sft(dev_path) if dev_path.exists() else []
        return SFTBundle(train=train, dev=dev, source=str(train_path),
                         eval_examples=_eval_examples(cfg, spider_data))
    return _rebuild_sft(cfg, spider_data)


def _rebuild_sft(cfg: Any, spider_data: Any) -> SFTBundle:
    """Rebuild SFT records from validated records (or raw examples) if needed."""
    from ..format import build_sft_records  # noqa: PLC0415 - lazy by design

    validated_path = Path(cfg.paths.validated_dir) / "validated.jsonl"
    raw_train = _read_raw_examples(cfg, "train") or list(spider_data.train)
    raw_dev = _read_raw_examples(cfg, "dev") or list(spider_data.dev)

    if validated_path.exists():
        from ..records import read_validated  # noqa: PLC0415 - local import

        validated = read_validated(validated_path)
        train, stats = build_sft_records(cfg, validated, spider_data, raw_train)
        dev = []
        source = str(validated_path)
    else:
        # No validation artifacts: build straight from raw examples so the
        # runner still works. Clearly labelled as such in the report.
        train, stats = build_sft_records(cfg, [], spider_data, raw_train)
        dev = []
        source = "raw_examples (validated.jsonl missing)"
    if not dev and raw_dev:
        dev_records, _ = build_sft_records(cfg, [], spider_data, raw_dev)
        dev = dev_records
    return SFTBundle(train=train, dev=dev, source=source,
                     eval_examples=_eval_examples(cfg, spider_data))


def _read_raw_examples(cfg: Any, split: str) -> list[Example]:
    from ..records import read_jsonl  # noqa: PLC0415 - local import

    path = Path(cfg.paths.raw_dir) / f"{split}.jsonl"
    if not path.exists():
        return []
    return [Example(**d) for d in read_jsonl(path)]


def _eval_examples(cfg: Any, spider_data: Any) -> list[Example]:
    """Examples the trained model is scored on (dev split, capped by config)."""
    split = str(getattr(getattr(cfg, "ev", None), "split", "dev") or "dev")
    examples = list(getattr(spider_data, split, []) or [])
    if not examples:
        examples = list(getattr(spider_data, "dev", []) or [])
    limit = int(getattr(getattr(cfg, "ev", None), "max_examples", 0) or 0)
    return examples[:limit] if limit > 0 else examples


# --------------------------------------------------------------------------- #
# slicing / resampling
# --------------------------------------------------------------------------- #
def select_kinds(records: Sequence[SFTRecord], kinds: Iterable[str]) -> list[SFTRecord]:
    """Keep only records whose ``kind`` is in ``kinds`` (order preserved)."""
    wanted = set(kinds)
    return [r for r in records if r.kind in wanted]


def resample_by_difficulty(records: Sequence[SFTRecord],
                           seed: int = 17,
                           max_factor: int = MAX_RESAMPLE_FACTOR) -> list[SFTRecord]:
    """Difficulty-weighted resampling for hard-negative-heavy configs.

    Each record is repeated ``round(weight)`` times, where ``weight`` is
    ``SFTRecord.weight`` scaled by difficulty for hard negatives. Repetition (as
    opposed to loss weighting) is used because the reference trainer consumes a
    flat record list; it is deterministic for a fixed seed and capped by
    ``max_factor`` so no single example can dominate.

    The returned list is shuffled deterministically, so two runs with the same
    seed train on the same order.
    """
    rng = random.Random(seed)
    out: list[SFTRecord] = []
    for rec in records:
        weight = float(getattr(rec, "weight", 1.0) or 1.0)
        if rec.kind == "hard_negative":
            weight *= 1.0 + float(getattr(rec, "difficulty", 0.0) or 0.0)
        elif rec.kind == "cot":
            weight *= 1.0 + 0.25 * float(getattr(rec, "difficulty", 0.0) or 0.0)
        factor = int(round(weight))
        factor = max(1, min(int(max_factor), factor))
        out.extend([rec] * factor)
    rng.shuffle(out)
    return out


def synth_tokens_of(records: Sequence[SFTRecord]) -> int:
    """Synthesis tokens attributable to a set of SFT records."""
    return sum(int(getattr(r, "synth_tokens", 0) or 0) for r in records)


# --------------------------------------------------------------------------- #
# synthesis cost accounting
# --------------------------------------------------------------------------- #
def synth_cost_by_kind(cfg: Any) -> dict[str, dict[str, Any]]:
    """Tokens and USD spent per synthesis kind, from the real ledger.

    Reads every ledger JSONL the pipeline writes and aggregates the ``task``
    field (which the synth stage sets to the record kind). When no ledger exists
    (e.g. a stubbed test run) the result is empty and callers fall back to the
    configured price applied to ``SFTRecord.synth_tokens``.
    """
    candidates = [
        Path(cfg.paths.synth_dir) / "ledger.jsonl",
        Path(cfg.paths.validated_dir) / "ledger.jsonl",
        Path(cfg.paths.resolve(cfg.paths.artifacts_dir)) / "ledger.jsonl",
    ]
    ledger = TokenLedger(None)
    found = False
    for path in candidates:
        if not path.exists():
            continue
        found = True
        for rec in load_ledger_jsonl(path):
            ledger.record(rec)
    if not found:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rec in ledger.records:
        kind = rec.task or "unknown"
        entry = out.setdefault(kind, {"calls": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "total_tokens": 0,
                                      "cost_usd": 0.0, "cached_calls": 0})
        entry["calls"] += 1
        entry["cached_calls"] += int(rec.cached)
        entry["prompt_tokens"] += int(rec.prompt_tokens)
        entry["completion_tokens"] += int(rec.completion_tokens)
        entry["total_tokens"] += int(rec.total_tokens)
        entry["cost_usd"] += float(rec.cost_usd)
    for entry in out.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return out


def estimate_cost_usd(cfg: Any, total_tokens: int) -> float:
    """Blended USD estimate for ``total_tokens`` of cheap-model synthesis.

    Used only when no ledger exists; the report marks such rows as estimated.
    Assumes the configured input/output split of a synthesis call, which is
    dominated by the prompt (DDL + gold SQL + question).
    """
    if total_tokens <= 0:
        return 0.0
    prov = cfg.provider("cheap")
    # typical synthesis call: ~80% prompt tokens, ~20% completion
    prompt = int(total_tokens * 0.8)
    completion = total_tokens - prompt
    usd = (prompt / 1_000_000.0) * prov.price_in_per_mtok + (
        completion / 1_000_000.0) * prov.price_out_per_mtok
    return round(usd, 6)


def cost_for_kinds(cfg: Any,
                   kinds: Iterable[str],
                   records: Sequence[SFTRecord],
                   cost_by_kind: dict[str, dict[str, Any]] | None) -> tuple[float, bool]:
    """USD cost for a config plus a flag saying whether it came from the ledger.

    Returns ``(cost_usd, from_ledger)``. ``raw`` records cost nothing to
    synthesize (they are the original dataset), which is the honest accounting.

    The ledger total for a kind is the cost of synthesizing the *whole* corpus of
    that kind, so it is prorated by the share of that kind's records the config
    actually trains on. Without proration every point of the data-scale curve
    would report the full-corpus cost and the cost axis would be flat -- the one
    thing that curve exists to measure. Proration is by record count, which is
    exact when records of a kind have similar prompt sizes and is stated in the
    report as an apportionment.
    """
    kinds_set = set(kinds)
    synth_kinds = kinds_set - {"raw"}
    if not synth_kinds:
        return (0.0, True)
    if cost_by_kind:
        total = 0.0
        covered = True
        for kind in sorted(synth_kinds):
            entry = cost_by_kind.get(kind)
            if entry is None:
                covered = False
                continue
            ledger_records = int(entry.get("calls", 0) or 0)
            used = sum(1 for r in records if r.kind == kind)
            share = (used / ledger_records) if ledger_records > 0 else 1.0
            total += float(entry.get("cost_usd", 0.0)) * min(1.0, share)
        if covered:
            return (round(total, 6), True)
    return (estimate_cost_usd(cfg, synth_tokens_of(records)), False)


# --------------------------------------------------------------------------- #
# quick mode
# --------------------------------------------------------------------------- #
def quick_cfg(cfg: Any, *, epochs: int = 1, eval_examples: int = 8) -> Any:
    """Copy of ``cfg`` with training/eval shrunk so a smoke run finishes fast.

    Used by the ``--quick`` CLI path and by tests. Returns a deep copy so the
    caller's config is never mutated. Also stamps ``train.max_steps`` /
    ``train.max_records`` when the config supports them, because the trainer
    reads those as a config-driven fallback for hard step/record caps -- that is
    what keeps a ``--quick`` ablation from silently doing a full epoch per rung.

    Decoding is shortened too: a random-init model rarely emits ``<eos>``, so
    ``max_new_tokens`` dominates smoke-run wall time. Shrinking it changes only
    how much the (deliberately untrained) model rambles, never how the score is
    computed.
    """
    out = copy.deepcopy(cfg)
    out.train.epochs = max(1, int(epochs))
    out.train.eval_every = 0
    out.train.log_every = max(1, int(out.train.log_every))
    out.ev.max_examples = max(1, int(eval_examples))
    out.ev.max_new_tokens = min(int(getattr(out.ev, "max_new_tokens", 160)),
                                QUICK_MAX_NEW_TOKENS)
    out.ablation.dev_sample = max(1, int(eval_examples))
    # optional knobs: only set them when the TrainCfg dataclass has the field
    for name, value in (("max_steps", QUICK_MAX_STEPS), ("max_records", QUICK_MAX_RECORDS)):
        if hasattr(out.train, name):
            setattr(out.train, name, int(value))
    return out


#: Hard caps used by ``quick_cfg``; small enough for a ~2s CPU smoke train.
QUICK_MAX_STEPS = 5
QUICK_MAX_RECORDS = 64
#: Smoke runs decode at most this many tokens (a trained model needs far fewer).
QUICK_MAX_NEW_TOKENS = 48


# --------------------------------------------------------------------------- #
# train + evaluate (the experiment body)
# --------------------------------------------------------------------------- #
def train_and_evaluate(cfg: Any,
                       spider_data: Any,
                       train_records: Sequence[SFTRecord],
                       dev_records: Sequence[SFTRecord],
                       eval_examples: Sequence[Example],
                       work_dir: str | Path,
                       *,
                       train_fn: Callable[..., dict[str, Any]] | None = None,
                       load_fn: Callable[..., tuple[Any, Any]] | None = None,
                       generate_fn: Callable[..., str] | None = None,
                       prompt_fn: Callable[..., str] | None = None,
                       progress: bool = True) -> dict[str, Any]:
    """Train one model on ``train_records`` and evaluate it on ``eval_examples``.

    Returns a row dict with the train metrics, the *real* eval metrics, and the
    directories used. The model is reloaded from its checkpoint before eval so
    the save/load path is exercised on every experiment -- an ablation number is
    therefore always the accuracy of a model that was actually written to disk
    and read back.
    """
    from .infer import evaluate_checkpoint  # noqa: PLC0415 - local import

    wd = Path(work_dir)
    train_dir = wd / "train"
    eval_dir = wd / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if train_fn is None:
        from ..train import train as _train  # noqa: PLC0415 - lazy: pulls in torch

        train_fn = _train

    # Each experiment arm has a different record count, so the epoch count that
    # makes ``max_steps`` reachable differs per arm. Deriving it here (instead of
    # once in the caller) is what stops a small arm -- e.g. raw_only with 98
    # records at 7 steps/epoch -- from silently finishing at 28 steps when 800
    # were requested. ``train()`` stops at whichever cap comes first.
    arm_cfg = _cfg_with_reachable_steps(cfg, len(train_records))

    t0 = _now()
    train_metrics = _call_train(train_fn, arm_cfg, list(train_records),
                                list(dev_records), str(train_dir))
    train_seconds = round(_now() - t0, 2)

    if load_fn is None:
        from ..train.reference import load_model as _load  # noqa: PLC0415 - lazy

        load_fn = _load
    model, tokenizer = load_fn(str(train_dir / "model.pt"))

    eval_metrics = evaluate_checkpoint(
        cfg, train_dir, spider_data, eval_examples, eval_dir,
        model=model, tokenizer=tokenizer, prompt_fn=prompt_fn,
        generate_fn=generate_fn, progress=progress)

    return {
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "train_seconds": train_seconds,
        "train_metrics": train_metrics,
        "eval": eval_metrics,
    }


def _cfg_with_reachable_steps(cfg: Any, n_records: int) -> Any:
    """Return ``cfg`` with ``epochs`` high enough that ``max_steps`` can be hit.

    ``train()`` stops at whichever comes first, ``max_steps`` or the epoch count.
    A caller who sets ``max_steps=800`` on a 98-record arm (7 steps/epoch) would
    otherwise stop at 28 steps and quietly under-train that arm relative to the
    others -- which would corrupt the ablation comparison, since the arms would
    differ in *training budget* as well as in data mix. That is precisely the
    confound an ablation exists to remove, so the epoch count is derived from the
    arm's own record count here.

    Returns the original config unchanged when no step cap is set (so a
    fully-epoch-driven run keeps the user's epoch count).
    """
    max_steps = int(getattr(cfg.train, "max_steps", 0) or 0)
    if max_steps <= 0:
        return cfg
    batch = max(1, int(getattr(cfg.train, "batch_size", 1) or 1))
    records = max(1, int(n_records))
    steps_per_epoch = max(1, (records + batch - 1) // batch)
    needed = (max_steps // steps_per_epoch) + 1
    if int(getattr(cfg.train, "epochs", 1) or 1) >= needed:
        return cfg
    out = copy.deepcopy(cfg)
    out.train.epochs = needed
    return out


def _call_train(train_fn: Callable[..., dict[str, Any]], cfg: Any,                train_records: list[SFTRecord], dev_records: list[SFTRecord],
                out_dir: str) -> dict[str, Any]:
    """Call the trainer, passing quick-mode caps only when it accepts them.

    The trainer's contract is ``train(cfg, sft_train, sft_dev, out_dir)``; it
    optionally accepts ``max_steps`` / ``max_records`` keyword caps. Probing the
    signature keeps this module compatible with both versions instead of
    hard-failing on a ``TypeError``.
    """
    import inspect  # noqa: PLC0415 - only needed on this path

    try:
        params = inspect.signature(train_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        params = {}
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
    kwargs: dict[str, Any] = {}
    for name in ("max_steps", "max_records"):
        if accepts_kwargs or name in params:
            value = int(getattr(cfg.train, name, 0) or 0)
            if value > 0:
                kwargs[name] = value
    if kwargs:
        return train_fn(cfg, train_records, dev_records, out_dir, **kwargs)
    return train_fn(cfg, train_records, dev_records, out_dir)


def _now() -> float:
    import time  # noqa: PLC0415 - trivial, keeps module import light

    return time.perf_counter()


# --------------------------------------------------------------------------- #
# artifact writing
# --------------------------------------------------------------------------- #
def write_rows_json_csv(rows: Sequence[dict[str, Any]],
                        json_path: str | Path,
                        csv_path: str | Path,
                        columns: Sequence[str],
                        payload: dict[str, Any] | None = None) -> None:
    """Write the JSON payload and a CSV with a fixed column order."""
    import csv  # noqa: PLC0415 - stdlib, imported at call time

    write_json(json_path, payload if payload is not None else {"rows": list(rows)})
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def read_json_if_exists(path: str | Path) -> Any:
    """``read_json`` that tolerates a missing file (returns ``None``)."""
    return read_json(path, None)
