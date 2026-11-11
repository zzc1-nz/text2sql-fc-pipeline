"""Format stage -- validated records -> function-calling chat SFT corpus.

WHY this stage exists
---------------------
The pipeline's product is a *function-calling* dataset, so the conversion from
"SQL string" to "chat messages with a tool call" is not plumbing, it is the
deliverable.  This module owns the corpus-level decisions that
:mod:`fcdata.format.chat` (which owns the syntax) should not make:

* **Which kinds enter the corpus and with what weight.**  Hard negatives are
  scarce and high-signal, so they get a resampling weight
  (``cfg.fmt.hard_negative_weight``, default 1.5) instead of being duplicated --
  the trainer samples by weight, the dataset stays honest about its composition.
* **The raw Spider half.**  ``kind="raw"`` records are included so the
  ``raw_only`` ablation is a genuine baseline: same tokenizer, same templates,
  same training loop, only the synthesis differs.
* **Token attribution.**  Every synthetic record carries the token investment
  that produced it (``SFTRecord.synth_tokens`` from ``SynthRecord.total_tokens``,
  0 for raw), which is what makes the "tokens in -> capability out" curve
  computable at record granularity.
* **A self-check on every target.**  :func:`fcdata.format.chat.validate_target`
  is run over the whole corpus while building it; ``stats`` reports the
  tool-call validity rate, so a formatting regression shows up as a number in the
  artifact instead of as a silently degraded training run.

Determinism: record ids are content hashes, so the same inputs always produce
byte-identical JSONL (required for the ablation/scale comparisons to be
reproducible).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ..records import (Example, SFTRecord, SynthRecord, ValidatedRecord, write_json,
                       write_jsonl)
from ..spider import SpiderData
from ..tokens import estimate_messages_tokens

from .chat import (DEFAULT_TEMPLATE, DEFAULT_TOOL_NAME, TEMPLATES, assistant_sql,
                   build_assistant_message, build_eval_messages, build_eval_prompt,
                   build_raw_messages, build_record_messages, build_system_message,
                   build_user_message, derive_plan, parse_tool_call, render_messages,
                   render_prompt, render_record_prompt, render_supervised_pair,
                   render_target_text, sanitize_reasoning, tool_definitions, tool_name,
                   tool_schema, validate_target)

__all__ = [
    # syntax / templates
    "TEMPLATES", "DEFAULT_TEMPLATE", "DEFAULT_TOOL_NAME",
    "tool_schema", "tool_definitions", "tool_name",
    "build_system_message", "build_user_message", "build_assistant_message",
    "render_messages", "render_prompt", "render_target_text", "render_record_prompt",
    "render_supervised_pair",
    "parse_tool_call", "assistant_sql", "validate_target",
    "sanitize_reasoning", "derive_plan",
    # inference prompts (shared with fcdata.eval)
    "build_eval_messages", "build_eval_prompt", "build_record_messages", "build_raw_messages",
    # corpus
    "build_sft_records", "build_and_write", "split_train_dev", "hard_negative_weight",
    "kind_weight",
]


# --------------------------------------------------------------------------- #
# ids + weights
# --------------------------------------------------------------------------- #
def _sft_id(kind: str, source_id: str, sql: str, template: str) -> str:
    digest = hashlib.sha1(f"{kind}|{source_id}|{sql}|{template}".encode("utf-8")).hexdigest()
    return f"sft_{digest[:16]}"


def hard_negative_weight(cfg: Any) -> float:
    """Resampling weight for hard negatives (``cfg.fmt.hard_negative_weight``).

    Weighting -- rather than duplicating records -- is how the
    "+hard-negative resampling" ablation arm differs from "+CoT": the corpus
    composition stays auditable while the sampling pressure changes.
    """
    return float(cfg.fmt.hard_negative_weight)


def kind_weight(cfg: Any, kind: str) -> float:
    """Sampling weight for a synthetic kind (raw is always 1.0).

    Explicit per-kind weights let the ablation arms be defined purely in config:
    ``plus_cot`` and ``plus_hard_negatives`` differ only by the presence of
    hard negatives *and* their weight, not by a hidden constant in this module.
    """
    if kind == "hard_negative":
        return float(cfg.fmt.hard_negative_weight)
    if kind == "paraphrase":
        return float(cfg.fmt.paraphrase_weight)
    if kind == "cot":
        return float(cfg.fmt.cot_weight)
    return 1.0


# --------------------------------------------------------------------------- #
# one record
# --------------------------------------------------------------------------- #
def _make_sft_record(cfg: Any, record: Any, db: Any, *, db_path: Any,
                     template: str, tools: list[dict[str, Any]],
                     kind: str, split: str, weight: float,
                     synth_tokens: int, meta: dict[str, Any]) -> SFTRecord:
    messages = build_record_messages(cfg, record, db, db_path=db_path, template=template)
    target_sql = str(getattr(record, "sql", "") or "")
    source_id = str(getattr(record, "source_id", "") or getattr(record, "id", "") or "")
    return SFTRecord(
        id=_sft_id(kind, source_id, target_sql, template),
        source_id=source_id,
        kind=kind,
        db_id=str(getattr(record, "db_id", "") or ""),
        messages=messages,
        tools=tools,
        target_sql=target_sql,
        split=split,
        weight=float(weight),
        difficulty=float(getattr(record, "difficulty", 0.0) or 0.0),
        synth_tokens=int(synth_tokens),
        meta=meta,
    )


def _db_path_for(spider_data: SpiderData | None, db_id: str) -> Path | None:
    if spider_data is None:
        return None
    try:
        return spider_data.db_path(db_id)
    except (FileNotFoundError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def build_sft_records(cfg: Any,
                      validated: Sequence[ValidatedRecord] | Sequence[SynthRecord],
                      spider_data: SpiderData | None,
                      raw_examples: Sequence[Example] | None = None
                      ) -> tuple[list[SFTRecord], dict[str, Any]]:
    """Convert validated synthetic records (+ raw Spider examples) into SFT records.

    Parameters
    ----------
    cfg:
        Pipeline config; ``cfg.fmt`` selects the chat template, tool name, schema
        truncation and the hard-negative weight.
    validated:
        Output of :func:`fcdata.validate.validate_records`.  Raw
        :class:`~fcdata.records.SynthRecord` objects are also accepted (their
        validation status is not re-checked here) which keeps the function usable
        for a quick "what would this look like" dry run.
    spider_data:
        Supplies schemas + sqlite paths.  ``None`` is allowed (DDL and the
        hard-negative symptom are then omitted) but the pipeline always passes it.
    raw_examples:
        Plain Spider examples to include as ``kind="raw"``.  Pass
        ``spider_data.train + spider_data.dev`` to get both splits; the example's
        own ``.split`` decides which JSONL it lands in.

    Returns
    -------
    (records, stats)
        ``stats`` contains per-kind counts, token attribution, weight sums and the
        tool-call validity rate measured by re-parsing every rendered target.
    """
    template = str(getattr(cfg.fmt, "chat_template", DEFAULT_TEMPLATE))
    if template not in TEMPLATES:
        raise ValueError(f"unknown chat_template {template!r}; expected one of {TEMPLATES}")
    tools = tool_definitions(cfg)
    databases = spider_data.databases if spider_data is not None else {}

    records: list[SFTRecord] = []
    used_ids: Counter[str] = Counter()

    def _dedupe(rec: SFTRecord) -> SFTRecord:
        used_ids[rec.id] += 1
        if used_ids[rec.id] > 1:
            rec.id = f"{rec.id}_{used_ids[rec.id]}"
        return rec

    # ---- synthetic records -------------------------------------------
    for item in validated:
        rec: SynthRecord = item.record if isinstance(item, ValidatedRecord) else item
        report = item.report if isinstance(item, ValidatedRecord) else None
        db = databases.get(rec.db_id)
        kind = str(rec.kind or "raw")
        reasoning_synth = False
        if kind == "cot":
            reasoning_synth = not sanitize_reasoning(rec.reasoning or "", rec.sql or "")
        meta: dict[str, Any] = {
            "template": template,
            "source": "synth",
            "semantic_score": round(float(report.semantic_score), 4) if report else 0.0,
            "reasoning_synthesized": reasoning_synth,
        }
        if kind == "hard_negative":
            meta["error_kind"] = rec.error_kind
            meta["bad_sql"] = rec.bad_sql
        records.append(_dedupe(_make_sft_record(
            cfg, rec, db, db_path=_db_path_for(spider_data, rec.db_id),
            template=template, tools=tools, kind=kind, split="train",
            weight=kind_weight(cfg, kind),
            synth_tokens=int(rec.total_tokens), meta=meta)))

    # ---- raw Spider examples (the baseline half) ----------------------
    for ex in raw_examples or ():
        db = databases.get(str(ex.db_id))
        split = str(getattr(ex, "split", "train") or "train")
        records.append(_dedupe(_make_sft_record(
            cfg, ex, db, db_path=_db_path_for(spider_data, str(ex.db_id)),
            template=template, tools=tools, kind="raw", split=split, weight=1.0,
            synth_tokens=0,
            meta={"template": template, "source": "raw_spider"})))

    stats = _corpus_stats(cfg, records, template)
    return records, stats


def _corpus_stats(cfg: Any, records: Sequence[SFTRecord], template: str) -> dict[str, Any]:
    """Measure the corpus instead of asserting it is well formed."""
    by_kind: dict[str, dict[str, Any]] = {}
    problems: Counter[str] = Counter()
    prompt_tokens = 0
    for rec in records:
        k = by_kind.setdefault(rec.kind, {"n": 0, "weight_sum": 0.0,
                                          "synth_tokens": 0, "tool_call_ok": 0,
                                          "avg_semantic_score": 0.0, "_score_sum": 0.0,
                                          "_scored": 0})
        k["n"] += 1
        k["weight_sum"] += rec.weight
        k["synth_tokens"] += rec.synth_tokens
        issues = validate_target(rec.messages[-1], rec.target_sql, template)
        k["tool_call_ok"] += int(not issues)
        for issue in issues:
            problems[issue] += 1
        score = rec.meta.get("semantic_score")
        if score is not None and rec.kind != "raw":
            k["_score_sum"] += float(score)
            k["_scored"] += 1
        prompt_tokens += estimate_messages_tokens(rec.messages[:-1])

    n_ok = sum(v["tool_call_ok"] for v in by_kind.values())
    for v in by_kind.values():
        v["avg_semantic_score"] = (round(v["_score_sum"] / v["_scored"], 4)
                                   if v["_scored"] else 0.0)
        del v["_score_sum"], v["_scored"]

    train = [r for r in records if r.split == "train"]
    dev = [r for r in records if r.split != "train"]
    return {
        "template": template,
        "tool_name": tool_name(cfg),
        "n_records": len(records),
        "n_train": len(train),
        "n_dev": len(dev),
        "by_kind": by_kind,
        "kinds": sorted(by_kind),
        "synth_tokens_total": sum(r.synth_tokens for r in records),
        "weight_sum_total": round(sum(r.weight for r in records), 4),
        "tool_call_valid": n_ok,
        "tool_call_valid_rate": round(n_ok / len(records), 4) if records else 0.0,
        "target_problems": dict(problems),
        "prompt_tokens_total_est": prompt_tokens,
        "prompt_tokens_avg_est": round(prompt_tokens / len(records), 1) if records else 0.0,
        "avg_weight": round(sum(r.weight for r in records) / len(records), 4) if records else 0.0,
        "avg_difficulty": round(sum(r.difficulty for r in records) / len(records), 4) if records else 0.0,
    }


def split_train_dev(records: Sequence[SFTRecord]) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Split by ``SFTRecord.split`` (the raw examples carry their Spider split)."""
    return ([r for r in records if r.split == "train"],
            [r for r in records if r.split != "train"])


# --------------------------------------------------------------------------- #
# artifact writer
# --------------------------------------------------------------------------- #
def build_and_write(cfg: Any,
                    validated: Sequence[ValidatedRecord] | Sequence[SynthRecord],
                    spider_data: SpiderData | None,
                    raw_examples: Sequence[Example] | None = None,
                    out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build the corpus and write the stage artifacts.

    Writes ``train.jsonl``, ``dev.jsonl``, ``tool_schema.json`` and
    ``format_stats.json`` into ``out_dir`` (default ``cfg.paths.sft_dir``).
    Returns the stats dict with the write summary merged in.
    """
    out = Path(out_dir) if out_dir is not None else Path(cfg.paths.sft_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, stats = build_sft_records(cfg, validated, spider_data, raw_examples)
    train, dev = split_train_dev(records)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_json(out / "tool_schema.json", tool_schema(cfg))
    stats["written"] = {"train": len(train), "dev": len(dev), "out_dir": str(out)}
    write_json(out / "format_stats.json", stats)
    return stats
