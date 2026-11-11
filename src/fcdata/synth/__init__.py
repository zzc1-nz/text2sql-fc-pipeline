"""Synthesis stage: turn raw Spider examples into an augmented training corpus.

Public API (frozen by CONTRACTS.md)::

    build_synth_records(cfg, spider_data, ledger) -> list[SynthRecord]

Why one orchestrator
--------------------
The three augmentation kinds share a source plan and a token budget. Running them
independently would (a) recompute the plan three times, (b) let each kind
independently blow the budget, and (c) make the stage summary lie about where the
tokens went. So the orchestrator:

1. builds the :class:`~fcdata.synth.common.SourcePlan` list once,
2. runs the kinds **in a fixed, deliberate order** (``cot`` -> ``paraphrase`` ->
   ``hard_negative``) so a budget cutoff degrades the corpus in a known way: the
   highest-value augmentation survives, the cheapest-to-drop one goes last,
3. shares a single :class:`~fcdata.providers.AsyncBatchRunner` per kind so budget
   accounting is continuous across the whole stage,
4. returns records sorted by ``(kind, source_id, index)`` so the JSONL is stable
   across runs and diffs stay readable.

The stage never raises on a per-record failure: a failed call is counted in the
report and simply produces no record. Only a genuine programming error (bad
config, missing dataset) propagates.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..config import Config
from ..records import SynthRecord
from ..tokens import TokenLedger
from .common import (SYNTH_STAGE, KindReport, SourcePlan, build_sources,
                     summarize)
from .cot import synth_cot
from .hard_negative import synth_hard_negative
from .paraphrase import synth_paraphrase

#: execution order; earlier kinds survive a budget cutoff, later ones are dropped
KIND_ORDER: tuple[str, ...] = ("cot", "paraphrase", "hard_negative")

__all__ = ["build_synth_records", "synthesize", "KIND_ORDER", "SYNTH_STAGE",
           "summarize_synthesis"]


def synthesize(cfg: Config, spider_data: Any, ledger: TokenLedger,
               *, plans: Sequence[SourcePlan] | None = None,
               kinds: Sequence[str] | None = None,
               verbose: bool = False) -> tuple[list[SynthRecord], dict[str, Any]]:
    """Run the synthesis stage and return ``(records, summary)``.

    ``kinds`` restricts the run to a subset -- used by the ablation runner
    (``raw_only`` / ``plus_cot`` / ``plus_hard_negatives``) without duplicating
    any generator logic.
    """
    requested = [k for k in KIND_ORDER if kinds is None or k in kinds]
    if plans is None:
        plans = build_sources(cfg, spider_data)
    if verbose:
        print(f"[synth] {len(plans)} source examples; kinds={requested}")

    records: list[SynthRecord] = []
    reports: list[KindReport] = []
    generators = {"cot": synth_cot, "paraphrase": synth_paraphrase,
                  "hard_negative": synth_hard_negative}
    for kind in requested:
        gen = generators[kind]
        recs, report = gen(cfg, spider_data, ledger, plans=plans)
        records.extend(recs)
        reports.append(report)
        if verbose:
            print(f"[synth] {kind}: {report.ok}/{report.planned} records, "
                  f"{report.total_tokens} tokens, "
                  f"failed={report.failed} dropped={report.dropped_invalid}")

    records.sort(key=lambda r: (r.kind, str(r.source_id), r.id))
    return records, summarize(records, reports, ledger)


def build_synth_records(cfg: Config, spider_data: Any,
                        ledger: TokenLedger) -> list[SynthRecord]:
    """Contract entry point: produce the augmented corpus for ``spider_data``."""
    records, _summary = synthesize(cfg, spider_data, ledger)
    return records


def summarize_synthesis(cfg: Config, spider_data: Any, ledger: TokenLedger) -> dict[str, Any]:
    """Re-derive the stage summary without re-running generation.

    Useful for the report stage and for the CLI, which needs the token numbers even
    when it loaded the corpus from disk.
    """
    records, summary = synthesize(cfg, spider_data, ledger, kinds=())
    return summary
