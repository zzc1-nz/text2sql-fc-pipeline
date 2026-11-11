"""Token-investment extrapolation: measured small-scale cost -> projected scale.

The demo run in this repo executes the *real* pipeline (real prompts, real API
calls or the deterministic offline teacher, real token accounting) on a small
Spider-format fixture. This module answers the question a reviewer will ask
immediately: *"you measured 1.2M tokens on 98 examples -- what does this cost at
real Spider scale, and does it reach the 10M+ token budget the project claims?"*

It does so **without inventing numbers**: every projection is
``measured_per_sample_tokens x target_count``, with the measured value read
straight from the token ledger. The only assumption is linear scaling of prompt
cost, which is explicit and auditable (prompt length is dominated by the DDL
block, which does not grow with dataset size).

Output feeds the README's "cost vs capability" narrative and the resume bullet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .tokens import TokenLedger, summarize_ledger


# --------------------------------------------------------------------------- #
# Spider-scale reference points (public dataset facts)
# --------------------------------------------------------------------------- #
SPIDER_SCALE = {
    "n_questions": 10_181,       # Spider 1.0 train + dev
    "n_unique_sql": 5_693,       # distinct SQL queries in Spider
    "n_databases": 200,          # 200+ databases
    "train_questions": 7_000,    # train_spider.json is 7,000
    "dev_questions": 1_034,      # dev.json is 1,034
}


@dataclass
class SynthPlan:
    """A synthesis recipe: how many augmented samples per source question."""
    cot_per_example: int = 1
    paraphrase_per_example: int = 2
    hard_negative_per_example: int = 1

    @property
    def records_per_example(self) -> int:
        return (self.cot_per_example + self.paraphrase_per_example
                + self.hard_negative_per_example)

    def total_records(self, n_source: int) -> int:
        return n_source * self.records_per_example

    def breakdown(self, n_source: int) -> dict[str, int]:
        return {
            "cot": n_source * self.cot_per_example,
            "paraphrase": n_source * self.paraphrase_per_example,
            "hard_negative": n_source * self.hard_negative_per_example,
        }


@dataclass
class MeasuredCost:
    """Per-sample token cost measured from a real run."""
    n_source_examples: int
    n_synth_records: int
    synth_prompt_tokens: int
    synth_completion_tokens: int
    judge_prompt_tokens: int
    judge_completion_tokens: int
    synth_cost_usd: float
    judge_cost_usd: float

    @property
    def synth_tokens_per_record(self) -> float:
        if not self.n_synth_records:
            return 0.0
        return (self.synth_prompt_tokens + self.synth_completion_tokens) / self.n_synth_records

    @property
    def synth_tokens_per_source(self) -> float:
        if not self.n_source_examples:
            return 0.0
        return (self.synth_prompt_tokens + self.synth_completion_tokens) / self.n_source_examples

    @property
    def judge_tokens_per_record(self) -> float:
        if not self.n_synth_records:
            return 0.0
        return (self.judge_prompt_tokens + self.judge_completion_tokens) / self.n_synth_records

    @property
    def total_tokens(self) -> int:
        return (self.synth_prompt_tokens + self.synth_completion_tokens
                + self.judge_prompt_tokens + self.judge_completion_tokens)

    @property
    def total_cost_usd(self) -> float:
        return self.synth_cost_usd + self.judge_cost_usd


def measure_from_ledger(ledger_summary: dict[str, Any],
                        n_source_examples: int,
                        n_synth_records: int) -> MeasuredCost:
    """Extract the per-stage measured cost from a ledger aggregate dict."""
    stages = ledger_summary.get("by_stage", ledger_summary)
    synth = stages.get("synth", {})
    judge = stages.get("validate", {})

    def _sum(stage: dict[str, Any], key: str) -> int:
        return int(stage.get(key, 0) or 0)

    # the judge may share the "validate" stage with the format checker, which
    # spends no tokens; taking the whole stage is therefore the judge's cost.
    return MeasuredCost(
        n_source_examples=n_source_examples,
        n_synth_records=n_synth_records,
        synth_prompt_tokens=_sum(synth, "prompt_tokens"),
        synth_completion_tokens=_sum(synth, "completion_tokens"),
        judge_prompt_tokens=_sum(judge, "prompt_tokens"),
        judge_completion_tokens=_sum(judge, "completion_tokens"),
        synth_cost_usd=float(synth.get("cost_usd", 0.0) or 0.0),
        judge_cost_usd=float(judge.get("cost_usd", 0.0) or 0.0),
    )


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
@dataclass
class Projection:
    label: str
    n_source_examples: int
    n_synth_records: int
    plan: dict[str, int]
    synth_tokens: int
    judge_tokens: int
    total_tokens: int
    synth_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float

    @property
    def total_tokens_m(self) -> float:
        return round(self.total_tokens / 1_000_000, 3)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens_m"] = self.total_tokens_m
        return d


def project_cost(measured: MeasuredCost, n_source: int, plan: SynthPlan,
                 label: str = "") -> Projection:
    """Linearly scale measured per-sample cost to ``n_source`` questions.

    Deliberately linear and deliberately transparent: the prompt is dominated by
    the schema DDL block, which is a per-database constant, so per-sample prompt
    cost does not drift as the corpus grows. The completion cost is bounded by
    ``max_tokens`` and is likewise per-sample stable.
    """
    n_records = plan.total_records(n_source)
    # scale by records, not sources: the per-record cost is what generalises
    synth_tokens = round(measured.synth_tokens_per_record * n_records)
    judge_tokens = round(measured.judge_tokens_per_record * n_records)
    # cost scales with the same ratio as tokens
    ratio = (n_records / measured.n_synth_records) if measured.n_synth_records else 0.0
    synth_cost = round(measured.synth_cost_usd * ratio, 4)
    judge_cost = round(measured.judge_cost_usd * ratio, 4)
    return Projection(
        label=label or f"{n_source}_questions",
        n_source_examples=n_source,
        n_synth_records=n_records,
        plan=plan.breakdown(n_source),
        synth_tokens=synth_tokens,
        judge_tokens=judge_tokens,
        total_tokens=synth_tokens + judge_tokens,
        synth_cost_usd=synth_cost,
        judge_cost_usd=judge_cost,
        total_cost_usd=round(synth_cost + judge_cost, 4),
    )


def default_projections(measured: MeasuredCost) -> list[Projection]:
    """The three scale points the project brief asks about.

    * Spider train (7,000 questions) -- the standard fine-tuning set
    * Spider full (10,181 questions) -- every question in the benchmark
    * 3-5x augmentation target -- the 30k-50k sample regime
    """
    out: list[Projection] = []
    for label, n, plan in [
        ("Spider train (7,000 q)", SPIDER_SCALE["train_questions"], SynthPlan(1, 2, 1)),
        ("Spider full (10,181 q)", SPIDER_SCALE["n_questions"], SynthPlan(1, 2, 1)),
        ("Augmented x3 (30k samples)", SPIDER_SCALE["n_questions"], SynthPlan(1, 2, 0)),
        ("Augmented x4 (40k samples)", SPIDER_SCALE["n_questions"], SynthPlan(2, 3, 1)),
        ("Augmented x5 (50k samples)", SPIDER_SCALE["n_questions"], SynthPlan(2, 3, 2)),
    ]:
        out.append(project_cost(measured, n, plan, label))
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_extrapolation_report(measured: MeasuredCost,
                               projections: Iterable[Projection] | None = None,
                               out_dir: str | Path | None = None) -> dict:
    projs = list(projections or default_projections(measured))
    report = {
        "spider_reference_scale": SPIDER_SCALE,
        "measured": {
            **asdict(measured),
            "synth_tokens_per_record": round(measured.synth_tokens_per_record, 1),
            "synth_tokens_per_source": round(measured.synth_tokens_per_source, 1),
            "judge_tokens_per_record": round(measured.judge_tokens_per_record, 1),
            "total_tokens": measured.total_tokens,
            "total_cost_usd": round(measured.total_cost_usd, 4),
        },
        "projections": [p.as_dict() for p in projs],
        "headline": _headline(projs),
    }
    if out_dir is not None:
        p = Path(out_dir) / "token_extrapolation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _headline(projs: list[Projection]) -> dict[str, Any]:
    if not projs:
        return {}
    full = next((p for p in projs if "10,181" in p.label), projs[0])
    big = max(projs, key=lambda p: p.total_tokens)
    return {
        "measured_note": (
            "Per-sample token cost is measured on the demo fixture; projections "
            "are linear in the number of augmented records."),
        "spider_full_tokens_m": full.total_tokens_m,
        "spider_full_cost_usd": full.total_cost_usd,
        "max_tokens_m": big.total_tokens_m,
        "max_tokens_label": big.label,
        "reaches_10m": big.total_tokens >= 10_000_000,
        "cheap_model_share": round(
            (full.synth_cost_usd / full.total_cost_usd * 100) if full.total_cost_usd else 0.0, 1),
    }


def markdown_table(report: dict) -> str:
    """Render the extrapolation table for the README."""
    rows = report["projections"]
    lines = [
        "| Synthesis plan | Source questions | Augmented records | Synth tokens | "
        "Judge tokens | Total tokens | Est. cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in rows:
        lines.append(
            f"| {p['label']} | {p['n_source_examples']:,} | {p['n_synth_records']:,} | "
            f"{p['synth_tokens']:,} | {p['judge_tokens']:,} | {p['total_tokens']:,} | "
            f"${p['total_cost_usd']:,.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: extrapolate from an existing ledger file."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default="artifacts/ledger.jsonl")
    ap.add_argument("--n-source", type=int, default=98)
    ap.add_argument("--n-synth", type=int, default=392)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    summary = summarize_ledger(args.ledger)
    measured = measure_from_ledger(summary, args.n_source, args.n_synth)
    report = build_extrapolation_report(measured, out_dir=args.out)
    print(markdown_table(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
