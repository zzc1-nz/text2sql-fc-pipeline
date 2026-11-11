"""Pick the semantic-judge threshold from data instead of inheriting a default.

The semantic gate is only meaningful if its threshold is calibrated. This script
sweeps candidate thresholds and measures, for each one:

* how many records survive,
* how many of the survivors are *provably correct* (their SQL executes to the same
  result set as the source Spider gold query -- an independent ground truth that
  does not depend on the judge at all),
* how many correct records were wrongly dropped (false negatives).

The distinction matters because the offline teacher is a heuristic: a *real* LLM
judge should never be shown the gold SQL (that would make the check circular),
but that same blindness makes a heuristic judge under-score correct queries whose
columns the question never needed to name.

    $env:PYTHONPATH='src'; python scripts/tune_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.config import load_config
from fcdata.dbexec import evaluate_pair
from fcdata.records import read_synth_records
from fcdata.spider import load_spider
from fcdata.tokens import TokenLedger
from fcdata.validate import validate_records

THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config("configs/demo.yaml")
    data = load_spider(cfg.paths.fixture_dir)
    gold = {e.id: e for e in data.train + data.dev}

    synth_p = Path(cfg.paths.synth_dir) / "synth.jsonl"
    if not synth_p.exists():
        raise SystemExit(f"missing {synth_p}; run `python -m fcdata.cli synth` first")
    records = read_synth_records(synth_p)

    # score every record, including the ones that would normally be dropped
    cfg.validate.drop_failed = False
    validated = validate_records(cfg, records, data, TokenLedger(None))

    scored = [v for v in validated if v.report.semantic_checked]

    def is_correct(v) -> bool:
        """Ground truth: does this record's SQL reproduce the gold result?"""
        src = gold.get(v.record.source_id)
        if src is None:
            return False
        try:
            return bool(evaluate_pair(data.db_path(src.db_id), v.record.sql, src.sql)["exec_match"])
        except FileNotFoundError:
            return False

    correctness = {v.id: is_correct(v) for v in scored}
    n_total = len(scored)
    n_correct_total = sum(correctness.values())

    print(f"records scored : {n_total}")
    print(f"provably correct: {n_correct_total} ({n_correct_total / max(1, n_total):.1%})")
    print()
    header = (f"{'thr':>5} {'kept':>6} {'keep%':>7} {'kept_correct':>13} "
              f"{'precision':>10} {'false_neg':>10} {'FNR':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for th in THRESHOLDS:
        kept = [v for v in scored if v.report.semantic_score >= th]
        kept_correct = sum(1 for v in kept if correctness[v.id])
        false_neg = n_correct_total - kept_correct
        precision = kept_correct / max(1, len(kept))
        fnr = false_neg / max(1, n_correct_total)
        print(f"{th:>5.2f} {len(kept):>6} {len(kept) / max(1, n_total):>6.1%} "
              f"{kept_correct:>13} {precision:>9.1%} {false_neg:>10} {fnr:>6.1%}")
        rows.append({
            "threshold": th, "kept": len(kept),
            "keep_rate": round(len(kept) / max(1, n_total), 4),
            "kept_correct": kept_correct,
            "precision": round(precision, 4),
            "false_negatives": false_neg,
            "false_negative_rate": round(fnr, 4),
        })

    # Choose the threshold that maximises kept_correct, breaking ties toward the
    # highest precision (i.e. the strictest threshold that still keeps the data).
    best = max(rows, key=lambda r: (r["kept_correct"], r["precision"]))
    print(f"\nrecommended threshold: {best['threshold']:.2f} "
          f"(keeps {best['kept_correct']}/{n_correct_total} correct, "
          f"precision {best['precision']:.1%}, FNR {best['false_negative_rate']:.1%})")

    out = Path(cfg.paths.reports_dir) / "threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_scored": n_total, "n_correct_total": n_correct_total,
        "sweep": rows, "recommended": best,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
