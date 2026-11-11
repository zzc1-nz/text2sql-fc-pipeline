"""Independently recompute partial metrics from stored predictions.jsonl.

There are two competing `ablation.json` files on disk (one written by the CLI run,
one by eval-engineer's run) reporting different numbers. This script ignores both
and recomputes the metrics from the *predictions* each arm actually produced,
which is the only thing that cannot be a stale summary.

    $env:PYTHONPATH='src'; python scripts/verify_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.dbexec import evaluate_pair
from fcdata.records import read_jsonl
from fcdata.spider import load_spider
from fcdata.sqlutils import analyze_sql


def main() -> int:
    data = load_spider("data/fixture")
    # gold sql for each dev example, keyed by example id
    gold = {e.id: e for e in data.dev}

    arms = ["raw_only", "plus_cot", "plus_hard_negatives"]
    print("=" * 100)
    print("INDEPENDENT RECOMPUTATION from each arm's predictions.jsonl")
    print("=" * 100)
    hdr = (f"{'arm':22s} {'n':>4s} {'exec':>6s} {'execble':>8s} "
           f"{'tbl_match':>10s} {'skeleton':>9s}")
    print(hdr)
    print("-" * len(hdr))

    for arm in arms:
        preds = list(read_jsonl(Path("artifacts/06_eval/ablation") / arm / "eval" / "predictions.jsonl"))
        if not preds:
            print(f"{arm:22s} NO predictions")
            continue
        n = len(preds)
        n_exec = n_exe = n_tbl = n_skel = 0
        for p in preds:
            src = gold.get(p["example_id"])
            if src is None:
                continue
            db = data.databases[src.db_id]
            res = evaluate_pair(data.db_path(src.db_id), p["pred_sql"], p["gold_sql"])
            n_exec += int(res["exec_match"])
            n_exe += int(res["executable"])
            a_pred = analyze_sql(p["pred_sql"], db)
            a_gold = analyze_sql(p["gold_sql"], db)
            if set(a_pred.tables) and set(a_pred.tables) == set(a_gold.tables):
                n_tbl += 1
            if p["pred_sql"].strip() and _skeleton(p["pred_sql"]) == _skeleton(p["gold_sql"]):
                n_skel += 1
        print(f"{arm:22s} {n:>4d} {n_exec / n:>6.3f} {n_exe / n:>8.3f} "
              f"{n_tbl / n:>10.3f} {n_skel / n:>9.3f}")

    print()
    print("note: tbl_match here = non-empty predicted table set equal to gold's table set")
    return 0


def _skeleton(sql: str) -> str:
    from fcdata.sqlutils import sql_skeleton
    return sql_skeleton(sql)


if __name__ == "__main__":
    raise SystemExit(main())
