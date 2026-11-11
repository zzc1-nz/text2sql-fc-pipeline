"""Quantify run-to-run variance so the ablation effect can be compared to noise.

The pipeline trains each arm once. To know whether an observed difference between
arms is meaningful, we need the spread of *identical* configurations. This script
collects every run of the same data mix that exists on disk and reports the spread.

This is the single most important honesty check in the project: if the noise floor
is as large as the effect, the effect is not established.

    $env:PYTHONPATH='src'; python scripts/variance_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("artifacts/06_eval")


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pick(d: dict | None, key: str) -> float | None:
    """Read a metric from either the top level or the nested ``partial`` block."""
    if not d:
        return None
    for src in (d, d.get("partial") or {}):
        v = src.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def collect() -> list[dict]:
    """Every (label, eval_metrics, train_metrics) triple on disk for raw-only data."""
    runs: list[dict] = []

    # ablation arm raw_only (98 records)
    ev = load(ROOT / "ablation" / "raw_only" / "eval" / "metrics.json")
    tr = load(ROOT / "ablation" / "raw_only" / "train" / "metrics.json")
    if ev:
        runs.append({"label": "ablation/raw_only", "data": "raw (98)", "eval": ev, "train": tr})

    # scale point 0% (same 98 records)
    ev = load(ROOT / "frac_0p00" / "eval" / "metrics.json")
    tr = load(ROOT / "frac_0p00" / "train" / "metrics.json")
    if ev:
        runs.append({"label": "scale/frac_0p00", "data": "raw (98)", "eval": ev, "train": tr})

    return runs


def main() -> int:
    runs = collect()
    if len(runs) < 2:
        print("need >=2 runs of identical configuration to measure variance")
        return 1

    print("=" * 96)
    print("RUN-TO-RUN VARIANCE: identical data mix (98 raw records), identical config, 800 steps")
    print("=" * 96)
    keys = ["exec_acc", "executable_rate", "token_acc", "table_set_match", "column_f1"]
    print(f"{'run':24s} " + " ".join(f"{k[:13]:>13s}" for k in keys) + f" {'train_loss':>11s}")
    for r in runs:
        vals = " ".join(f"{(pick(r['eval'], k) or 0):>13.3f}" for k in keys)
        loss = pick(r["train"], "final_train_loss")
        print(f"{r['label']:24s} {vals} {loss if loss is None else f'{loss:.4f}':>11s}")

    print()
    print("spread between identical runs (this is the noise floor):")
    n_eval = runs[0]["eval"].get("n") or 24
    print(f"  eval set size: {n_eval} examples -> 1 example = {100.0 / n_eval:.2f} pp")
    for k in keys:
        vals = [pick(r["eval"], k) for r in runs if pick(r["eval"], k) is not None]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            examples = spread * n_eval
            print(f"  {k:18s} min={min(vals):.3f} max={max(vals):.3f} "
                  f"spread={spread:.3f} ({spread * 100:.1f} pp = {examples:.1f} examples)")

    print()
    print("Compare against the measured ablation effect (raw_only -> plus_hard_negatives):")
    print("  table_set_match: 12.5% -> 20.8% = +8.3 pp = 2 of 24 examples")
    print("  If the identical-run spread is of similar magnitude, the effect is within noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
