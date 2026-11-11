"""Re-serialize ``ablation.json`` / ``scale_curve.json`` from existing artifacts.

WHY this exists
---------------
The ablation and scale runners write their JSON as the last step of a *training*
run, which costs ~15 minutes per arm. When only the *serialization* changes (for
example a new field such as the nested ``partial`` block the report and
``scripts/show_results.py`` read), re-running the whole training sweep just to
re-emit JSON would be wasteful and would produce different numbers if anything
about the run is non-deterministic.

This script rebuilds the JSON payloads from what is already on disk:

* per-arm metrics: ``artifacts/06_eval/<arm>/eval/metrics.json`` (the real,
  measured evaluation output -- including the ``partial`` block);
* training timings: ``artifacts/06_eval/<arm>/train/metrics.json``;
* the SFT corpus: ``artifacts/04_sft/train.jsonl`` (for record counts and the
  synthesis tokens attributable to each arm);
* the synthesis ledger (for the real USD cost per kind).

It performs **no training and no evaluation**. Every accuracy number it writes is
read back from a metrics file that a real run produced.

Usage::

    $env:PYTHONPATH='src'
    python scripts/reserialize_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fcdata.config import load_config
from fcdata.eval.ablation import (
    ABLATION_CONFIGS,
    AblationRow,
    build_ablation_mix,
)
from fcdata.eval.pipeline import (
    SFTBundle,
    cost_for_kinds,
    load_sft_bundle,
    synth_cost_by_kind,
    synth_tokens_of,
    write_rows_json_csv,
)
from fcdata.eval.ablation import ABLATION_COLUMNS
from fcdata.eval.scale import SCALE_COLUMNS, ScalePoint, sample_synthetic_fraction
from fcdata.records import read_json
from fcdata.spider import load_spider


def _eval_metrics(path: Path) -> dict[str, Any] | None:
    """Read a run's ``metrics.json`` (returns ``None`` when the run is absent)."""
    data = read_json(path, None)
    return data if isinstance(data, dict) else None


def _train_seconds(path: Path) -> float:
    """Wall-clock training time recorded by the trainer, if available."""
    data = _eval_metrics(path)
    if not data:
        return 0.0
    return float(data.get("wall_time_s", 0.0) or 0.0)


def rebuild_ablation(cfg: Any, spider: Any, bundle: SFTBundle,
                     eval_dir: Path) -> dict[str, Any] | None:
    """Rebuild ``ablation.json`` from per-arm artifacts (no training)."""
    rows: list[AblationRow] = []
    cost_by_kind = synth_cost_by_kind(cfg)
    for name in cfg.ablation.configs:
        arm = eval_dir / name
        metrics = _eval_metrics(arm / "eval" / "metrics.json")
        if metrics is None:
            print(f"[reserialize] skipping {name}: no eval metrics at {arm}")
            continue
        spec = ABLATION_CONFIGS[name]
        records = build_ablation_mix(bundle, name, seed=int(cfg.seed))
        cost, from_ledger = cost_for_kinds(cfg, spec["kinds"], records, cost_by_kind)
        partial = metrics.get("partial") or {}
        ci = metrics.get("exec_acc_ci", [0.0, 0.0])
        row = AblationRow(
            config=name,
            description=str(spec["description"]),
            kinds=tuple(spec["kinds"]),
            n_sft_records=len(records),
            n_unique_records=len({r.id for r in records}),
            synth_tokens=synth_tokens_of(records),
            synth_cost_usd=cost,
            cost_from_ledger=from_ledger,
            exec_acc=float(metrics.get("exec_acc", 0.0)),
            exec_acc_ci=(float(ci[0]), float(ci[1])),
            exact_match=float(metrics.get("exact_match", 0.0)),
            executable_rate=float(metrics.get("executable_rate", 0.0)),
            n_eval=int(metrics.get("n", 0)),
            train_seconds=_train_seconds(arm / "train" / "metrics.json"),
            train_dir=str(arm / "train"),
            eval_dir=str(arm / "eval"),
            token_acc=float(partial.get("token_acc", 0.0)),
            skeleton_match=float(partial.get("skeleton_match", 0.0)),
            table_set_match=float(partial.get("table_set_match", 0.0)),
            column_f1=float(partial.get("column_f1", 0.0)),
            clause_match_mean=float(partial.get("clause_match_mean", 0.0)),
            partial_failures={
                k: int(v.get("n", 0))
                for k, v in (partial.get("failure_taxonomy") or {}).items()
                if int(v.get("n", 0) or 0) > 0
            },
            partial_ci={
                k: [float(v[0]), float(v[1])]
                for k, v in (partial.get("ci") or {}).items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            },
        )
        rows.append(row)
    if not rows:
        return None
    baseline = next((r for r in rows if r.config == "raw_only"), rows[0])
    for row in rows:
        row.delta_exec_acc = round(row.exec_acc - baseline.exec_acc, 6)
        row.delta_table_set_match = round(
            row.table_set_match - baseline.table_set_match, 6)
    flat = [r.as_row() for r in rows]
    return {
        "rows": flat,
        "baseline": baseline.config,
        "best": max(rows, key=lambda r: (r.exec_acc, -r.synth_tokens)).as_row(),
        "n_configs": len(rows),
        "source": bundle.source,
        "quick": False,
        "eval_split": str(cfg.ev.split),
        "n_eval_examples": rows[0].n_eval,
        "synth_cost_by_kind": cost_by_kind,
        "sft_kind_counts": bundle.kind_counts(),
        "n_sft_available": len(bundle.train),
        "warnings": [],
        "degenerate": False,
        "reserialized_from": "per-arm eval/metrics.json (no training re-run)",
    }


def rebuild_scale(cfg: Any, bundle: SFTBundle, eval_dir: Path) -> dict[str, Any] | None:
    """Rebuild ``scale_curve.json`` from per-fraction artifacts (no training)."""
    cost_by_kind = synth_cost_by_kind(cfg)
    points: list[ScalePoint] = []
    for frac_dir in sorted(eval_dir.glob("frac_*")):
        metrics = _eval_metrics(frac_dir / "eval" / "metrics.json")
        if metrics is None:
            continue
        label = frac_dir.name.replace("frac_", "").replace("p", ".")
        try:
            frac = float(label)
        except ValueError:
            continue
        records = sample_synthetic_fraction(bundle.train, frac, seed=int(cfg.seed))
        synth_records = [r for r in records if r.kind != "raw"]
        kinds = sorted({r.kind for r in synth_records})
        cost, from_ledger = cost_for_kinds(cfg, kinds, synth_records, cost_by_kind)
        partial = metrics.get("partial") or {}
        ci = metrics.get("exec_acc_ci", [0.0, 0.0])
        points.append(ScalePoint(
            fraction=frac,
            n_sft_records=len(records),
            n_synth_records=len(synth_records),
            synth_tokens=synth_tokens_of(synth_records),
            synth_cost_usd=cost,
            cost_from_ledger=from_ledger,
            exec_acc=float(metrics.get("exec_acc", 0.0)),
            exec_acc_ci=(float(ci[0]), float(ci[1])),
            exact_match=float(metrics.get("exact_match", 0.0)),
            executable_rate=float(metrics.get("executable_rate", 0.0)),
            n_eval=int(metrics.get("n", 0)),
            train_seconds=_train_seconds(frac_dir / "train" / "metrics.json"),
            train_dir=str(frac_dir / "train"),
            eval_dir=str(frac_dir / "eval"),
            token_acc=float(partial.get("token_acc", 0.0)),
            skeleton_match=float(partial.get("skeleton_match", 0.0)),
            table_set_match=float(partial.get("table_set_match", 0.0)),
            column_f1=float(partial.get("column_f1", 0.0)),
            clause_match_mean=float(partial.get("clause_match_mean", 0.0)),
        ))
    if not points:
        return None
    points.sort(key=lambda p: p.fraction)
    baseline = next((p for p in points if p.fraction <= 0.0), None)
    for p in points:
        if baseline is not None:
            p.delta_exec_acc = round(p.exec_acc - baseline.exec_acc, 6)
    from fcdata.eval.scale import _saturation  # noqa: PLC0415 - reuse exact logic

    flat = [p.as_row() for p in points]
    return {
        "points": flat,
        "fractions": [round(p.fraction, 4) for p in points],
        "source": bundle.source,
        "quick": False,
        "eval_split": str(cfg.ev.split),
        "n_eval_examples": points[0].n_eval,
        "n_sft_available": len(bundle.train),
        "n_synth_available": sum(1 for r in bundle.train if r.kind != "raw"),
        "synth_cost_by_kind": cost_by_kind,
        "saturation": _saturation(points),
        "baseline": baseline.as_row() if baseline is not None else None,
        "reserialized_from": "per-fraction eval/metrics.json (no training re-run)",
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_config("configs/demo.yaml", root=".")
    spider = load_spider("data/fixture")
    bundle = load_sft_bundle(cfg, spider)
    eval_dir = Path(cfg.paths.eval_dir)
    print(f"[reserialize] SFT corpus: {len(bundle.train)} records from {bundle.source}")

    ablation = rebuild_ablation(cfg, spider, bundle, eval_dir)
    if ablation:
        flat = ablation["rows"]
        write_rows_json_csv(flat, eval_dir / "ablation.json",
                            eval_dir / "ablation.csv", ABLATION_COLUMNS, ablation)
        print(f"[reserialize] wrote ablation.json ({len(flat)} arms, "
              f"partial nested={'partial' in flat[0]})")

    scale = rebuild_scale(cfg, bundle, eval_dir)
    if scale:
        flat = scale["points"]
        write_rows_json_csv(flat, eval_dir / "scale_curve.json",
                            eval_dir / "scale_curve.csv", SCALE_COLUMNS, scale)
        print(f"[reserialize] wrote scale_curve.json ({len(flat)} points, "
              f"partial nested={'partial' in flat[0]})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
