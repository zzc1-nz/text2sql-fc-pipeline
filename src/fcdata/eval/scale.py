"""Data-scale curve: capability as a function of how much synthetic data we bought.

WHY this experiment exists
--------------------------
The ablation answers "does synthesis help at all". This answers the follow-up a
reviewer always asks next: **"how much synthetic data do you need before it
pays off, and where does it saturate?"** That question is what turns a demo into
an engineering decision, because it lets you say "spend X tokens, expect Y
points of execution accuracy, and stop at Z".

Method: hold the model architecture, seed and step budget fixed, sweep the
*fraction of the available synthetic corpus* included in SFT
(``0.0, 0.25, 0.5, 1.0`` by default), and record for each point

* ``n_sft_records``  -- dataset size actually trained on,
* ``synth_tokens``   -- synthesis tokens invested to produce that subset,
* ``exec_acc`` / ``exact_match`` -- measured by real sqlite execution.

``fraction=0.0`` is the raw-only baseline, so the curve starts at the
"no synthesis investment" cost, which is exactly what the dual-axis figure needs.

Sampling is deterministic (``random.Random(seed)`` over the synthetic records,
stratified by kind) so re-running reproduces the same curve.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..records import Example, SFTRecord
from .pipeline import (
    SFTBundle,
    cost_for_kinds,
    load_sft_bundle,
    quick_cfg,
    select_kinds,
    synth_cost_by_kind,
    synth_tokens_of,
    train_and_evaluate,
    write_rows_json_csv,
)

#: Default sweep. 0.0 is the no-synthesis baseline point.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: CSV column order (also the markdown table order in the report).
SCALE_COLUMNS: tuple[str, ...] = (
    "fraction", "n_sft_records", "n_synth_records", "synth_tokens", "synth_cost_usd",
    "cost_from_ledger", "exec_acc", "exec_acc_ci_lo", "exec_acc_ci_hi", "exact_match",
    "executable_rate", "token_acc", "skeleton_match", "table_set_match",
    "column_f1", "clause_match_mean", "n_eval", "delta_exec_acc", "train_seconds",
)


@dataclass
class ScalePoint:
    """One measured point on the curve."""

    fraction: float
    n_sft_records: int = 0
    n_synth_records: int = 0
    synth_tokens: int = 0
    synth_cost_usd: float = 0.0
    cost_from_ledger: bool = False
    exec_acc: float = 0.0
    exec_acc_ci: tuple[float, float] = (0.0, 0.0)
    exact_match: float = 0.0
    executable_rate: float = 0.0
    n_eval: int = 0
    delta_exec_acc: float | None = None
    train_seconds: float = 0.0
    train_dir: str = ""
    eval_dir: str = ""
    # ---- partial credit: the signal that survives 0% strict accuracy -------
    token_acc: float = 0.0
    skeleton_match: float = 0.0
    table_set_match: float = 0.0
    column_f1: float = 0.0
    clause_match_mean: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat row plus a nested ``partial`` block (see AblationRow.as_row)."""
        lo, hi = self.exec_acc_ci
        return {
            "fraction": round(float(self.fraction), 4),
            "n_sft_records": self.n_sft_records,
            "n_synth_records": self.n_synth_records,
            "synth_tokens": self.synth_tokens,
            "synth_cost_usd": round(self.synth_cost_usd, 6),
            "cost_from_ledger": self.cost_from_ledger,
            "exec_acc": self.exec_acc,
            "exec_acc_ci_lo": lo,
            "exec_acc_ci_hi": hi,
            "exact_match": self.exact_match,
            "executable_rate": self.executable_rate,
            "token_acc": self.token_acc,
            "skeleton_match": self.skeleton_match,
            "table_set_match": self.table_set_match,
            "column_f1": self.column_f1,
            "clause_match_mean": self.clause_match_mean,
            "partial": {
                "token_acc": self.token_acc,
                "skeleton_match": self.skeleton_match,
                "table_set_match": self.table_set_match,
                "column_f1": self.column_f1,
                "clause_match_mean": self.clause_match_mean,
            },
            "n_eval": self.n_eval,
            "delta_exec_acc": self.delta_exec_acc,
            "train_seconds": self.train_seconds,
            "train_dir": self.train_dir,
            "eval_dir": self.eval_dir,
        }


def sample_synthetic_fraction(records: Sequence[SFTRecord],
                              fraction: float,
                              *,
                              seed: int = 17) -> list[SFTRecord]:
    """Take ``fraction`` of the *synthetic* records, stratified by kind.

    Raw records are never subsampled: the experiment varies the synthesis
    investment, not the base dataset. Stratifying by kind keeps the mix of
    CoT/paraphrase/hard-negative constant across points so the curve measures
    *quantity*, not *composition*.

    Deterministic for a fixed seed. ``fraction <= 0`` returns raw only;
    ``fraction >= 1`` returns everything.
    """
    raw = [r for r in records if r.kind == "raw"]
    synth = [r for r in records if r.kind != "raw"]
    if fraction <= 0.0 or not synth:
        return list(raw)
    if fraction >= 1.0:
        return list(raw) + list(synth)

    rng = random.Random(seed)
    by_kind: dict[str, list[SFTRecord]] = {}
    for rec in synth:
        by_kind.setdefault(rec.kind, []).append(rec)
    picked: list[SFTRecord] = []
    for kind in sorted(by_kind):
        bucket = sorted(by_kind[kind], key=lambda r: r.id)
        rng.shuffle(bucket)
        k = max(1, int(round(len(bucket) * float(fraction))))
        picked.extend(bucket[:k])
    return list(raw) + picked


def run_scale_curve(cfg: Any,
                    spider_data: Any,
                    fractions: Sequence[float] | None,
                    out_dir: str | Path,
                    *,
                    quick: bool = False,
                    bundle: SFTBundle | None = None,
                    train_fn: Callable[..., dict[str, Any]] | None = None,
                    load_fn: Callable[..., tuple[Any, Any]] | None = None,
                    generate_fn: Callable[..., str] | None = None,
                    prompt_fn: Callable[..., str] | None = None,
                    progress: bool = True) -> dict[str, Any]:
    """Sweep the synthetic-data fraction; train + evaluate at each point.

    Parameters
    ----------
    fractions:
        Fractions of the synthetic corpus to include (``0.0`` = raw only).
        ``None`` uses :data:`DEFAULT_FRACTIONS`.
    quick:
        Smoke mode: 1 epoch and a handful of eval examples.
    train_fn / load_fn / generate_fn:
        Injection points for tests.

    Returns
    -------
    dict
        ``{"points": [...], "fractions": [...], "saturation": {...},
        "source": ..., "quick": bool, ...}``. ``saturation`` reports where the
        curve flattens: the first point whose exec_acc is within one point of
        the final accuracy (with its token cost), so the report can state the
        cheapest configuration that reaches the best observed accuracy.
    """
    run_cfg = quick_cfg(cfg) if quick else cfg
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)

    sft = bundle or load_sft_bundle(run_cfg, spider_data)
    if not sft.train:
        raise RuntimeError(
            "no SFT records available for the scale curve: run the format stage "
            f"first (looked in {cfg.paths.sft_dir})")
    eval_examples = list(sft.eval_examples)
    if not eval_examples:
        raise RuntimeError("no dev examples available to evaluate the scale curve")

    fracs = [float(f) for f in (fractions if fractions is not None else DEFAULT_FRACTIONS)]
    fracs = _normalize_fractions(fracs)
    cost_by_kind = synth_cost_by_kind(run_cfg)

    points: list[ScalePoint] = []
    for i, frac in enumerate(fracs, start=1):
        if progress:
            print(f"[scale] ({i}/{len(fracs)}) fraction={frac:.2f}: training...", flush=True)
        records = sample_synthetic_fraction(sft.train, frac,
                                            seed=int(getattr(run_cfg, "seed", 17)))
        synth_records = [r for r in records if r.kind != "raw"]
        point = ScalePoint(
            fraction=frac,
            n_sft_records=len(records),
            n_synth_records=len(synth_records),
            synth_tokens=synth_tokens_of(synth_records),
        )
        kinds = sorted({r.kind for r in synth_records})
        cost, from_ledger = cost_for_kinds(run_cfg, kinds, synth_records, cost_by_kind)
        point.synth_cost_usd = cost
        point.cost_from_ledger = from_ledger

        label = f"frac_{frac:.2f}".replace(".", "p")
        result = train_and_evaluate(
            run_cfg, spider_data, records, sft.dev, eval_examples,
            work / label, train_fn=train_fn, load_fn=load_fn,
            generate_fn=generate_fn, prompt_fn=prompt_fn, progress=progress)
        ev = result["eval"]
        point.exec_acc = float(ev.get("exec_acc", 0.0))
        ci = ev.get("exec_acc_ci", [0.0, 0.0])
        point.exec_acc_ci = (float(ci[0]), float(ci[1]))
        point.exact_match = float(ev.get("exact_match", 0.0))
        point.executable_rate = float(ev.get("executable_rate", 0.0))
        point.n_eval = int(ev.get("n", 0))
        point.train_seconds = float(result.get("train_seconds", 0.0))
        point.train_dir = result["train_dir"]
        point.eval_dir = result["eval_dir"]
        partial = ev.get("partial") or {}
        point.token_acc = float(partial.get("token_acc", 0.0))
        point.skeleton_match = float(partial.get("skeleton_match", 0.0))
        point.table_set_match = float(partial.get("table_set_match", 0.0))
        point.column_f1 = float(partial.get("column_f1", 0.0))
        point.clause_match_mean = float(partial.get("clause_match_mean", 0.0))
        points.append(point)
        if progress:
            print(f"[scale] fraction={frac:.2f} n_sft={point.n_sft_records} "
                  f"exec_acc={point.exec_acc * 100:.1f}% "
                  f"token_acc={point.token_acc * 100:.1f}% "
                  f"tables={point.table_set_match * 100:.1f}% "
                  f"synth_tokens={point.synth_tokens:,}", flush=True)

    baseline = next((p for p in points if p.fraction <= 0.0), None)
    for p in points:
        if baseline is not None:
            p.delta_exec_acc = round(p.exec_acc - baseline.exec_acc, 6)

    flat = [p.as_row() for p in points]
    payload = {
        "points": flat,
        "fractions": fracs,
        "source": sft.source,
        "quick": bool(quick),
        "eval_split": str(getattr(getattr(run_cfg, "ev", None), "split", "dev")),
        "n_eval_examples": len(eval_examples),
        "n_sft_available": len(sft.train),
        "n_synth_available": sum(1 for r in sft.train if r.kind != "raw"),
        "synth_cost_by_kind": cost_by_kind,
        "saturation": _saturation(points),
        "baseline": baseline.as_row() if baseline is not None else None,
    }
    write_rows_json_csv(flat, work / "scale_curve.json", work / "scale_curve.csv",
                        SCALE_COLUMNS, payload)
    if progress:
        print(f"[scale] wrote {work / 'scale_curve.json'} and scale_curve.csv", flush=True)
    return payload


def _normalize_fractions(fracs: Sequence[float]) -> list[float]:
    """Clamp to [0, 1], deduplicate, and sort so the curve is monotone in x."""
    seen: set[float] = set()
    out: list[float] = []
    for f in fracs:
        v = round(min(1.0, max(0.0, float(f))), 4)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def _saturation(points: Sequence[ScalePoint], tolerance: float = 0.01) -> dict[str, Any]:
    """Cheapest point that reaches (almost) the best observed accuracy.

    "Almost" is ``tolerance`` (one percentage point) so a curve that wobbles by
    noise does not report an absurd saturation point. Returns the fraction, the
    token cost and the accuracy gap versus the best point, which is the number
    the README quotes as "we could stop at 50% of the corpus".

    Two cases are reported honestly rather than dressed up:

    * **no measurable capability yet** (best accuracy is 0): there is no
      saturation point to find, because nothing has saturated. Reporting
      "0% of the corpus is enough" would be actively misleading.
    * **flat curve with non-zero accuracy**: every point ties, so the cheapest
      one genuinely is the right operating point.
    """
    if not points:
        return {}
    best = max(points, key=lambda p: p.exec_acc)
    if best.exec_acc <= 0.0:
        return {
            "available": False,
            "reason": ("no measured capability on any point (all exec_acc are 0), "
                       "so there is no saturation point to report; train longer "
                       "or on more data before interpreting this curve"),
            "best_fraction": round(best.fraction, 4),
            "best_exec_acc": best.exec_acc,
        }
    candidates = [p for p in points if p.exec_acc >= best.exec_acc - tolerance]
    if not candidates:
        return {}
    cheapest = min(candidates, key=lambda p: (p.synth_tokens, p.fraction))
    return {
        "available": True,
        "fraction": round(cheapest.fraction, 4),
        "n_sft_records": cheapest.n_sft_records,
        "synth_tokens": cheapest.synth_tokens,
        "synth_cost_usd": cheapest.synth_cost_usd,
        "exec_acc": cheapest.exec_acc,
        "best_fraction": round(best.fraction, 4),
        "best_exec_acc": best.exec_acc,
        "accuracy_gap": round(best.exec_acc - cheapest.exec_acc, 6),
        "tokens_saved": int(best.synth_tokens - cheapest.synth_tokens),
        "tolerance": tolerance,
    }


def load_scale_curve(path: str | Path) -> dict[str, Any]:
    """Read a previously written ``scale_curve.json`` (used by the report stage)."""
    from ..records import read_json  # noqa: PLC0415 - local import

    data = read_json(path, None)
    if data is None:
        raise FileNotFoundError(f"no scale curve results at {path}")
    return data
