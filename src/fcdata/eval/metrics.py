"""Metric aggregation for text2sql predictions, plus bootstrap uncertainty.

WHY this module exists
----------------------
A single ``exec_acc`` number is not a result, it is a claim. To make the claim
defensible the report needs three things that all live here:

* **the aggregate** (execution accuracy, exact match, executability) computed
  from ``Prediction`` records that were produced by *really executing* SQL
  against sqlite (see :mod:`fcdata.dbexec`),
* **slices** (difficulty quartiles, database, synthesis kind) so a gain cannot
  hide inside one easy bucket,
* **an uncertainty interval**, because these dev sets are small and a +2pp
  difference between two ablations is often noise. ``bootstrap_ci`` resamples
  the per-example 0/1 outcomes and returns a percentile interval; it is
  deterministic given the seed so the report is reproducible.

Nothing here invents a number: every value is a mean/count over real outcomes.

Conventions
-----------
* Accuracy-like fields are **fractions in [0, 1]**, not percentages. The report
  layer formats them as percentages with one decimal; keeping the canonical
  value a fraction avoids the classic "is 92 a percent or a fraction" bug.
* Failure buckets are always emitted, even when zero, so downstream tables have
  a stable schema.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from ..records import Prediction

#: Stable order/labels for the failure taxonomy. A prediction that failed
#: execution can only be in exactly one bucket, checked in this order.
FAILURE_TAXONOMY: tuple[str, ...] = (
    "empty_prediction",
    "not_executable",
    "wrong_row_count",
    "wrong_values",
    "gold_error",
)


# --------------------------------------------------------------------------- #
# failure taxonomy
# --------------------------------------------------------------------------- #
def classify_failure(pred: Prediction) -> str:
    """Return the single most specific failure bucket, or ``""`` on success.

    The information comes from the ``evaluate_pair`` result that
    :func:`fcdata.eval.infer.predict_examples` stored in ``pred.meta``. When a
    prediction was loaded from disk without that metadata we fall back to
    parsing the human-readable ``reason`` string, so old prediction files stay
    scoreable.
    """
    if pred.exec_match:
        return ""
    meta: dict[str, Any] = pred.meta or {}
    if not (pred.pred_sql or "").strip():
        return "empty_prediction"
    if meta.get("gold_ok") is False:
        return "gold_error"
    if not pred.executable:
        return "not_executable"
    pred_rows = meta.get("pred_rows")
    gold_rows = meta.get("gold_rows")
    if pred_rows is not None and gold_rows is not None and int(pred_rows) != int(gold_rows):
        return "wrong_row_count"
    # metadata missing -> recover the bucket from the reason string
    reason = (pred.reason or "").lower()
    if "gold query failed" in reason:
        return "gold_error"
    if "row count" in reason:
        return "wrong_row_count"
    if "pred error" in reason or "timeout" in reason:
        return "not_executable"
    if "row values differ" in reason:
        return "wrong_values"
    return "wrong_values"


# --------------------------------------------------------------------------- #
# bootstrap confidence interval
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 17,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    ``values`` are per-example outcomes (0/1 for accuracy, or any numeric score).
    The point of reporting this next to an accuracy number is honesty: with
    n=40 dev questions the 95% interval is roughly +-10pp wide, so the report can
    say "we cannot resolve differences smaller than that".

    Deterministic for a fixed ``seed``; returns ``(lo, hi)`` and degrades to the
    point estimate when there is nothing to resample.
    """
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0)
    mean = sum(vals) / len(vals)
    if len(vals) == 1 or n <= 0:
        return (mean, mean)
    rng = random.Random(seed)
    k = len(vals)
    means: list[float] = []
    for _ in range(int(n)):
        total = 0.0
        for _ in range(k):
            total += vals[rng.randrange(k)]
        means.append(total / k)
    means.sort()
    lo = _percentile(means, alpha / 2.0)
    hi = _percentile(means, 1.0 - alpha / 2.0)
    return (round(lo, 6), round(hi, 6))


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


# --------------------------------------------------------------------------- #
# slicing helpers
# --------------------------------------------------------------------------- #
def _rate(flags: Iterable[bool]) -> float:
    items = [bool(f) for f in flags]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _difficulty(pred: Prediction) -> float:
    """Difficulty of the underlying example (0.0 when unknown)."""
    meta = pred.meta or {}
    try:
        return float(meta.get("difficulty", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _kind(pred: Prediction) -> str:
    return str((pred.meta or {}).get("kind") or "raw")


def _difficulty_quartiles(preds: Sequence[Prediction],
                          scores: Sequence[Any] | None = None
                          ) -> dict[str, dict[str, Any]]:
    """Rank-based difficulty quartiles (Q1 = easiest ... Q4 = hardest).

    Rank-based bucketing (instead of fixed score cut points) guarantees four
    non-empty buckets on tiny dev sets and is robust to ties in the heuristic
    difficulty score, which are common because the score is rounded.

    When ``scores`` (parallel to ``preds``) is supplied, partial-credit metrics
    are included per bucket so the report can show whether the model is closer to
    correct on easy questions even when nothing executes.
    """
    n = len(preds)
    if n == 0:
        return {}
    order = sorted(range(n), key=lambda i: _difficulty(preds[i]))
    buckets: dict[str, list[int]] = {}
    for rank, idx in enumerate(order):
        bucket = min(3, (rank * 4) // n) + 1
        buckets.setdefault(f"Q{bucket}", []).append(idx)

    out: dict[str, dict[str, Any]] = {}
    for key, idxs in sorted(buckets.items()):
        items = [preds[i] for i in idxs]
        entry: dict[str, Any] = {
            "n": len(items),
            "exec_match": sum(int(p.exec_match) for p in items),
            "exact_match": sum(int(p.exact_match) for p in items),
            "executable": sum(int(p.executable) for p in items),
            "exec_acc": round(_rate(p.exec_match for p in items), 6),
            "exact_match_rate": round(_rate(p.exact_match for p in items), 6),
            "executable_rate": round(_rate(p.executable for p in items), 6),
            "difficulty_min": round(min(_difficulty(p) for p in items), 4),
            "difficulty_max": round(max(_difficulty(p) for p in items), 4),
        }
        if scores is not None:
            sub = [scores[i] for i in idxs]
            entry.update(_partial_slice(sub))
        out[key] = entry
    return out


def _partial_slice(scores: Sequence[Any]) -> dict[str, Any]:
    """Compact partial-credit block for one slice (difficulty bucket / db / kind)."""
    from .partial import aggregate_partial  # noqa: PLC0415 - local import

    agg = aggregate_partial(scores)
    return {
        "token_acc": agg["token_acc"],
        "prefix_match": agg["prefix_match"],
        "skeleton_match": agg["skeleton_match"],
        "table_set_match": agg["table_set_match"],
        "column_f1": agg["column_f1"],
        "clause_match_mean": agg["clause_match_mean"],
    }


def _group(preds: Sequence[Prediction], key_fn: Any,
           scores: Sequence[Any] | None = None) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(preds):
        groups.setdefault(key_fn(p), []).append(i)
    out: dict[str, dict[str, Any]] = {}
    for key, idxs in sorted(groups.items()):
        items = [preds[i] for i in idxs]
        entry: dict[str, Any] = {
            "n": len(items),
            "exec_match": sum(int(p.exec_match) for p in items),
            "exact_match": sum(int(p.exact_match) for p in items),
            "executable": sum(int(p.executable) for p in items),
            "exec_acc": round(_rate(p.exec_match for p in items), 6),
            "exact_match_rate": round(_rate(p.exact_match for p in items), 6),
            "executable_rate": round(_rate(p.executable for p in items), 6),
        }
        if scores is not None:
            entry.update(_partial_slice([scores[i] for i in idxs]))
        out[key] = entry
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def summarize(predictions: Sequence[Prediction], *, seed: int = 17,
              n_boot: int = 1000, databases: dict[str, Any] | None = None
              ) -> dict[str, Any]:
    """Full metric summary for a prediction set (JSON-serializable).

    Strict metrics (``exec_acc``, ``exact_match``, ``executable_rate``) stay
    exactly as they are -- they are all-or-nothing and must not be softened.

    Alongside them this reports **partial-credit** metrics
    (:mod:`fcdata.eval.partial`), because strict execution accuracy is ~0 for any
    model below ~99% token accuracy and therefore cannot show a learning signal
    at reference-model scale. See that module's docstring for the arithmetic.

    Parameters
    ----------
    databases:
        ``{db_id: Database}`` used for schema-aware analysis (unknown-table /
        column detection, table-set match). When omitted, partial metrics are
        still computed from the SQL text alone, minus schema-error attribution.

    Keys: ``n``, strict metrics + CIs, ``failures`` taxonomy, ``by_difficulty``,
    ``by_db``, ``by_kind``, and the partial-credit block (``token_acc``,
    ``skeleton_match``, ``table_set_match``, ``column_f1``, ``clause_match``,
    ``failure_taxonomy``, plus bootstrap CIs for the key partial metrics).
    """
    from .partial import (  # noqa: PLC0415 - local: keeps import graph acyclic
        aggregate_partial,
        partial_metric_values,
        score_prediction,
    )

    preds = list(predictions)
    n = len(preds)
    exec_flags = [bool(p.exec_match) for p in preds]
    em_flags = [bool(p.exact_match) for p in preds]
    exe_flags = [bool(p.executable) for p in preds]

    failures = {k: 0 for k in FAILURE_TAXONOMY}
    for p in preds:
        bucket = classify_failure(p)
        if bucket:
            failures[bucket] += 1
    failure_rows = {
        k: {"n": v, "rate": round(v / n, 6) if n else 0.0}
        for k, v in failures.items()
    }
    failure_rows["passed"] = {
        "n": sum(exec_flags),
        "rate": round(_rate(exec_flags), 6),
    }

    ci_lo, ci_hi = bootstrap_ci(exec_flags, n=n_boot, seed=seed)
    em_lo, em_hi = bootstrap_ci(em_flags, n=n_boot, seed=seed)

    # ---- partial credit (the learning signal) --------------------------
    scores = [score_prediction(p, (databases or {}).get(p.db_id)) for p in preds]
    partial = aggregate_partial(scores)
    partial_ci: dict[str, list[float]] = {}
    for metric in ("token_acc", "skeleton_match", "table_set_match",
                   "column_f1", "clause_match_mean", "prefix_match"):
        vals = partial_metric_values(scores, metric)
        lo, hi = bootstrap_ci(vals, n=n_boot, seed=seed)
        partial_ci[metric] = [lo, hi]
    partial["ci"] = partial_ci

    return {
        "n": n,
        "exec_acc": round(_rate(exec_flags), 6),
        "exact_match": round(_rate(em_flags), 6),
        "executable_rate": round(_rate(exe_flags), 6),
        "n_exec_match": sum(exec_flags),
        "n_exact_match": sum(em_flags),
        "exec_acc_ci": [ci_lo, ci_hi],
        "exact_match_ci": [em_lo, em_hi],
        "exec_acc_ci_width": round(ci_hi - ci_lo, 6),
        "failures": failure_rows,
        "partial": partial,
        "by_difficulty": _difficulty_quartiles(preds, scores),
        "by_db": _group(preds, lambda p: p.db_id, scores),
        "by_kind": _group(preds, _kind, scores),
    }


def format_summary_line(metrics: dict[str, Any]) -> str:
    """One-line human summary used by CLI progress output.

    Strict accuracy is printed first (it is the headline claim), followed by the
    partial-credit signals that show learning when strict accuracy is 0.
    """
    lo, hi = metrics.get("exec_acc_ci", [0.0, 0.0])
    line = (
        f"n={metrics.get('n', 0)} "
        f"exec_acc={metrics.get('exec_acc', 0.0) * 100:.1f}% "
        f"[{lo * 100:.1f}, {hi * 100:.1f}] "
        f"EM={metrics.get('exact_match', 0.0) * 100:.1f}% "
        f"executable={metrics.get('executable_rate', 0.0) * 100:.1f}%"
    )
    partial = metrics.get("partial") or {}
    if partial:
        line += (
            f" | token_acc={partial.get('token_acc', 0.0) * 100:.1f}%"
            f" skeleton={partial.get('skeleton_match', 0.0) * 100:.1f}%"
            f" tables={partial.get('table_set_match', 0.0) * 100:.1f}%"
            f" col_f1={partial.get('column_f1', 0.0) * 100:.1f}%"
        )
    return line
