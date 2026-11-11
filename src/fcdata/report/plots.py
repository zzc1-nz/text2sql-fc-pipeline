"""Headline figures: cost-vs-quality and the data-scale curve.

WHY these two figures exist
---------------------------
A resume README gets skimmed in about eight seconds. Two charts have to carry the
whole argument:

``cost_vs_quality.png``
    x = synthesis tokens invested, y = measured execution accuracy, one annotated
    point per ablation config plus a fitted trend line. The story is "spending
    tokens on synthesis moved accuracy from A to B" -- and because the x axis is
    real ledger tokens and the y axis is real sqlite execution accuracy, the
    chart cannot be inflated.

``scale_curve.png``
    dual axis: accuracy against synthesis tokens (left/primary) and against SFT
    dataset size (right/secondary), so a reader sees both "how much data" and
    "how much money" on one plot, plus where the curve saturates.

Implementation notes
--------------------
* matplotlib is forced onto the ``Agg`` backend *before* pyplot is imported, so
  the module is safe on a headless box and in CI.
* Every figure is deterministic: no random jitter, fixed figure size, fixed
  colours, ``bbox_inches="tight"``, 160 dpi.
* A trend line is only drawn when there are at least two distinct x values; with
  one point we say so in the legend instead of drawing a meaningless fit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import; keeps the module headless

import matplotlib.pyplot as plt  # noqa: E402  (backend set above by necessity)
from matplotlib.figure import Figure  # noqa: E402

#: 160 dpi is the portfolio default: sharp on GitHub retina rendering, still small.
DPI = 160

#: Deterministic palette (colour-blind safe-ish, fixed order).
_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
_GRID = dict(color="#cccccc", linewidth=0.6, alpha=0.7)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save(fig: Figure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _fmt_tokens(v: float) -> str:
    """Compact token label: 1_200_000 -> '1.2M', 45_000 -> '45k'."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Least-squares ``(slope, intercept)``; ``None`` when degenerate."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (slope, my - slope * mx)


def _r_squared(xs: Sequence[float], ys: Sequence[float],
               slope: float, intercept: float) -> float:
    n = len(xs)
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot == 0.0:
        return 0.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return max(0.0, 1.0 - ss_res / ss_tot)


def _zero_accuracy_note(accs: Sequence[float]) -> str:
    """Explain an all-zero y axis instead of letting it look like a plotting bug.

    When every measured accuracy is 0 the chart is a flat line on the x axis. A
    reader who does not know that would reasonably assume the plot failed. The
    figure says so explicitly, in the same words the report uses.
    """
    if not accs or any(a > 0.0 for a in accs):
        return ""
    return ("All measured execution accuracies are 0.0%: the model produced no "
            "executable-and-correct SQL, so there is no capability signal to plot.")


def _draw_note(ax: Any, text: str, *, where: str = "top") -> None:
    """Place an explanatory note clear of the plotted data.

    ``where`` positions it: ``top`` just under the axes title, ``bottom`` above
    the x axis, ``center`` in the middle of an empty axes. The text is wrapped to
    a width that fits inside the axes so it never collides with the y-axis label.
    """
    if not text:
        return
    y = 0.97 if where == "top" else 0.13
    va = "top" if where == "top" else "bottom"
    if where == "center":
        y, va = 0.5, "center"
    ax.text(0.5, y, _wrap_note(text), transform=ax.transAxes, fontsize=7.2,
            color="#a03030", ha="center", va=va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4f4",
                      edgecolor="#e0b0b0", linewidth=0.7))


def _wrap_note(text: str, width: int = 62) -> str:
    """Hard-wrap an annotation so it stays inside the axes."""
    import textwrap  # noqa: PLC0415 - stdlib, only needed for annotation text

    return "\n".join(textwrap.wrap(text, width=width))


def _style_axes(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, **_GRID)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# figure 1: cost vs quality (ablation)
# --------------------------------------------------------------------------- #
def plot_cost_vs_quality(rows: Sequence[dict[str, Any]],
                         out_path: str | Path,
                         *,
                         title: str = "Synthesis token investment vs. execution accuracy",
                         note: str = "") -> Path:
    """Scatter the ablation rungs (x=synth tokens, y=exec acc) with a trend line.

    ``rows`` are the ``ablation.json`` row dicts; each needs ``config``,
    ``synth_tokens``, ``exec_acc`` and optionally ``exec_acc_ci_lo/hi``.
    The x axis uses a symlog scale so the zero-cost baseline (raw_only) can share
    the axis with a million-token config without collapsing the plot.
    """
    pts = [r for r in rows if r]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if not pts:
        ax.text(0.5, 0.5, "no ablation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        _style_axes(ax, "Synthesis tokens invested", "Execution accuracy", title)
        return _save(fig, out_path)

    xs = [float(r.get("synth_tokens", 0) or 0) for r in pts]
    ys = [float(r.get("exec_acc", 0.0) or 0.0) * 100.0 for r in pts]
    # The zero-cost baseline shares the axis with million-token configs, so a
    # plain linear axis would collapse everything to the left edge and a plain
    # log axis cannot represent 0 at all. ``symlog`` with a small linear region
    # keeps the baseline at x=0 and the true positions of every other point.
    max_x = max(xs) if xs else 0.0
    linthresh = max(1.0, max_x * 0.02)
    plot_x = list(xs)

    for i, (r, x, y) in enumerate(zip(pts, plot_x, ys)):
        color = _COLORS[i % len(_COLORS)]
        lo = r.get("exec_acc_ci_lo")
        hi = r.get("exec_acc_ci_hi")
        if lo is not None and hi is not None:
            ax.errorbar([x], [y], yerr=[[y - float(lo) * 100.0], [float(hi) * 100.0 - y]],
                        fmt="none", ecolor=color, elinewidth=1.1, capsize=3, alpha=0.75)
        ax.scatter([x], [y], s=68, color=color, zorder=3, edgecolor="white", linewidth=0.8)
        label = str(r.get("config", f"cfg{i}"))
        ax.annotate(f"{label}\n{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=7.5, color="#222222",
                    ha="left", va="bottom")

    fit = _linear_fit(plot_x, ys)
    if fit is not None and len(set(round(y, 6) for y in ys)) > 1:
        slope, intercept = fit
        x0, x1 = min(plot_x), max(plot_x)
        ax.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
                linestyle="--", linewidth=1.3, color="#555555", alpha=0.85, zorder=2)
        r2 = _r_squared(plot_x, ys, slope, intercept)
        # express the slope per 100k tokens so the number is readable
        per_100k = slope * 100_000.0
        ax.plot([], [], linestyle="--", color="#555555",
                label=f"linear fit: {per_100k:+.1f} pp / 100k tokens (R²={r2:.2f})")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    else:
        # No trend line to explain. Showing a legend that only says "no trend
        # line" would collide with the explanatory note; omit it.
        pass

    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ticks = sorted({round(x, 3) for x in plot_x})
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_tokens(t) for t in ticks])
    ax.minorticks_off()
    _style_axes(ax, "Synthesis tokens invested (symlog scale)", "Execution accuracy (%)", title)
    # Always show a meaningful accuracy window. Autoscaling a 0-0 series produces
    # a +-0.04% axis, which looks like noise rather than a real zero.
    ax.set_ylim(0.0, max(10.0, min(100.0, max(ys) * 1.6 if ys else 10.0)))
    _draw_note(ax, _zero_accuracy_note(ys))
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7, color="#666666")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# figure 2: scale curve (dual axis)
# --------------------------------------------------------------------------- #
def plot_scale_curve(points: Sequence[dict[str, Any]],
                     out_path: str | Path,
                     *,
                     title: str = "Data-scale curve: accuracy vs. synthesis tokens and SFT size",
                     note: str = "") -> Path:
    """One accuracy curve with two cost axes: tokens (bottom) and records (top).

    WHY a secondary axis rather than two lines: in this sweep synthesis tokens and
    SFT dataset size are monotonically related (more synthetic records cost more
    tokens), so drawing two curves would plot the *same* accuracy measurements
    twice against two different x variables and invite a reader to read a
    spurious second trend. Instead the single measured accuracy curve is drawn
    against the real token cost, and a top axis shows the dataset size that
    corresponded to each token count. Both readings are exact -- the top axis is
    a piecewise-linear mapping through the measured points, not an estimate.
    """
    pts = [p for p in points if p]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not pts:
        ax.text(0.5, 0.5, "no scale data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        _style_axes(ax, "Synthesis tokens invested", "Execution accuracy", title)
        return _save(fig, out_path)

    ordered = sorted(pts, key=lambda p: float(p.get("synth_tokens", 0) or 0))
    tokens = [float(p.get("synth_tokens", 0) or 0) for p in ordered]
    n_recs = [float(p.get("n_sft_records", 0) or 0) for p in ordered]
    accs = [float(p.get("exec_acc", 0.0) or 0.0) * 100.0 for p in ordered]
    los = [p.get("exec_acc_ci_lo") for p in ordered]
    his = [p.get("exec_acc_ci_hi") for p in ordered]

    color = _COLORS[0]
    if all(v is not None for v in los) and all(v is not None for v in his):
        yerr = [[a - float(lo) * 100.0 for a, lo in zip(accs, los)],
                [float(hi) * 100.0 - a for a, hi in zip(accs, his)]]
        ax.errorbar(tokens, accs, yerr=yerr, marker="o", markersize=6,
                    linewidth=1.8, capsize=3, color=color,
                    label="measured exec accuracy (95% bootstrap CI)")
    else:
        ax.plot(tokens, accs, marker="o", markersize=6, linewidth=1.8, color=color,
                label="measured exec accuracy")
    for x, y, p in zip(tokens, accs, ordered):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(6, 7),
                    fontsize=7.5, color=color)

    _style_axes(ax, "Synthesis tokens invested", "Execution accuracy (%)", title)
    # A fixed, meaningful accuracy window: autoscaling an all-zero series would
    # produce a +-0.04% axis that reads as noise instead of a genuine zero.
    ax.set_ylim(0.0, max(10.0, min(100.0, max(accs) * 1.25 if accs else 10.0)))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{int(round(v)):,}"))
    # Legend upper-left: the measured points sit on the x axis, so the lower
    # half holds the annotations and the top edge belongs to the secondary axis.
    if _zero_accuracy_note(accs):
        # All-zero run: the whole plot area is empty, so put the explanation in
        # the middle and drop the legend (it would only describe a flat line).
        ax.legend().remove() if ax.get_legend() else None
    else:
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # ---- top axis: SFT dataset size at the same measured points ------------
    mapping = _TokenRecordMapping(tokens, n_recs)
    if mapping.ok:
        top = ax.secondary_xaxis("top", functions=(mapping.to_records, mapping.to_tokens))
        top.set_xlabel("SFT dataset size (records, same measured points)", fontsize=9,
                       color=_COLORS[1], labelpad=8)
        top.tick_params(axis="x", labelsize=8, colors=_COLORS[1], pad=2)
    else:
        ax.text(0.99, 1.02, "dataset size unavailable", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#888888")

    _draw_note(ax, _zero_accuracy_note(accs), where="center")
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7, color="#666666")
    return _save(fig, out_path)


class _TokenRecordMapping:
    """Monotone piecewise-linear map between synth tokens and SFT record count.

    Built only from measured points, so the top axis of the scale figure is a
    faithful re-reading of the same data rather than an interpolation model.
    Values outside the measured range are clamped (never extrapolated) so the
    axis cannot display a dataset size that was never trained on.
    """

    def __init__(self, tokens: Sequence[float], records: Sequence[float]) -> None:
        pairs = sorted(zip(tokens, records))
        # collapse duplicate token counts (e.g. two fractions with 0 tokens)
        merged: list[tuple[float, float]] = []
        for t, r in pairs:
            if merged and merged[-1][0] == t:
                merged[-1] = (t, max(merged[-1][1], r))
            else:
                merged.append((t, r))
        self._tokens = [p[0] for p in merged]
        self._records = [p[1] for p in merged]
        self.ok = len(self._tokens) >= 2 and self._tokens[-1] > self._tokens[0]

    def to_records(self, t: Any) -> Any:
        import numpy as np  # noqa: PLC0415 - local: keeps matplotlib-only paths light

        return np.interp(np.asarray(t, dtype=float), self._tokens, self._records)

    def to_tokens(self, r: Any) -> Any:
        import numpy as np  # noqa: PLC0415 - local import

        return np.interp(np.asarray(r, dtype=float), self._records, self._tokens)


# --------------------------------------------------------------------------- #
# figure 4: cost vs partial credit (three panels)
# --------------------------------------------------------------------------- #
#: Panels for the partial-credit figure: (metric key, panel title, short note).
PARTIAL_PANELS: tuple[tuple[str, str, str], ...] = (
    ("table_set_match",
     "A. Table-level schema grounding",
     "did the model pick the right tables"),
    ("executable_rate",
     "B. Executability",
     "did the generated SQL run at all"),
    ("token_acc",
     "C. Exact-token fidelity",
     "how close the generated SQL text is"),
)


def plot_cost_vs_partial(rows: Sequence[dict[str, Any]],
                         out_path: str | Path,
                         *,
                         title: str = "Partial credit: what improved when exec accuracy did not",
                         note: str = "") -> Path:
    """Three stacked panels: each partial metric against synthesis tokens.

    WHY this figure exists alongside the flat ``cost_vs_quality.png``: strict
    execution accuracy is all-or-nothing, so at reference-model scale it is a flat
    line at 0% -- correct, but it shows nothing. The partial-credit metrics are
    where the measured effect actually lives, so they get their own figure rather
    than being buried in a table. The flat figure is kept: it is the honest
    statement of the headline metric.

    Panels share the x axis (synthesis tokens, symlog so the zero-cost baseline
    sits at 0). Each point is annotated with its config name and value, because
    with three points the annotation *is* the legend.
    """
    pts = [r for r in rows if r]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.6), sharex=True)
    if not pts:
        for ax in axes:
            ax.text(0.5, 0.5, "no ablation data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888888")
            ax.set_axis_off()
        fig.suptitle(title, fontsize=11, fontweight="bold")
        return _save(fig, out_path)

    xs = [float(r.get("synth_tokens", 0) or 0) for r in pts]
    max_x = max(xs) if xs else 0.0
    linthresh = max(1.0, max_x * 0.02)

    for ax, (metric, panel_title, note_txt) in zip(axes, PARTIAL_PANELS):
        ys = [float(r.get(metric, 0.0) or 0.0) * 100.0 for r in pts]
        color = _COLORS[0]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=1.8, color=color,
                zorder=3, markerfacecolor="white", markeredgewidth=1.6)
        # Stagger labels vertically when neighbouring values are close: with only
        # three points, two configs frequently land on the same y and their
        # annotations would otherwise overlap into an unreadable smear.
        last_y: float | None = None
        flip = 1
        for x, y, r in zip(xs, ys, pts):
            frac = (x / max_x) if max_x > 0 else 0.0
            if frac >= 0.75:
                ha = "right"
                dx = -10
            else:
                ha = "left"
                dx = 10
            if last_y is not None and abs(y - last_y) < 2.5:
                dy = 16 * flip
                flip = -flip
            else:
                dy = 4
                flip = 1
            ax.annotate(f"{r.get('config', '')}\n{y:.1f}%", (x, y),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=7.5, color="#222222", ha=ha, va="center")
            last_y = y
        ax.set_ylabel(f"{metric} (%)", fontsize=8.5)
        ax.set_title(panel_title, fontsize=9.5, fontweight="bold", loc="left")
        ax.text(0.99, 0.04, note_txt, transform=ax.transAxes, fontsize=7,
                color="#777777", ha="right", va="bottom", style="italic")
        ax.grid(True, **_GRID)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # headroom so annotations never clip
        top = max(ys) if ys else 0.0
        ax.set_ylim(0.0, max(10.0, min(100.0, top * 1.45)))

    axes[-1].set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ticks = sorted({round(x, 3) for x in xs})
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_fmt_tokens(t) for t in ticks])
    axes[-1].minorticks_off()
    # Extra room on the right so the last point's label is never clipped.
    right_pad = max(1.0, max_x * 2.2) if max_x > 0 else 10.0
    axes[-1].set_xlim(left=0.0, right=max_x + right_pad)
    axes[-1].set_xlabel("Synthesis tokens invested (symlog scale)", fontsize=9)

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# figure 3: training loss curve
# --------------------------------------------------------------------------- #
def plot_loss_curve(metrics: dict[str, Any],
                    out_path: str | Path,
                    *,
                    title: str = "LoRA SFT: training loss and token accuracy") -> Path:
    """Loss (left axis) and token accuracy (right axis) from a run's metrics.json.

    Accepts ``{"steps": [{"loss":..,"token_acc":..}, ...]}`` or parallel
    ``{"loss": [...], "token_acc": [...]}``; tolerates missing series so a
    partially-written metrics file still produces a figure.
    """
    losses, accs = _extract_curves(metrics)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if not losses:
        ax.text(0.5, 0.5, "no training steps recorded", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        _style_axes(ax, "step", "loss", title)
        return _save(fig, out_path)

    steps = list(range(1, len(losses) + 1))
    ax.plot(steps, losses, linewidth=1.6, color=_COLORS[0], label="train loss")
    _style_axes(ax, "optimizer step", "cross-entropy loss (supervised tokens only)", title)

    if accs:
        ax2 = ax.twinx()
        ax2.plot(steps[:len(accs)], [a * 100.0 for a in accs], linewidth=1.2,
                 linestyle="--", color=_COLORS[2], label="token accuracy")
        ax2.set_ylabel("token accuracy (%)", fontsize=9, color=_COLORS[2])
        ax2.tick_params(axis="y", labelsize=8, colors=_COLORS[2])
        ax2.set_ylim(0, 100)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5, framealpha=0.9)
    else:
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    return _save(fig, out_path)


def _extract_curves(metrics: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Pull ``(losses, token_accs)`` out of the several shapes metrics.json uses.

    The reference trainer writes ``history: [{step, epoch, loss, token_acc}, ...]``
    (per-step, the canonical shape). Parallel-list shapes are accepted too so a
    loss curve can be drawn from an HF trainer's log history without conversion.
    """
    if not isinstance(metrics, dict):
        return ([], [])
    for key in ("history", "steps"):
        seq = metrics.get(key)
        if isinstance(seq, list) and seq:
            entries = [s for s in seq if isinstance(s, dict)]
            losses = [float(s.get("loss", 0.0)) for s in entries if "loss" in s]
            accs = [float(s.get("token_acc", 0.0)) for s in entries if "token_acc" in s]
            if losses:
                return (losses, accs)
    for lk, ak in (("loss", "token_acc"), ("losses", "token_accs"),
                   ("train_loss", "train_token_acc"), ("step_loss", "step_token_acc")):
        lv = metrics.get(lk)
        av = metrics.get(ak)
        if isinstance(lv, list) and lv:
            losses = [float(v) for v in lv]
            accs = [float(v) for v in av] if isinstance(av, list) else []
            return (losses, accs)
    return ([], [])


# --------------------------------------------------------------------------- #
# convenience: build every figure the report needs
# --------------------------------------------------------------------------- #
def make_all_figures(reports_dir: str | Path,
                     *,
                     ablation: dict[str, Any] | None = None,
                     scale: dict[str, Any] | None = None,
                     train_metrics: dict[str, Any] | None = None,
                     note: str = "") -> dict[str, str]:
    """Render all figures that have data; returns ``{name: path}``.

    Missing inputs are skipped (not an error) so a partial pipeline run still
    produces the figures it can.
    """
    out = Path(reports_dir)
    made: dict[str, str] = {}
    if ablation and ablation.get("rows"):
        p = plot_cost_vs_quality(ablation["rows"], out / "cost_vs_quality.png", note=note)
        made["cost_vs_quality"] = str(p)
        if any(float(r.get("token_acc", 0.0) or 0.0) > 0.0
               or float(r.get("table_set_match", 0.0) or 0.0) > 0.0
               for r in ablation["rows"]):
            p2 = plot_cost_vs_partial(ablation["rows"],
                                      out / "cost_vs_partial.png", note=note)
            made["cost_vs_partial"] = str(p2)
    if scale and scale.get("points"):
        p = plot_scale_curve(scale["points"], out / "scale_curve.png", note=note)
        made["scale_curve"] = str(p)
    if train_metrics:
        losses, _ = _extract_curves(train_metrics)
        if losses:
            p = plot_loss_curve(train_metrics, out / "loss_curve.png")
            made["loss_curve"] = str(p)
    return made
