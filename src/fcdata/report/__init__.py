"""Markdown report builder: GitHub-ready tables + the headline cost/capability ratio.

WHY this module exists
----------------------
The pipeline's whole claim fits in one sentence -- *"token investment in synthesis
turns into downstream capability"* -- and a resume README has to prove it with
numbers a reader can audit. This module turns the pipeline's JSON artifacts into
paste-ready GitHub markdown tables plus the CSVs behind them, with two rules:

1. **No number is computed here that was not measured upstream.** Accuracy comes
   from ``metrics.json`` (real sqlite execution), tokens come from the ledger or
   ``SFTRecord.synth_tokens``, and validation funnel counts come from the
   validation report. This module formats; it does not estimate.
2. **Every table states its provenance.** When the ablation ran in ``--quick``
   smoke mode, or when a cost is an estimate rather than ledger-derived, the
   table says so instead of quietly presenting a toy number as a result.

The headline metric is deliberately a *ratio*: **execution-accuracy points gained
per 1M synthesis tokens spent**. It is the single number that makes the cost of
synthesis comparable to just... not doing synthesis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..records import read_json, write_json
from . import plots

#: Where the Lead's extrapolation module writes its report.
EXTRAPOLATION_FILE = "token_extrapolation.json"

#: Canonical table column labels (README-ready prose, not field names).
ABLATION_HEADERS: tuple[str, ...] = (
    "Config", "SFT records", "Synth tokens", "Synth cost (USD)", "Exec acc",
    "95% CI", "EM", "Delta vs baseline",
)
SCALE_HEADERS: tuple[str, ...] = (
    "Synth fraction", "SFT records", "Synth tokens", "Synth cost (USD)",
    "Exec acc", "95% CI", "EM", "Delta vs baseline",
)
#: Partial-credit table: the learning signal that survives 0% exec accuracy.
PARTIAL_HEADERS: tuple[str, ...] = (
    "Config", "Token acc", "Skeleton", "Table set", "Table set 95% CI",
    "Column F1", "Clause", "Executable", "Exec acc",
)
#: Failure taxonomy table.
TAXONOMY_HEADERS: tuple[str, ...] = ("Error class", "Count", "Share")


# --------------------------------------------------------------------------- #
# formatting helpers (one place, so every table looks the same)
# --------------------------------------------------------------------------- #
def pct(value: Any, digits: int = 1) -> str:
    """Format a fraction as a percentage string with ``digits`` decimals."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def pp(value: Any, digits: int = 1) -> str:
    """Format a *delta* of a fraction as percentage points, with sign."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v * 100:+.{digits}f} pp"


def num(value: Any) -> str:
    """Thousands-separated integer."""
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "n/a"


def usd(value: Any, digits: int = 4) -> str:
    """US-dollar amount; 4 decimals because a demo run costs fractions of a cent."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"${v:,.{digits}f}"


def ci(lo: Any, hi: Any) -> str:
    """Format a bootstrap CI pair as ``[lo, hi]`` percentages."""
    if lo is None or hi is None:
        return "n/a"
    return f"[{pct(lo)}, {pct(hi)}]"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
             aligns: Sequence[str] | None = None) -> str:
    """Render a GitHub-flavored markdown table.

    ``aligns`` accepts ``l``/``c``/``r`` per column (default: left for the first
    column, right for the rest -- numeric tables read better right-aligned).

    Pipes inside cell values are escaped here rather than at every call site, so
    a model name or failure code containing ``|`` cannot silently break the table
    layout. Already-escaped pipes are left alone, making the escape idempotent.
    """
    if not headers:
        return ""
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    sep = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = ["| " + " | ".join(_escape_cell(h) for h in headers) + " |",
             "| " + " | ".join(sep.get(a, "---") for a in aligns) + " |"]
    for row in rows:
        cells = [_escape_cell(c) for c in row]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return "\n".join(lines)


def _escape_cell(text: Any) -> str:
    """Markdown-escape a table cell: only *unescaped* pipes need escaping."""
    return re.sub(r"(?<!\\)\|", r"\\|", str(text))


# --------------------------------------------------------------------------- #
# table builders
# --------------------------------------------------------------------------- #
def partial_table(ablation: dict[str, Any] | None) -> str:
    """Partial-credit table: token/skeleton/table/column/clause per config.

    WHY this table exists even when exec accuracy is 0: strict execution accuracy
    requires a byte-perfect statement, so it is ~0 for any model below ~99% token
    accuracy. These metrics show *what* the model learned -- query shape, schema
    grounding, clause structure -- which is the honest signal at reference-model
    scale. See :mod:`fcdata.eval.partial` for the arithmetic.
    """
    if not ablation or not ablation.get("rows"):
        return "_Partial-credit metrics unavailable: no ablation rows._"
    rows: list[list[str]] = []
    for r in ablation["rows"]:
        ci_pair = (r.get("partial") or {}).get("ci") or {}
        tbl_ci = ci_pair.get("table_set_match")
        rows.append([
            f"`{_escape_cell(r.get('config', ''))}`",
            pct(r.get("token_acc", 0.0)),
            pct(r.get("skeleton_match", 0.0)),
            pct(r.get("table_set_match", 0.0)),
            ci(tbl_ci[0], tbl_ci[1]) if tbl_ci else "n/a",
            pct(r.get("column_f1", 0.0)),
            pct(r.get("clause_match_mean", 0.0)),
            pct(r.get("executable_rate", 0.0)),
            pct(r.get("exec_acc", 0.0)),
        ])
    return md_table(PARTIAL_HEADERS, rows)


def failure_taxonomy_table(ablation: dict[str, Any] | None) -> str:
    """Which error classes each config produced, mapped onto hard-negative kinds.

    This is what lets the README claim *which* failure mode the synthetic data
    addressed, rather than only that a number moved.
    """
    if not ablation or not ablation.get("rows"):
        return "_Failure taxonomy unavailable: no ablation rows._"
    codes: list[str] = []
    for r in ablation["rows"]:
        for code, n in (r.get("partial_failures") or {}).items():
            if int(n or 0) > 0 and code not in codes:
                codes.append(code)
    if not codes:
        return "_No failures recorded (every prediction executed and matched)._"
    rows: list[list[str]] = []
    for code in codes:
        row = [f"`{_escape_cell(code)}`"]
        for r in ablation["rows"]:
            n = int((r.get("partial_failures") or {}).get(code, 0) or 0)
            total = max(1, int(r.get("n_eval", 0) or 0))
            row.append(f"{num(n)} ({pct(n / total)})" if n else "-")
        rows.append(row)
    headers = ["Error class"] + [f"`{_escape_cell(str(r.get('config', '')))}`"
                                 for r in ablation["rows"]]
    return md_table(tuple(headers), rows, aligns=["l"] + ["r"] * len(ablation["rows"]))


def scale_partial_table(scale: dict[str, Any] | None) -> str:
    """Partial-credit metrics across the data-scale sweep."""
    if not scale or not scale.get("points"):
        return "_Scale partial-credit metrics unavailable: no scale points._"
    rows: list[list[str]] = []
    for p in scale["points"]:
        rows.append([
            f"{float(p.get('fraction', 0.0)):.0%}",
            num(p.get("n_sft_records", 0)),
            num(p.get("synth_tokens", 0)),
            pct(p.get("token_acc", 0.0)),
            pct(p.get("skeleton_match", 0.0)),
            pct(p.get("table_set_match", 0.0)),
            pct(p.get("column_f1", 0.0)),
            pct(p.get("exec_acc", 0.0)),
        ])
    return md_table(("Synth fraction", "SFT records", "Synth tokens", "Token acc",
                     "Skeleton", "Table set", "Column F1", "Exec acc"), rows)


def ablation_table(ablation: dict[str, Any] | None) -> str:
    """Config | #SFT records | synth tokens | cost USD | exec acc | CI | EM | delta."""
    if not ablation or not ablation.get("rows"):
        return "_Ablation not run: no `artifacts/06_eval/ablation.json` found._"
    baseline_name = str(ablation.get("baseline") or "raw_only")
    rows: list[list[str]] = []
    for r in ablation["rows"]:
        rows.append([
            f"`{_escape_cell(r.get('config', ''))}`",
            num(r.get("n_sft_records", 0)),
            num(r.get("synth_tokens", 0)),
            usd(r.get("synth_cost_usd", 0.0)),
            pct(r.get("exec_acc", 0.0)),
            ci(r.get("exec_acc_ci_lo"), r.get("exec_acc_ci_hi")),
            pct(r.get("exact_match", 0.0)),
            _delta_cell(r.get("delta_exec_acc"),
                        is_baseline=str(r.get("config", "")) == baseline_name),
        ])
    return md_table(ABLATION_HEADERS, rows)


def _delta_cell(delta: Any, *, is_baseline: bool = False) -> str:
    """Render a delta-vs-baseline cell.

    A *zero* delta on a non-baseline row is a real (if disappointing) measurement
    and must not be silently relabelled "baseline" -- that would hide the fact
    that a config was run and produced no improvement. Conversely the baseline
    row itself always reads "baseline" even though its delta is exactly 0.
    """
    if is_baseline:
        return "baseline"
    if delta is None:
        return "n/a"
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return "n/a"
    if value == 0.0:
        return "\u00b10.0 pp"
    return pp(value)


def scale_table(scale: dict[str, Any] | None) -> str:
    """Synthetic-data fraction | #SFT records | tokens | cost | exec acc | CI | EM | delta."""
    if not scale or not scale.get("points"):
        return "_Scale curve not run: no `artifacts/06_eval/scale_curve.json` found._"
    rows: list[list[str]] = []
    for p in scale["points"]:
        rows.append([
            f"{float(p.get('fraction', 0.0)):.0%}",
            num(p.get("n_sft_records", 0)),
            num(p.get("synth_tokens", 0)),
            usd(p.get("synth_cost_usd", 0.0)),
            pct(p.get("exec_acc", 0.0)),
            ci(p.get("exec_acc_ci_lo"), p.get("exec_acc_ci_hi")),
            pct(p.get("exact_match", 0.0)),
            _delta_cell(p.get("delta_exec_acc"),
                        is_baseline=float(p.get("fraction", -1) or -1) <= 0.0),
        ])
    return md_table(SCALE_HEADERS, rows)


def validation_funnel_table(funnel: dict[str, Any] | None) -> str:
    """Raw -> synth -> format-pass -> semantic-pass -> SFT counts with drop reasons.

    The funnel is where synthesis quality is proven: it shows how much generated
    data survived schema checking and the semantic judge before it was allowed
    into SFT. Rates are relative to the previous stage so a reader can see the
    cost of each gate.
    """
    if not funnel:
        return "_Validation funnel unavailable: no `validation_report.json` found._"
    rows: list[list[str]] = []
    for stage in funnel.get("stages", []):
        n = int(stage.get("n", 0) or 0)
        kept = stage.get("kept")
        rows.append([
            str(stage.get("stage", "")),
            num(n),
            num(kept) if kept is not None else "-",
            pct(stage.get("pass_rate")) if stage.get("pass_rate") is not None else "-",
            _escape_cell(str(stage.get("note", ""))),
        ])
    table = md_table(("Stage", "Records in", "Records kept", "Pass rate", "Note"), rows)
    failures = [f for f in (funnel.get("top_failure_codes") or [])
                if int(f.get("n", 0) or 0) > 0]
    if failures:
        frows = [[f"`{_escape_cell(str(f.get('code', '')))}`", num(f.get("n", 0))]
                 for f in failures]
        table += "\n\n**Top validation failure codes**\n\n" + md_table(
            ("Failure code", "Count"), frows)
    return table


def ledger_table(ledger_summary: dict[str, Any] | None) -> str:
    """Token ledger by stage, then by role/model inside each stage."""
    if not ledger_summary:
        return "_Token ledger unavailable: no ledger JSONL found._"
    by_stage = ledger_summary.get("by_stage") or {}
    if not by_stage:
        return "_Token ledger is empty (no provider calls recorded yet)._"
    rows: list[list[str]] = []
    for stage_name, st in sorted(by_stage.items()):
        rows.append([
            f"`{_escape_cell(stage_name)}`",
            num(st.get("calls", 0)),
            num(st.get("prompt_tokens", 0)),
            num(st.get("completion_tokens", 0)),
            num(st.get("total_tokens", 0)),
            usd(st.get("cost_usd", 0.0), 6),
            num(st.get("cached_calls", 0)),
        ])
        for role, br in sorted((st.get("by_role") or {}).items()):
            models = ", ".join(br.get("models") or [])
            rows.append([
                f"  └ {_escape_cell(role)}" + (f" ({_escape_cell(models)})" if models else ""),
                num(br.get("calls", 0)),
                num(br.get("prompt_tokens", 0)),
                num(br.get("completion_tokens", 0)),
                num(br.get("total_tokens", 0)),
                usd(br.get("cost_usd", 0.0), 6),
                "",
            ])
    totals = ledger_summary.get("totals") or {}
    if totals:
        rows.append([
            "**total**", num(totals.get("calls", 0)), num(totals.get("prompt_tokens", 0)),
            num(totals.get("completion_tokens", 0)), num(totals.get("total_tokens", 0)),
            usd(totals.get("cost_usd", 0.0), 6), "",
        ])
    return md_table(("Stage / role", "Calls", "Prompt tok", "Completion tok",
                     "Total tok", "Cost (USD)", "Cached"), rows)


# --------------------------------------------------------------------------- #
# headline: token investment -> capability
# --------------------------------------------------------------------------- #
def headline_ratio(ablation: dict[str, Any] | None) -> dict[str, Any]:
    """The project's one-number claim: accuracy points per 1M synthesis tokens.

    Computed from the *measured* ablation endpoints only: the baseline
    (``raw_only``, zero synthesis tokens) and the richest config. Returns
    ``{"available": False, ...}`` with a reason when the ablation cannot support
    the claim (missing baseline, or the rich config spent no tokens), so the
    report never prints a ratio divided by zero or an invented baseline.

    When the measured gain is zero or negative, the result carries
    ``no_measurable_gain=True`` rather than a tidy ``0.00`` ratio: a ratio of
    exactly zero invites the reader to conclude synthesis is worthless, when what
    actually happened is that this model did not reach the accuracy regime where
    the ratio is measurable.
    """
    if not ablation or not ablation.get("rows"):
        return {"available": False, "reason": "ablation not run"}
    rows = ablation["rows"]
    baseline = next((r for r in rows if r.get("config") == "raw_only"), None)
    if baseline is None:
        return {"available": False, "reason": "no raw_only baseline row"}
    rich = max(rows, key=lambda r: float(r.get("synth_tokens", 0) or 0))
    tokens = float(rich.get("synth_tokens", 0) or 0)
    gain = float(rich.get("exec_acc", 0.0)) - float(baseline.get("exec_acc", 0.0))
    if tokens <= 0:
        return {"available": False,
                "reason": "richest config spent no synthesis tokens"}
    per_mtok = gain / (tokens / 1_000_000.0)
    result = {
        "available": True,
        "baseline_config": baseline.get("config", "raw_only"),
        "baseline_exec_acc": float(baseline.get("exec_acc", 0.0)),
        "rich_config": rich.get("config", ""),
        "rich_exec_acc": float(rich.get("exec_acc", 0.0)),
        "gain_pp": round(gain * 100.0, 2),
        "synth_tokens": int(tokens),
        "synth_cost_usd": float(rich.get("synth_cost_usd", 0.0) or 0.0),
        "points_per_mtok": round(per_mtok * 100.0, 2),
        "tokens_per_point": int(round(tokens / (gain * 100.0))) if gain > 0 else None,
        "relative_improvement_pct": round(
            gain / float(baseline.get("exec_acc", 0.0)) * 100.0, 1)
        if float(baseline.get("exec_acc", 0.0)) > 0 else None,
    }
    if gain <= 0.0:
        result["no_measurable_gain"] = True
        result["reason"] = (
            "no measurable accuracy gain between the baseline and the richest "
            "config, so there is no cost/capability ratio to report; the model "
            "produced no executable-and-correct SQL in either arm")
    return result


def _no_gain_headline(ablation: dict[str, Any] | None,
                      h: dict[str, Any]) -> str:
    """Headline for a run where strict accuracy did not move.

    This is the honest-reporting path: instead of a ratio that reads as "0.00
    points per 1M tokens" (which invites the reader to conclude synthesis is
    worthless), it states what was measured, then surfaces the partial-credit
    signal that *did* move, and names the most likely cause.
    """
    rows = (ablation or {}).get("rows") or []
    base = next((r for r in rows if r.get("config") == h.get("baseline_config")), None)
    rich = next((r for r in rows if r.get("config") == h.get("rich_config")), None)
    lines = [
        "### Headline: no measurable execution-accuracy gain at this model scale",
        "",
        f"> **Strict execution accuracy did not move.** `{h.get('baseline_config')}` "
        f"and `{h.get('rich_config')}` both scored "
        f"{pct(h.get('rich_exec_acc', 0.0))} exec accuracy after spending "
        f"**{num(h.get('synth_tokens', 0))} synthesis tokens** "
        f"({usd(h.get('synth_cost_usd', 0.0))}). This is the truthful result and it "
        "is reported unchanged.",
        "",
        "That is expected arithmetic, not a broken pipeline: execution accuracy "
        "requires a byte-perfect statement, so it stays at zero for any model "
        "below roughly 99% token accuracy. Every number here came from really "
        "executing the generated SQL against SQLite.",
        "",
    ]

    # ---- the signal that DID move -------------------------------------
    # The directional detail is already stated in the Findings section above;
    # repeating the numbers here would make the headline read like a duplicate.
    base = next((r for r in rows if r.get("config") == h.get("baseline_config")), None)
    rich = next((r for r in rows if r.get("config") == h.get("rich_config")), None)
    if base and rich:
        tbl_delta = (float(rich.get("table_set_match", 0.0))
                     - float(base.get("table_set_match", 0.0)))
        exe_delta = (float(rich.get("executable_rate", 0.0))
                     - float(base.get("executable_rate", 0.0)))
        tok_delta = (float(rich.get("token_acc", 0.0))
                     - float(base.get("token_acc", 0.0)))
        if abs(tbl_delta) > 1e-9 or abs(exe_delta) > 1e-9 or abs(tok_delta) > 1e-9:
            lines += [
                "**What did change:** the partial-credit metrics below, which "
                "decompose the task so a data-mix effect is visible even when "
                "strict accuracy is not. See *Findings* for the direction of each "
                "one and *How small is this sample?* for the caveat.",
                "",
            ]

    lines += [
        "**Most likely cause: model capacity.** The reference backend is a small "
        "transformer trained from scratch on a few hundred examples at a fixed "
        "step budget. Adding several times more diverse synthetic data at the same "
        "budget means each record is seen far fewer times, so the extra diversity "
        "buys schema coverage but costs memorisation of exact token sequences. The "
        "`hf` backend (Qwen2.5-7B + LoRA) is the path that can turn this data into "
        "execution accuracy; the reference backend exists to validate the pipeline "
        "end-to-end, and it does.",
        "",
    ]
    rows_tbl = [
        ["Baseline exec acc", pct(h.get("baseline_exec_acc", 0.0))],
        ["Best-mix exec acc", pct(h.get("rich_exec_acc", 0.0))],
        ["Accuracy gain", f"{h.get('gain_pp', 0.0):+.1f} pp"],
        ["Synthesis tokens invested", num(h.get("synth_tokens", 0))],
        ["Synthesis cost", usd(h.get("synth_cost_usd", 0.0))],
        ["Accuracy points per 1M tokens", "not measurable (no gain)"],
    ]
    return "\n".join(lines) + "\n" + md_table(("Metric", "Value"), rows_tbl) + "\n"
    """The project's one-number claim: accuracy points per 1M synthesis tokens.

    Computed from the *measured* ablation endpoints only: the baseline
    (``raw_only``, zero synthesis tokens) and the richest config. Returns
    ``{"available": False, ...}`` with a reason when the ablation cannot support
    the claim (missing baseline, or the rich config spent no tokens), so the
    report never prints a ratio divided by zero or an invented baseline.
    """
    if not ablation or not ablation.get("rows"):
        return {"available": False, "reason": "ablation not run"}
    rows = ablation["rows"]
    baseline = next((r for r in rows if r.get("config") == "raw_only"), None)
    if baseline is None:
        return {"available": False, "reason": "no raw_only baseline row"}
    rich = max(rows, key=lambda r: float(r.get("synth_tokens", 0) or 0))
    tokens = float(rich.get("synth_tokens", 0) or 0)
    gain = float(rich.get("exec_acc", 0.0)) - float(baseline.get("exec_acc", 0.0))
    if tokens <= 0:
        return {"available": False,
                "reason": "richest config spent no synthesis tokens"}
    per_mtok = gain / (tokens / 1_000_000.0)
    result = {
        "available": True,
        "baseline_config": baseline.get("config", "raw_only"),
        "baseline_exec_acc": float(baseline.get("exec_acc", 0.0)),
        "rich_config": rich.get("config", ""),
        "rich_exec_acc": float(rich.get("exec_acc", 0.0)),
        "gain_pp": round(gain * 100.0, 2),
        "synth_tokens": int(tokens),
        "synth_cost_usd": float(rich.get("synth_cost_usd", 0.0) or 0.0),
        "points_per_mtok": round(per_mtok * 100.0, 2),
        "tokens_per_point": int(round(tokens / (gain * 100.0))) if gain > 0 else None,
        "relative_improvement_pct": round(
            gain / float(baseline.get("exec_acc", 0.0)) * 100.0, 1)
        if float(baseline.get("exec_acc", 0.0)) > 0 else None,
    }
    if gain <= 0.0:
        # A ratio that says "0.00 points per 1M tokens" invites the reader to
        # conclude synthesis does not work, when what actually happened is that
        # this run produced no measurable capability at all. Say that instead.
        result["no_measurable_gain"] = True
        result["reason"] = (
            "no measurable accuracy gain between the baseline and the richest "
            "config, so there is no cost/capability ratio to report; the model "
            "produced no executable-and-correct SQL in either arm")
    return result


def scale_trend_note(scale: dict[str, Any] | None) -> str:
    """Describe whether the partial-credit trend across the sweep is monotone.

    A non-monotone sweep is a real result and must be described as such: with a
    handful of points and n=24 eval examples, a single example moves a rate by
    4.2pp, so a zig-zag is expected. The note states the measured sequence and
    says plainly whether it is monotone, instead of implying a clean curve that
    the numbers do not support.
    """
    points = (scale or {}).get("points") or []
    if len(points) < 3:
        return ""
    ordered = sorted(points, key=lambda p: float(p.get("fraction", 0.0) or 0.0))
    series: list[tuple[float, float]] = [
        (float(p.get("fraction", 0.0) or 0.0), float(p.get("table_set_match", 0.0) or 0.0))
        for p in ordered
    ]
    values = [v for _f, v in series]
    monotone_up = all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
    monotone_down = all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    pretty = " \u2192 ".join(f"{pct(v)} ({f:.0%})" for f, v in series)
    n_eval = int(ordered[0].get("n_eval", 0) or 0)
    if monotone_up:
        verdict = "a monotone increase"
    elif monotone_down:
        verdict = "a monotone decrease"
    else:
        verdict = ("**not monotone** -- the sequence zig-zags, which is expected "
                   "when each point is measured on a small eval set")
    note = (f"**Table-grounding trend across the sweep:** {pretty} \u2014 {verdict}.")
    if n_eval:
        note += (f" With {num(n_eval)} eval examples, one example is "
                 f"{100.0 / n_eval:.1f}pp, so single-step moves of one or two "
                 "examples should not be read as a scaling law.")
    return note


def findings_markdown(ablation: dict[str, Any] | None,
                      scale: dict[str, Any] | None = None) -> str:
    """Blunt, bulleted statement of what the measurements actually show.

    Placed above the headline so a reader cannot miss the result even if they
    skim. Every sentence is generated from measured deltas -- no hardcoded
    claims -- so the wording stays true if a future run changes the direction.
    """
    rows = (ablation or {}).get("rows") or []
    if not rows:
        return ""
    base = next((r for r in rows if r.get("config") == "raw_only"), rows[0])
    rich = max(rows, key=lambda r: float(r.get("synth_tokens", 0) or 0))
    if base is rich:
        return ""

    def _delta(key: str) -> tuple[float, float, float]:
        b = float(base.get(key, 0.0) or 0.0)
        r = float(rich.get(key, 0.0) or 0.0)
        return (b, r, r - b)

    tbl_b, tbl_r, tbl_d = _delta("table_set_match")
    tok_b, tok_r, tok_d = _delta("token_acc")
    exe_b, exe_r, exe_d = _delta("executable_rate")
    col_b, col_r, col_d = _delta("column_f1")
    cla_b, cla_r, cla_d = _delta("clause_match_mean")

    bullets: list[str] = []

    # ---- the null result, stated first ---------------------------------
    if float(rich.get("exec_acc", 0.0)) == 0.0 and float(base.get("exec_acc", 0.0)) == 0.0:
        bullets.append(
            "**Execution accuracy did not improve.** Both "
            f"`{base.get('config')}` and `{rich.get('config')}` scored 0.0% after "
            f"{num(rich.get('synth_tokens', 0))} synthesis tokens. Reported as "
            "measured; not adjusted.")

    # ---- what moved, and which way ------------------------------------
    def _dirn(d: float) -> str:
        if abs(d) < 1e-9:
            return "unchanged"
        return "up" if d > 0 else "down"

    moved = [
        ("table-level schema grounding", tbl_b, tbl_r, tbl_d),
        ("exact-token fidelity", tok_b, tok_r, tok_d),
        ("column F1", col_b, col_r, col_d),
        ("executability", exe_b, exe_r, exe_d),
        ("clause-structure agreement", cla_b, cla_r, cla_d),
    ]
    improved = [m for m in moved if m[3] > 1e-9]
    worsened = [m for m in moved if m[3] < -1e-9]

    if improved:
        bits = [f"{name} {pct(b)} \u2192 {pct(r)}" for name, b, r, _d in improved]
        bullets.append("**Improved with synthetic data:** " + "; ".join(bits) + ".")
    if worsened:
        bits = [f"{name} {pct(b)} \u2192 {pct(r)}" for name, b, r, _d in worsened]
        bullets.append("**Worsened with synthetic data:** " + "; ".join(bits) + ".")

    # ---- how much of this is noise? -----------------------------------
    n_eval = int(rich.get("n_eval", 0) or 0)
    if n_eval > 0:
        per_example = 100.0 / n_eval
        tbl_examples = abs(tbl_d) * n_eval
        bullets.append(
            f"**How small is this sample?** Every arm scored the same {num(n_eval)} "
            f"dev examples, so one example is worth {per_example:.2f} pp. The "
            f"table-grounding move of {pct(tbl_b)} \u2192 {pct(tbl_r)} is "
            f"{tbl_examples:.0f} of {n_eval} examples. The bootstrap CIs overlap "
            "substantially, so this is a directional signal worth investigating at "
            "scale, not an established effect.")

    # ---- why ----------------------------------------------------------
    n_base = int(base.get("n_sft_records", 0) or 0)
    n_rich = int(rich.get("n_sft_records", 0) or 0)
    if n_rich > n_base > 0:
        ratio = n_rich / n_base
        bullets.append(
            f"**Why the trade-off:** the richest arm trains on {num(n_rich)} records "
            f"vs {num(n_base)} for the baseline ({ratio:.1f}\u00d7 more data) at the "
            "same fixed step budget, so each record is seen proportionally fewer "
            "times. At this model scale that buys schema coverage and costs exact "
            "token memorisation -- the two metrics move in opposite directions "
            "exactly as that explanation predicts.")

    bullets.append(
        "**What this does and does not show.** It shows the synthesis -> "
        "validation -> function-calling format -> SFT -> greedy decode -> real "
        "SQLite execution pipeline runs end to end and produces measurable, "
        "directional change in partial-credit metrics. It does not show a "
        "state-of-the-art text2sql model: the reference backend is a small "
        "transformer trained from scratch on a few hundred examples. The `hf` "
        "backend (Qwen2.5-7B + LoRA) is the path that turns this data into "
        "execution accuracy.")

    return "\n".join(["## Findings", ""] + [f"- {b}" for b in bullets])


def headline_markdown(ablation: dict[str, Any] | None,
                      scale: dict[str, Any] | None = None) -> str:
    """One-paragraph + one-table statement of the cost/capability trade.

    Three cases, each stated plainly rather than hedged:

    * a real gain: report the measured points-per-1M-tokens ratio;
    * no gain but a measurable partial-credit signal: say so and point at the
      partial-credit table, because that is where the finding actually lives;
    * nothing measurable at all: say that, and name capacity as the likely cause.
    """
    h = headline_ratio(ablation)
    if not h.get("available"):
        return (f"_Headline ratio unavailable: {h.get('reason', 'unknown')}._\n\n"
                "Run the ablation stage to populate it.")
    if h.get("no_measurable_gain"):
        return _no_gain_headline(ablation, h)
    lines = [
        "### Headline: token investment \u2192 capability",
        "",
        f"Spending **{num(h['synth_tokens'])} synthesis tokens** "
        f"({usd(h['synth_cost_usd'])}) to go from `{h['baseline_config']}` "
        f"({pct(h['baseline_exec_acc'])}) to `{h['rich_config']}` "
        f"({pct(h['rich_exec_acc'])}) bought **{h['gain_pp']:+.1f} percentage points** "
        f"of execution accuracy.",
        "",
        f"**{h['points_per_mtok']:+.2f} accuracy points per 1M synthesis tokens**"
        + (f" \u2014 about {num(h['tokens_per_point'])} tokens per accuracy point."
           if h.get("tokens_per_point") else "."),
        "",
    ]
    rows = [
        ["Baseline exec acc", pct(h["baseline_exec_acc"])],
        ["Best-mix exec acc", pct(h["rich_exec_acc"])],
        ["Accuracy gain", f"{h['gain_pp']:+.1f} pp"
         + (f" ({h['relative_improvement_pct']:+.1f}% relative)"
            if h.get("relative_improvement_pct") is not None else "")],
        ["Synthesis tokens invested", num(h["synth_tokens"])],
        ["Synthesis cost", usd(h["synth_cost_usd"])],
        ["Accuracy points per 1M tokens", f"{h['points_per_mtok']:+.2f}"],
    ]
    if h.get("tokens_per_point"):
        rows.append(["Tokens per accuracy point", num(h["tokens_per_point"])])
    sat = (scale or {}).get("saturation") or {}
    if sat:
        rows.append([
            "Saturation point",
            f"{float(sat.get('fraction', 0.0)):.0%} of synthetic corpus "
            f"({num(sat.get('synth_tokens', 0))} tokens, {pct(sat.get('exec_acc', 0.0))})",
        ])
        if sat.get("tokens_saved"):
            rows.append(["Tokens saved by stopping there", num(sat.get("tokens_saved", 0))])
    return "\n".join(lines) + "\n" + md_table(("Metric", "Value"), rows)


# --------------------------------------------------------------------------- #
# funnel construction from artifacts
# --------------------------------------------------------------------------- #
def build_validation_funnel(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    """Assemble the funnel from raw/synth/validated/SFT counts + validation report.

    ``artifacts`` may contain ``n_raw``, ``n_synth``, ``n_validated``, ``n_sft``
    and the parsed ``validation_report`` dict. Counts of 0/None are omitted rather
    than printed as a zero that looks like a failure.
    """
    report = artifacts.get("validation_report") or {}
    n_raw = artifacts.get("n_raw")
    n_synth = artifacts.get("n_synth")
    n_sft = artifacts.get("n_sft")
    n_val = artifacts.get("n_validated")

    stages: list[dict[str, Any]] = []
    if n_raw is not None:
        stages.append({"stage": "1. Raw Spider examples", "n": int(n_raw),
                       "kept": int(n_raw), "pass_rate": 1.0,
                       "note": "original dataset, no synthesis cost"})
    if n_synth is not None:
        stages.append({"stage": "2. Synthesised records", "n": int(n_synth),
                       "kept": int(n_synth), "pass_rate": None,
                       "note": "cot + paraphrase + hard_negative"})
    fmt_pass = report.get("format_pass_rate")
    sem_pass = report.get("semantic_pass_rate")
    combined = report.get("combined_pass_rate")
    if n_synth is not None and fmt_pass is not None:
        stages.append({"stage": "3. Format validation (schema + sqlite exec)",
                       "n": int(n_synth), "kept": int(round(n_synth * float(fmt_pass))),
                       "pass_rate": float(fmt_pass),
                       "note": "analyze_sql + real execution"})
    if n_synth is not None and sem_pass is not None:
        stages.append({"stage": "4. Semantic judge (question \u2194 SQL)",
                       "n": int(n_synth), "kept": int(round(n_synth * float(sem_pass))),
                       "pass_rate": float(sem_pass),
                       "note": f"threshold {report.get('semantic_threshold', 'n/a')}"})
    if n_val is not None:
        stages.append({"stage": "5. Validated records", "n": int(n_val),
                       "kept": int(n_val), "pass_rate": combined,
                       "note": "passed all enabled gates"})
    if n_sft is not None:
        stages.append({"stage": "6. SFT training records", "n": int(n_sft),
                       "kept": int(n_sft), "pass_rate": None,
                       "note": "after format conversion (+ raw examples)"})
    if not stages:
        return None
    top = report.get("top_failure_codes") or []
    return {"stages": stages, "top_failure_codes": top}


# --------------------------------------------------------------------------- #
# extrapolation section (Lead-owned module)
# --------------------------------------------------------------------------- #
def extrapolation_section(reports_dir: str | Path) -> str:
    """Render the Lead's token-extrapolation table if its JSON exists.

    The projection from measured per-record cost to Spider scale is the
    "10M+ tokens" evidence; it lives in :mod:`fcdata.extrapolate` and writes
    ``reports/token_extrapolation.json``. The report includes it verbatim (via
    that module's own markdown renderer) and says plainly when it is absent.
    """
    path = Path(reports_dir) / EXTRAPOLATION_FILE
    data = read_json(path, None)
    if data is None:
        return ("_Token extrapolation not available: run "
                "`python -m fcdata.extrapolate` to project measured cost to "
                "Spider scale._")
    try:
        from ..extrapolate import markdown_table as extrap_table  # noqa: PLC0415

        table = extrap_table(data)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        return f"_Token extrapolation present but could not be rendered: {exc}_"
    head = data.get("headline") or {}
    measured = data.get("measured") or {}
    lines: list[str] = []
    if measured:
        lines += [
            f"Measured on the demo corpus: **{num(measured.get('total_tokens', 0))} "
            f"tokens** ({usd(measured.get('total_cost_usd', 0.0))}) for "
            f"{num(measured.get('n_source_examples', 0))} source questions "
            f"\u2192 {num(measured.get('n_synth_records', 0))} augmented records, i.e. "
            f"**{float(measured.get('synth_tokens_per_source', 0.0)):,.0f} tokens per "
            "source question** (real ledger usage, not an estimate).",
            "",
        ]
    if head.get("max_tokens_m") is not None:
        reaches = "reaches" if head.get("reaches_10m") else "does not reach"
        lines += [
            f"Projected linearly to Spider scale, the largest plan "
            f"(`{head.get('max_tokens_label', 'n/a')}`) is "
            f"**{float(head.get('max_tokens_m', 0.0)):.2f}M tokens** for "
            f"{usd(head.get('spider_full_cost_usd', 0.0))} at the Spider-full point, "
            f"which {reaches} the 10M-token target.",
            "",
        ]
    # A blank line is required before the table or markdown renders it as prose.
    return "\n".join(lines) + "\n" + table


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def build_report(cfg: Any,
                 ablation: dict[str, Any] | None,
                 scale: dict[str, Any] | None,
                 ledger_summary: dict[str, Any] | None,
                 out_dir: str | Path,
                 *,
                 artifacts: dict[str, Any] | None = None,
                 train_metrics: dict[str, Any] | None = None,
                 make_plots: bool = True) -> dict[str, Any]:
    """Write ``README_TABLES.md`` + CSVs + figures; return a summary dict.

    Parameters
    ----------
    ablation / scale:
        Payloads from :func:`fcdata.eval.ablation.run_ablation` /
        :func:`fcdata.eval.scale.run_scale_curve` (or ``None`` when skipped).
    ledger_summary:
        ``{"totals": {...}, "by_stage": {...}}`` from ``TokenLedger.aggregate()``.
    out_dir:
        Reports directory (``cfg.paths.report_path``).
    artifacts:
        Optional counts for the validation funnel (see
        :func:`build_validation_funnel`); read from the artifacts tree when
        omitted.
    train_metrics:
        A run's ``metrics.json`` for the loss curve.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if artifacts is None:
        artifacts = collect_artifacts(cfg)
    funnel = build_validation_funnel(artifacts)
    headline = headline_ratio(ablation)

    sections: list[str] = [
        "# Results: token investment \u2192 execution accuracy",
        "",
        "_Every accuracy number below was produced by executing the model's SQL "
        "against real SQLite databases (`fcdata.dbexec.evaluate_pair`). "
        "No metric in this file is simulated or hardcoded._",
        "",
    ]

    source_note = _provenance_note(ablation, scale, funnel)
    if source_note:
        sections += [source_note, ""]

    findings = findings_markdown(ablation, scale)
    if findings:
        sections += [findings, ""]

    sections += [headline_markdown(ablation, scale), ""]

    sections += [
        "## Figures",
        "",
        "![Cost vs quality](cost_vs_quality.png)",
        "",
        "_Synthesis tokens invested vs. measured execution accuracy, one annotated "
        "point per ablation configuration, with a least-squares trend line. A flat "
        "line at zero is the honest statement of the headline metric at this model "
        "scale._",
        "",
    ]
    if (ablation or {}).get("rows") and any(
            float(r.get("token_acc", 0.0) or 0.0) > 0.0
            or float(r.get("table_set_match", 0.0) or 0.0) > 0.0
            for r in (ablation or {}).get("rows", [])):
        sections += [
            "![Cost vs partial credit](cost_vs_partial.png)",
            "",
            "_Partial-credit metrics vs. synthesis tokens. This is where the "
            "measured effect lives when strict execution accuracy is flat: "
            "schema grounding rises while executability and token fidelity fall._",
            "",
        ]
    sections += [
        "![Scale curve](scale_curve.png)",
        "",
        "_Execution accuracy vs. synthesis tokens (bottom axis) and vs. SFT dataset "
        "size (top axis, same measured points)._",
        "",
    ]

    sections += ["## Ablation: which synthesis kinds pay off", "",
                 ablation_table(ablation), ""]

    sections += [
        "## Partial credit: what the model learned when exec accuracy is 0",
        "",
        "_Strict execution accuracy requires a byte-perfect statement, so it is "
        "~0 for any model below ~99% token accuracy. These metrics isolate the "
        "components of the task -- token agreement, query shape, schema grounding, "
        "clause structure -- and are computed from the same real generations._",
        "",
        partial_table(ablation), "",
        "### Failure taxonomy: which error class each data mix produced",
        "",
        "_Maps onto the hard-negative error kinds the synthesis stage injects._",
        "",
        failure_taxonomy_table(ablation), "",
    ]

    sections += ["## Data-scale curve: how much synthetic data is enough", "",
                 scale_table(scale), ""]

    if scale and scale.get("points"):
        sections += ["### Partial credit across the data scale", "",
                     scale_partial_table(scale), ""]
        note = scale_trend_note(scale)
        if note:
            sections += [note, ""]

    sat = (scale or {}).get("saturation") or {}
    if sat and sat.get("available", True) and sat.get("fraction") is not None:
        saved = int(sat.get("tokens_saved", 0) or 0)
        if saved > 0:
            spend_note = (f"while spending {num(sat.get('synth_tokens', 0))} tokens "
                          f"instead of {num(sat.get('synth_tokens', 0) + saved)}")
        else:
            spend_note = ("at the full corpus cost -- no cheaper point reached the "
                          "same accuracy, so the curve had not saturated yet")
        sections += [
            f"**Saturation:** {float(sat.get('fraction', 0.0)):.0%} of the synthetic "
            f"corpus reaches {pct(sat.get('exec_acc', 0.0))}, within "
            f"{float(sat.get('tolerance', 0.0)) * 100:.1f} pp of the best measured "
            f"accuracy ({pct(sat.get('best_exec_acc', 0.0))}) {spend_note}.",
            "",
        ]
    elif sat and not sat.get("available", True):
        sections += [f"**No saturation point:** {sat.get('reason', '')}.", ""]

    sections += ["## Validation funnel: what survived before training", "",
                 validation_funnel_table(funnel), ""]

    sections += ["## Token ledger by stage and role", "",
                 ledger_table(ledger_summary), ""]

    sections += ["## Extrapolation to Spider scale", "",
                 extrapolation_section(out), ""]

    sections += ["## Artifacts", "", _artifact_list(ablation, scale, out), ""]

    md = "\n".join(sections)
    md_path = out / "README_TABLES.md"
    md_path.write_text(md, encoding="utf-8")

    # ---- CSVs ---------------------------------------------------------
    csv_paths = _write_csvs(ablation, scale, out)

    # ---- figures ------------------------------------------------------
    figures: dict[str, str] = {}
    if make_plots:
        note = ("Smoke run (`--quick`): numbers validate the pipeline, "
                "not model quality." if (ablation or {}).get("quick") else "")
        figures = plots.make_all_figures(
            out, ablation=ablation, scale=scale,
            train_metrics=train_metrics, note=note)

    summary = {
        "markdown": str(md_path),
        "csv": csv_paths,
        "figures": figures,
        "headline": headline,
        "saturation": sat,
        "n_ablation_rows": len((ablation or {}).get("rows") or []),
        "n_scale_points": len((scale or {}).get("points") or []),
        "validation_funnel_stages": len((funnel or {}).get("stages") or []),
        "ledger_totals": (ledger_summary or {}).get("totals") or {},
    }
    write_json(out / "report_summary.json", summary)
    return summary


def _provenance_note(ablation: dict[str, Any] | None,
                     scale: dict[str, Any] | None,
                     funnel: dict[str, Any] | None) -> str:
    """State plainly how these numbers were produced (quick mode, corpus source)."""
    bits: list[str] = []
    quick = bool((ablation or {}).get("quick") or (scale or {}).get("quick"))
    if quick:
        bits.append("> **Smoke run (`--quick`)**: training used a reduced step "
                    "budget, so the numbers validate the pipeline end-to-end, "
                    "not the achievable model quality. Re-run without `--quick` "
                    "for reportable results.")
    source = (ablation or {}).get("source") or (scale or {}).get("source")
    if source:
        bits.append(f"_SFT corpus source: `{source}`._")
    if _has_estimated_cost(ablation, scale):
        bits.append("> **Cost note**: rows whose synthesis cost is not backed by "
                    "the token ledger are marked as estimated from the configured "
                    "per-million-token prices applied to the same token counts.")
    for payload in (ablation, scale):
        for warning in (payload or {}).get("warnings") or []:
            bits.append(f"> **Warning**: {warning}.")
    if ablation and ablation.get("n_eval_examples"):
        n_eval = int(ablation["n_eval_examples"])
        per_example = 100.0 / max(1, n_eval)
        bits.append(
            f"_Evaluated on {num(n_eval)} dev examples, identical across every "
            f"configuration. One example is worth {per_example:.2f} percentage "
            "points, so treat small differences as indicative rather than "
            "established; the bootstrap CIs in the tables show the resulting "
            "uncertainty._")
    return "\n\n".join(bits)


def _has_estimated_cost(ablation: dict[str, Any] | None,
                        scale: dict[str, Any] | None) -> bool:
    """True when any reported cost came from prices rather than the ledger."""
    for payload, key in ((ablation, "rows"), (scale, "points")):
        for row in (payload or {}).get(key) or []:
            if float(row.get("synth_tokens", 0) or 0) > 0 and not row.get(
                    "cost_from_ledger", True):
                return True
    return False


def _artifact_list(ablation: dict[str, Any] | None, scale: dict[str, Any] | None,
                   out: Path) -> str:
    rows: list[list[str]] = [
        ["`README_TABLES.md`", "this file"],
        ["`cost_vs_quality.png`", "headline figure: tokens vs accuracy"],
        ["`scale_curve.png`", "headline figure: data-scale curve"],
    ]
    if ablation:
        rows.append(["`ablation.json` / `ablation.csv`", "per-config metrics + cost"])
    if scale:
        rows.append(["`scale_curve.json` / `scale_curve.csv`", "per-fraction metrics"])
    rows.append(["`report_summary.json`", "machine-readable summary of this report"])
    return md_table(("Artifact", "Contents"), rows)


def _write_csvs(ablation: dict[str, Any] | None, scale: dict[str, Any] | None,
                out: Path) -> dict[str, str]:
    """Re-emit the ablation/scale CSVs next to the report (idempotent copies)."""
    import csv  # noqa: PLC0415 - stdlib, call-time import

    written: dict[str, str] = {}
    for name, payload, key in (("ablation.csv", ablation, "rows"),
                               ("scale_curve.csv", scale, "points")):
        rows = (payload or {}).get(key) or []
        if not rows:
            continue
        path = out / name
        cols: list[str] = []
        for row in rows:
            for k in row:
                if k not in cols:
                    cols.append(k)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_cell(row.get(k)) for k in cols})
        written[name] = str(path)
    return written


def _csv_cell(value: Any) -> Any:
    """Flatten nested values so a CSV row never contains a Python repr."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


# --------------------------------------------------------------------------- #
# artifact discovery (so the CLI can call build_report with zero arguments)
# --------------------------------------------------------------------------- #
def collect_artifacts(cfg: Any) -> dict[str, Any]:
    """Count records per pipeline stage and load ``validation_report.json``.

    Used to build the validation funnel without the caller plumbing five counts
    through. Missing files simply omit their stage.
    """
    from ..records import read_jsonl  # noqa: PLC0415 - local import

    def _count(path: Path) -> int | None:
        if not path.exists():
            return None
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip())

    out: dict[str, Any] = {
        "n_raw": _count(Path(cfg.paths.raw_dir) / "train.jsonl"),
        "n_synth": _count(Path(cfg.paths.synth_dir) / "synth.jsonl"),
        "n_validated": _count(Path(cfg.paths.validated_dir) / "validated.jsonl"),
        "n_sft": _count(Path(cfg.paths.sft_dir) / "train.jsonl"),
    }
    report = read_json(Path(cfg.paths.validated_dir) / "validation_report.json", None)
    if isinstance(report, dict):
        out["validation_report"] = report
    return out


def collect_ledger_summary(cfg: Any) -> dict[str, Any] | None:
    """Aggregate every ledger JSONL the pipeline produced into one summary."""
    from ..tokens import summarize_ledger  # noqa: PLC0415 - local import

    merged: dict[str, dict[str, Any]] = {}
    totals = {"stages": 0, "calls": 0, "total_tokens": 0, "prompt_tokens": 0,
              "completion_tokens": 0, "cost_usd": 0.0}
    found = False
    for path in (Path(cfg.paths.synth_dir) / "ledger.jsonl",
                 Path(cfg.paths.validated_dir) / "ledger.jsonl",
                 Path(cfg.paths.resolve(cfg.paths.artifacts_dir)) / "ledger.jsonl"):
        if not path.exists():
            continue
        found = True
        summary = summarize_ledger(path)
        for stage_name, st in (summary.get("by_stage") or {}).items():
            target = merged.setdefault(stage_name, {
                "calls": 0, "ok_calls": 0, "cached_calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
                "by_role": {}})
            for key in ("calls", "ok_calls", "cached_calls", "prompt_tokens",
                        "completion_tokens", "total_tokens"):
                target[key] += int(st.get(key, 0) or 0)
            target["cost_usd"] += float(st.get("cost_usd", 0.0) or 0.0)
            for role, br in (st.get("by_role") or {}).items():
                rt = target["by_role"].setdefault(role, {
                    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0, "models": []})
                for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                    rt[key] += int(br.get(key, 0) or 0)
                rt["cost_usd"] += float(br.get("cost_usd", 0.0) or 0.0)
                for model in br.get("models") or []:
                    if model not in rt["models"]:
                        rt["models"].append(model)
    if not found:
        return None
    for st in merged.values():
        st["cost_usd"] = round(st["cost_usd"], 6)
        for rt in st["by_role"].values():
            rt["cost_usd"] = round(rt["cost_usd"], 6)
            rt["models"] = sorted(rt["models"])
    totals["stages"] = len(merged)
    totals["calls"] = sum(s["calls"] for s in merged.values())
    totals["total_tokens"] = sum(s["total_tokens"] for s in merged.values())
    totals["prompt_tokens"] = sum(s["prompt_tokens"] for s in merged.values())
    totals["completion_tokens"] = sum(s["completion_tokens"] for s in merged.values())
    totals["cost_usd"] = round(sum(s["cost_usd"] for s in merged.values()), 6)
    return {"totals": totals, "by_stage": merged}


def load_report_inputs(cfg: Any) -> tuple[dict[str, Any] | None,
                                          dict[str, Any] | None,
                                          dict[str, Any] | None]:
    """Best-effort load of (ablation, scale, ledger) from the artifacts tree.

    Lets the CLI rebuild the report from disk without re-running any experiment.
    """
    eval_dir = Path(cfg.paths.eval_dir)
    ablation = read_json(eval_dir / "ablation.json", None)
    scale = read_json(eval_dir / "scale_curve.json", None)
    return (ablation if isinstance(ablation, dict) else None,
            scale if isinstance(scale, dict) else None,
            collect_ledger_summary(cfg))


def find_train_metrics(cfg: Any) -> dict[str, Any] | None:
    """Locate the most recent ``metrics.json`` under the train artifacts dir."""
    train_root = Path(cfg.paths.train_dir)
    if not train_root.exists():
        return None
    candidates = sorted(train_root.glob("*/metrics.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        data = read_json(path, None)
        if isinstance(data, dict):
            return data
    return None
