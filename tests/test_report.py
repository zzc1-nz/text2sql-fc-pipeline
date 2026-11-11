"""Tests for the report layer: markdown tables, headline ratio, figures, CSVs.

WHAT IS ASSERTED
----------------
The report is the artifact a reviewer actually reads, so the tests focus on the
properties that make it trustworthy rather than on exact byte output:

* every required table section exists and is a *valid* GitHub markdown table
  (header, separator, uniform column count);
* the headline ratio is arithmetic on measured numbers, and it refuses to invent
  a ratio when the ablation cannot support one (no baseline / no tokens);
* the "smoke run" and "estimated cost" caveats are present when they apply, so a
  toy number can never be presented as a result;
* figures are written as real, non-trivial PNGs at the configured dpi with the
  Agg backend, and the scale figure's secondary axis is a faithful mapping;
* CSVs round-trip through ``csv.DictReader`` with the documented columns.

No figure content is asserted pixel-by-pixel (that would be brittle); instead we
assert the PNG signature, dimensions and that both axes carry data.
"""

from __future__ import annotations

import csv
import shutil
import unittest
from pathlib import Path
from typing import Any

import matplotlib

from fcdata.config import Config, scratch_dir
from fcdata.report import (
    build_report,
    failure_taxonomy_table,
    findings_markdown,
    scale_trend_note,
    partial_table,
    scale_partial_table,
    headline_markdown,
    build_validation_funnel,
    headline_ratio,
    ledger_table,
    md_table,
    num,
    pct,
    pp,
    usd,
    ablation_table,
    scale_table,
    validation_funnel_table,
)
from fcdata.report import plots
from fcdata.records import read_json


# --------------------------------------------------------------------------- #
# fixtures: payloads shaped exactly like ablation.json / scale_curve.json
# --------------------------------------------------------------------------- #
def _ablation_payload(quick: bool = False) -> dict[str, Any]:
    return {
        "rows": [
            {"config": "raw_only", "n_sft_records": 98, "n_unique_records": 98,
             "synth_tokens": 0, "synth_cost_usd": 0.0, "cost_from_ledger": True,
             "exec_acc": 0.25, "exec_acc_ci_lo": 0.15, "exec_acc_ci_hi": 0.35,
             "exact_match": 0.10, "executable_rate": 0.9, "n_eval": 24,
             "delta_exec_acc": 0.0, "train_seconds": 12.0,
             "kind_counts": {"raw": 98}},
            {"config": "plus_cot", "n_sft_records": 196, "n_unique_records": 196,
             "synth_tokens": 45_000, "synth_cost_usd": 0.0034, "cost_from_ledger": True,
             "exec_acc": 0.4167, "exec_acc_ci_lo": 0.30, "exec_acc_ci_hi": 0.54,
             "exact_match": 0.2083, "executable_rate": 0.958, "n_eval": 24,
             "delta_exec_acc": 0.1667, "train_seconds": 15.0,
             "kind_counts": {"raw": 98, "cot": 98}},
            {"config": "plus_hard_negatives", "n_sft_records": 520,
             "n_unique_records": 310, "synth_tokens": 120_000,
             "synth_cost_usd": 0.0091, "cost_from_ledger": True,
             "exec_acc": 0.5833, "exec_acc_ci_lo": 0.45, "exec_acc_ci_hi": 0.70,
             "exact_match": 0.3125, "executable_rate": 1.0, "n_eval": 24,
             "delta_exec_acc": 0.3333, "train_seconds": 22.0,
             "kind_counts": {"raw": 98, "cot": 98, "paraphrase": 80,
                             "hard_negative": 144}},
        ],
        "baseline": "raw_only",
        "best": {"config": "plus_hard_negatives", "exec_acc": 0.5833,
                 "synth_tokens": 120_000, "synth_cost_usd": 0.0091},
        "n_configs": 3,
        "source": "artifacts/04_sft/train.jsonl",
        "quick": quick,
        "eval_split": "dev",
        "n_eval_examples": 24,
        "sft_kind_counts": {"raw": 98, "cot": 98, "paraphrase": 80,
                            "hard_negative": 144},
        "n_sft_available": 420,
    }


def _scale_payload(quick: bool = False) -> dict[str, Any]:
    return {
        "points": [
            {"fraction": 0.0, "n_sft_records": 98, "n_synth_records": 0,
             "synth_tokens": 0, "synth_cost_usd": 0.0, "cost_from_ledger": True,
             "exec_acc": 0.25, "exec_acc_ci_lo": 0.15, "exec_acc_ci_hi": 0.35,
             "exact_match": 0.10, "executable_rate": 0.9, "n_eval": 24,
             "delta_exec_acc": 0.0, "train_seconds": 12.0},
            {"fraction": 0.5, "n_sft_records": 260, "n_synth_records": 162,
             "synth_tokens": 60_000, "synth_cost_usd": 0.0046,
             "cost_from_ledger": True, "exec_acc": 0.4583,
             "exec_acc_ci_lo": 0.36, "exec_acc_ci_hi": 0.56, "exact_match": 0.2083,
             "executable_rate": 0.958, "n_eval": 24, "delta_exec_acc": 0.2083,
             "train_seconds": 17.0},
            {"fraction": 1.0, "n_sft_records": 420, "n_synth_records": 322,
             "synth_tokens": 120_000, "synth_cost_usd": 0.0091,
             "cost_from_ledger": True, "exec_acc": 0.5208,
             "exec_acc_ci_lo": 0.42, "exec_acc_ci_hi": 0.62, "exact_match": 0.25,
             "executable_rate": 1.0, "n_eval": 24, "delta_exec_acc": 0.2708,
             "train_seconds": 21.0},
        ],
        "fractions": [0.0, 0.5, 1.0],
        "source": "artifacts/04_sft/train.jsonl",
        "quick": quick,
        "eval_split": "dev",
        "n_eval_examples": 24,
        "n_sft_available": 420,
        "n_synth_available": 322,
        "saturation": {"fraction": 0.5, "n_sft_records": 260,
                       "synth_tokens": 60_000, "synth_cost_usd": 0.0046,
                       "exec_acc": 0.4583, "best_fraction": 1.0,
                       "best_exec_acc": 0.5208, "accuracy_gap": 0.0625,
                       "tokens_saved": 60_000, "tolerance": 0.01},
        "baseline": {"fraction": 0.0, "exec_acc": 0.25, "synth_tokens": 0},
    }


def _ledger_payload() -> dict[str, Any]:
    return {
        "totals": {"stages": 2, "calls": 48, "total_tokens": 120_000,
                   "prompt_tokens": 90_000, "completion_tokens": 30_000,
                   "cost_usd": 0.0091},
        "by_stage": {
            "synth": {"calls": 40, "ok_calls": 40, "cached_calls": 2,
                      "prompt_tokens": 78_000, "completion_tokens": 26_000,
                      "total_tokens": 104_000, "cost_usd": 0.0078,
                      "by_role": {"cheap": {"calls": 40, "prompt_tokens": 78_000,
                                            "completion_tokens": 26_000,
                                            "total_tokens": 104_000,
                                            "cost_usd": 0.0078,
                                            "models": ["mock-cheap-1"]}}},
            "validate": {"calls": 8, "ok_calls": 8, "cached_calls": 0,
                         "prompt_tokens": 12_000, "completion_tokens": 4_000,
                         "total_tokens": 16_000, "cost_usd": 0.0013,
                         "by_role": {"judge": {"calls": 8, "prompt_tokens": 12_000,
                                               "completion_tokens": 4_000,
                                               "total_tokens": 16_000,
                                               "cost_usd": 0.0013,
                                               "models": ["mock-judge-1"]}}},
        },
    }


def _artifacts_payload() -> dict[str, Any]:
    return {
        "n_raw": 98,
        "n_synth": 322,
        "n_validated": 260,
        "n_sft": 420,
        "validation_report": {
            "format_pass_rate": 0.9,
            "semantic_pass_rate": 0.85,
            "combined_pass_rate": 0.78,
            "semantic_threshold": 0.5,
            "top_failure_codes": [{"code": "unknown_column", "n": 7},
                                  {"code": "exec_error", "n": 4}],
        },
    }


def _cfg(tmp: Path) -> Config:
    cfg = Config()
    cfg.paths.root = str(tmp)
    cfg.paths.reports_dir = str(tmp / "reports")
    cfg.paths.artifacts_dir = str(tmp / "artifacts")
    return cfg


# --------------------------------------------------------------------------- #
# formatting primitives
# --------------------------------------------------------------------------- #
class TestFormatting(unittest.TestCase):
    """One formatting rule per concept, used by every table."""

    def test_percent_has_one_decimal(self) -> None:
        self.assertEqual(pct(0.4167), "41.7%")
        self.assertEqual(pct(1.0), "100.0%")
        self.assertEqual(pct(0.0), "0.0%")

    def test_percentage_points_are_signed(self) -> None:
        self.assertEqual(pp(0.1667), "+16.7 pp")
        self.assertEqual(pp(-0.05), "-5.0 pp")

    def test_tokens_use_thousands_separators(self) -> None:
        self.assertEqual(num(120_000), "120,000")
        self.assertEqual(num(0), "0")
        self.assertEqual(num(None), "n/a")

    def test_usd_is_precise_enough_for_cent_fractions(self) -> None:
        self.assertEqual(usd(0.0091), "$0.0091")
        self.assertEqual(usd(1.5, 2), "$1.50")

    def test_bad_values_do_not_crash(self) -> None:
        for fn in (pct, pp, num, usd):
            self.assertEqual(fn("not-a-number"), "n/a")

    def test_md_table_is_well_formed(self) -> None:
        table = md_table(("A", "B"), [["x", "1"], ["y", "2"]])
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("| A | B |"))
        self.assertIn("---", lines[1])
        self.assertEqual(lines[2], "| x | 1 |")

    def test_md_table_escapes_pipes(self) -> None:
        table = md_table(("A",), [["a|b"]])
        self.assertIn("a\\|b", table)


# --------------------------------------------------------------------------- #
# table builders
# --------------------------------------------------------------------------- #
def _assert_markdown_table(test: unittest.TestCase, text: str, min_rows: int = 2) -> None:
    """Assert ``text`` contains a GitHub table with uniform column counts.

    Adjacent tables (separated only by prose) are checked as separate blocks, so
    a legitimate second table is not reported as a ragged first one.
    """
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    test.assertGreaterEqual(len(lines), min_rows, f"not a table:\n{text}")
    block: list[str] = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if block:
            widths = {b.count("|") for b in block}
            test.assertEqual(len(widths), 1, f"ragged table:\n" + "\n".join(block))
            block = []


class TestTables(unittest.TestCase):
    """Every required table must render, and must render *valid* markdown."""

    def test_ablation_table(self) -> None:
        text = ablation_table(_ablation_payload())
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("raw_only", text)
        self.assertIn("plus_hard_negatives", text)
        self.assertIn("120,000", text)     # thousands separator
        self.assertIn("58.3%", text)       # acc as percentage, 1 decimal
        self.assertIn("baseline", text)
        self.assertIn("+16.7 pp", text)

    def test_ablation_table_handles_missing_data(self) -> None:
        text = ablation_table(None)
        self.assertIn("not run", text)
        _assert_markdown_table(self, ablation_table(_ablation_payload()))

    def test_scale_table(self) -> None:
        text = scale_table(_scale_payload())
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("0%", text)
        self.assertIn("100%", text)
        self.assertIn("60,000", text)
        self.assertIn("45.8%", text)

    def test_scale_table_handles_missing_data(self) -> None:
        self.assertIn("not run", scale_table(None))

    def test_validation_funnel_table(self) -> None:
        funnel = build_validation_funnel(_artifacts_payload())
        self.assertIsNotNone(funnel)
        text = validation_funnel_table(funnel)
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("Raw Spider", text)
        self.assertIn("SFT training records", text)
        # the failure-code sub-table must also be present
        self.assertIn("unknown_column", text)
        self.assertIn("90.0%", text)   # format pass rate

    def test_validation_funnel_missing(self) -> None:
        self.assertIn("unavailable", validation_funnel_table(None))

    def test_ledger_table_splits_stage_and_role(self) -> None:
        text = ledger_table(_ledger_payload())
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("`synth`", text)
        self.assertIn("cheap", text)          # role row nested under stage
        self.assertIn("mock-cheap-1", text)   # model name surfaced
        self.assertIn("**total**", text)
        self.assertIn("120,000", text)

    def test_ledger_table_missing(self) -> None:
        self.assertIn("unavailable", ledger_table(None))
        self.assertIn("empty", ledger_table({"totals": {}, "by_stage": {}}))


# --------------------------------------------------------------------------- #
# partial-credit tables (the learning signal at 0% exec accuracy)
# --------------------------------------------------------------------------- #
class TestPartialTables(unittest.TestCase):
    """Partial-credit tables must render and must not hide a zero exec accuracy."""

    def _payload(self) -> dict[str, Any]:
        payload = _ablation_payload()
        for row, (tok, skel, tables, f1, clause) in zip(
                payload["rows"],
                [(0.18, 0.0, 0.10, 0.04, 0.62),
                 (0.31, 0.04, 0.29, 0.17, 0.71),
                 (0.52, 0.125, 0.54, 0.38, 0.83)]):
            row.update(token_acc=tok, skeleton_match=skel,
                       table_set_match=tables, column_f1=f1,
                       clause_match_mean=clause)
        payload["rows"][0]["partial_failures"] = {"not_executable": 18,
                                                 "wrong_table": 4}
        payload["rows"][2]["partial_failures"] = {"not_executable": 9,
                                                 "wrong_values": 3}
        return payload

    def test_partial_table_renders(self) -> None:
        text = partial_table(self._payload())
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("Token acc", text)
        self.assertIn("52.0%", text)   # plus_hard_negatives token acc
        self.assertIn("Table set", text)

    def test_partial_table_missing(self) -> None:
        self.assertIn("unavailable", partial_table(None))

    def test_failure_taxonomy_table_renders(self) -> None:
        text = failure_taxonomy_table(self._payload())
        _assert_markdown_table(self, text, min_rows=4)
        self.assertIn("not_executable", text)
        self.assertIn("wrong_values", text)
        self.assertIn("75.0%", text)   # 18/24 share

    def test_failure_taxonomy_empty(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["partial_failures"] = {}
        self.assertIn("No failures recorded", failure_taxonomy_table(payload))
        self.assertIn("unavailable", failure_taxonomy_table(None))

    def test_scale_partial_table_renders(self) -> None:
        payload = _scale_payload()
        for point, tok in zip(payload["points"], (0.18, 0.33, 0.52)):
            point.update(token_acc=tok, skeleton_match=0.05,
                         table_set_match=0.3, column_f1=0.2)
        text = scale_partial_table(payload)
        _assert_markdown_table(self, text, min_rows=5)
        self.assertIn("52.0%", text)
        self.assertIn("unavailable", scale_partial_table(None))

    def test_report_contains_partial_sections(self) -> None:
        d = scratch_dir("report_partial")
        try:
            summary = build_report(None, self._payload(), _scale_payload(),
                                   _ledger_payload(), d,
                                   artifacts=_artifacts_payload())
            md = Path(summary["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Partial credit", md)
            self.assertIn("Failure taxonomy", md)
            self.assertIn("Partial credit across the data scale", md)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_partial_metrics_appear_even_at_zero_exec_acc(self) -> None:
        """The whole point: a 0.0% exec accuracy row must still show learning."""
        payload = self._payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
            row["exact_match"] = 0.0
        text = partial_table(payload)
        self.assertIn("0.0%", text)      # honest zero retained
        self.assertIn("52.0%", text)     # but the learning signal is visible

    def test_saturation_without_savings_is_worded_honestly(self) -> None:
        """When the best point is the most expensive one, say the curve had not
        saturated -- do not claim a saving of zero tokens."""
        scale = _scale_payload()
        scale["saturation"]["tokens_saved"] = 0
        scale["saturation"]["fraction"] = 1.0
        d = scratch_dir("report_sat")
        try:
            summary = build_report(None, _ablation_payload(), scale, None, d,
                                   artifacts=_artifacts_payload())
            md = Path(summary["markdown"]).read_text(encoding="utf-8")
            self.assertIn("had not saturated yet", md)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# partial-credit figure + sample-size honesty
# --------------------------------------------------------------------------- #
class TestPartialFigure(unittest.TestCase):
    """The partial-credit figure must exist and be data-bearing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("report_partial_fig")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _rows(self) -> list[dict[str, Any]]:
        return [
            {"config": "raw_only", "synth_tokens": 0, "table_set_match": 0.083,
             "executable_rate": 0.208, "token_acc": 0.288},
            {"config": "plus_cot", "synth_tokens": 60_457,
             "table_set_match": 0.167, "executable_rate": 0.167, "token_acc": 0.226},
            {"config": "plus_hard_negatives", "synth_tokens": 239_883,
             "table_set_match": 0.167, "executable_rate": 0.042, "token_acc": 0.264},
        ]

    def test_figure_renders(self) -> None:
        path = plots.plot_cost_vs_partial(self._rows(),
                                          self.dir / "cost_vs_partial.png")
        raw = Path(path).read_bytes()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(raw), 5_000)

    def test_figure_handles_empty(self) -> None:
        path = plots.plot_cost_vs_partial([], self.dir / "empty.png")
        self.assertTrue(Path(path).exists())

    def test_figure_is_deterministic(self) -> None:
        a = plots.plot_cost_vs_partial(self._rows(), self.dir / "a.png")
        b = plots.plot_cost_vs_partial(self._rows(), self.dir / "b.png")
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())

    def test_figure_included_by_make_all_figures(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row.update(table_set_match=0.1, token_acc=0.25, executable_rate=0.2)
        made = plots.make_all_figures(self.dir / "all", ablation=payload)
        self.assertIn("cost_vs_partial", made)
        self.assertIn("cost_vs_quality", made)

    def test_figure_skipped_when_all_partial_zero(self) -> None:
        """No partial signal -> no misleading empty figure."""
        payload = _ablation_payload()
        for row in payload["rows"]:
            row.update(table_set_match=0.0, token_acc=0.0)
        made = plots.make_all_figures(self.dir / "none", ablation=payload)
        self.assertNotIn("cost_vs_partial", made)

    def test_report_embeds_the_figure(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row.update(table_set_match=0.1, token_acc=0.25)
        d = scratch_dir("report_pfig")
        try:
            summary = build_report(None, payload, _scale_payload(), None, d,
                                   artifacts=_artifacts_payload())
            md = Path(summary["markdown"]).read_text(encoding="utf-8")
            self.assertIn("cost_vs_partial.png", md)
            self.assertIn("cost_vs_quality.png", md)   # flat figure is kept
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestSampleSizeHonesty(unittest.TestCase):
    """The report must state how small the eval set is."""

    def test_sample_size_note_states_per_example_weight(self) -> None:
        d = scratch_dir("report_samplesize")
        try:
            summary = build_report(None, _ablation_payload(), _scale_payload(),
                                   None, d, artifacts=_artifacts_payload())
            md = Path(summary["markdown"]).read_text(encoding="utf-8")
            self.assertIn("identical across every configuration", md)
            self.assertIn("4.17 percentage points", md)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_findings_state_example_counts(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
        payload["rows"][0].update(table_set_match=0.083)
        payload["rows"][2].update(table_set_match=0.167)
        md = findings_markdown(payload)
        self.assertIn("How small is this sample", md)
        self.assertIn("2 of 24 examples", md)
        self.assertIn("not an established effect", md)

    def test_scale_trend_note_flags_non_monotone(self) -> None:
        scale = _scale_payload()
        for point, tbl in zip(scale["points"], (0.083, 0.25, 0.125)):
            point["table_set_match"] = tbl
        note = scale_trend_note(scale)
        self.assertIn("not monotone", note)
        self.assertIn("zig-zags", note)

    def test_scale_trend_note_accepts_monotone(self) -> None:
        scale = _scale_payload()
        for point, tbl in zip(scale["points"], (0.083, 0.125, 0.167)):
            point["table_set_match"] = tbl
        note = scale_trend_note(scale)
        self.assertIn("monotone increase", note)
        self.assertNotIn("not monotone", note)

    def test_scale_trend_note_needs_three_points(self) -> None:
        scale = _scale_payload()
        scale["points"] = scale["points"][:2]
        self.assertEqual(scale_trend_note(scale), "")


# --------------------------------------------------------------------------- #
# headline ratio
# --------------------------------------------------------------------------- #
class TestHeadline(unittest.TestCase):
    """The headline is arithmetic on measured values, or an explicit refusal."""

    def test_ratio_math(self) -> None:
        h = headline_ratio(_ablation_payload())
        self.assertTrue(h["available"])
        self.assertEqual(h["baseline_config"], "raw_only")
        self.assertEqual(h["rich_config"], "plus_hard_negatives")
        # (0.5833 - 0.25) * 100 = 33.33 pp over 0.12 Mtok -> 277.75 pp/Mtok
        self.assertAlmostEqual(h["gain_pp"], 33.33, places=2)
        self.assertAlmostEqual(h["points_per_mtok"], 277.75, places=2)
        self.assertEqual(h["synth_tokens"], 120_000)
        self.assertEqual(h["tokens_per_point"], 3600)
        self.assertAlmostEqual(h["relative_improvement_pct"], 133.3, places=1)

    def test_ratio_refuses_without_baseline(self) -> None:
        payload = _ablation_payload()
        payload["rows"] = [r for r in payload["rows"] if r["config"] != "raw_only"]
        h = headline_ratio(payload)
        self.assertFalse(h["available"])
        self.assertIn("baseline", h["reason"])

    def test_ratio_refuses_zero_token_investment(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["synth_tokens"] = 0
        h = headline_ratio(payload)
        self.assertFalse(h["available"])
        self.assertIn("no synthesis tokens", h["reason"])

    def test_ratio_refuses_missing_ablation(self) -> None:
        self.assertFalse(headline_ratio(None)["available"])

    def test_ratio_handles_zero_baseline_accuracy(self) -> None:
        """A 0% baseline must not divide-by-zero on relative improvement."""
        payload = _ablation_payload()
        payload["rows"][0]["exec_acc"] = 0.0
        h = headline_ratio(payload)
        self.assertTrue(h["available"])
        self.assertIsNone(h["relative_improvement_pct"])

    def test_ratio_flags_no_measurable_gain(self) -> None:
        """All-zero accuracy must not be reported as a tidy 0.00-points ratio."""
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
            row["exact_match"] = 0.0
        h = headline_ratio(payload)
        self.assertTrue(h["available"])
        self.assertTrue(h["no_measurable_gain"])
        self.assertIsNone(h["tokens_per_point"])
        self.assertIn("no measurable", h["reason"])

    def test_headline_markdown_explains_no_gain(self) -> None:
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
        md = headline_markdown(payload)
        self.assertIn("no measurable execution-accuracy gain", md.lower())
        self.assertIn("model capacity", md.lower())
        self.assertIn("not measurable (no gain)", md)
        self.assertNotIn("accuracy points per 1M synthesis tokens", md)

    def test_no_gain_headline_points_at_the_findings(self) -> None:
        """The headline must defer directional detail to Findings, not duplicate it."""
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
        payload["rows"][0].update(table_set_match=0.083)
        payload["rows"][2].update(table_set_match=0.167)
        md = headline_markdown(payload)
        self.assertIn("What did change", md)
        self.assertIn("See *Findings*", md)
        self.assertIn("How small is this sample", md)

    def test_findings_state_direction_of_each_partial_metric(self) -> None:
        """Findings must state which partial metric moved and in which direction."""
        payload = _ablation_payload()
        for row in payload["rows"]:
            row["exec_acc"] = 0.0
        payload["rows"][0].update(table_set_match=0.083, executable_rate=0.208,
                                  token_acc=0.288)
        payload["rows"][2].update(table_set_match=0.167, executable_rate=0.042,
                                  token_acc=0.264)
        md = findings_markdown(payload)
        self.assertIn("schema grounding", md)
        self.assertIn("8.3%", md)
        self.assertIn("16.7%", md)
        self.assertIn("Worsened with synthetic data", md)   # executability fell
        self.assertIn("4.2%", md)

    def test_zero_delta_is_not_labelled_baseline(self) -> None:
        """A config that was run and gained nothing must not look like the baseline."""
        payload = _ablation_payload()
        payload["rows"][1]["delta_exec_acc"] = 0.0
        text = ablation_table(payload)
        lines = [ln for ln in text.splitlines() if "plus_cot" in ln]
        self.assertTrue(lines)
        self.assertIn("±0.0 pp", lines[0])
        self.assertNotIn("baseline", lines[0])
        # the true baseline row still says "baseline"
        base = [ln for ln in text.splitlines() if "raw_only" in ln][0]
        self.assertIn("baseline", base)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
class TestPlots(unittest.TestCase):
    """Figures must be real PNGs, headless, deterministic and data-bearing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("report_plots")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _assert_png(self, path: Path) -> None:
        self.assertTrue(path.exists(), f"missing figure {path}")
        raw = path.read_bytes()
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n", "not a PNG")
        self.assertGreater(len(raw), 5_000, "suspiciously small PNG")
        # 160 dpi with the documented figsize -> at least ~1000px wide
        import struct

        width = struct.unpack(">I", raw[16:20])[0]
        height = struct.unpack(">I", raw[20:24])[0]
        self.assertGreater(width, 900)
        self.assertGreater(height, 500)

    def test_backend_is_headless(self) -> None:
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_cost_vs_quality_figure(self) -> None:
        out = self.dir / "cost_vs_quality.png"
        path = plots.plot_cost_vs_quality(_ablation_payload()["rows"], out)
        self._assert_png(Path(path))
        self.assertEqual(Path(path).name, "cost_vs_quality.png")

    def test_cost_vs_quality_handles_single_point(self) -> None:
        """One point means no trend line, but the figure must still render."""
        rows = _ablation_payload()["rows"][:1]
        path = plots.plot_cost_vs_quality(rows, self.dir / "single.png")
        self._assert_png(Path(path))

    def test_cost_vs_quality_handles_empty(self) -> None:
        path = plots.plot_cost_vs_quality([], self.dir / "empty.png")
        self._assert_png(Path(path))

    def test_scale_curve_figure(self) -> None:
        out = self.dir / "scale_curve.png"
        path = plots.plot_scale_curve(_scale_payload()["points"], out)
        self._assert_png(Path(path))

    def test_scale_curve_handles_empty(self) -> None:
        path = plots.plot_scale_curve([], self.dir / "scale_empty.png")
        self._assert_png(Path(path))

    def test_token_record_mapping_is_monotone(self) -> None:
        mapping = plots._TokenRecordMapping([0, 60_000, 120_000], [98, 260, 420])
        self.assertTrue(mapping.ok)
        # interior points are exact, and the map never extrapolates past the ends
        import numpy as np

        self.assertAlmostEqual(float(mapping.to_records(60_000)), 260.0)
        self.assertAlmostEqual(float(mapping.to_records(-5_000)), 98.0)
        self.assertAlmostEqual(float(mapping.to_records(500_000)), 420.0)
        self.assertAlmostEqual(float(mapping.to_tokens(420.0)), 120_000.0)
        self.assertTrue(np.all(np.diff(mapping.to_records([0, 60_000, 120_000])) >= 0))

    def test_token_record_mapping_rejects_degenerate_input(self) -> None:
        self.assertFalse(plots._TokenRecordMapping([0, 0], [1, 2]).ok)
        self.assertFalse(plots._TokenRecordMapping([5], [1]).ok)

    def test_loss_curve_from_trainer_history(self) -> None:
        metrics = {"history": [{"step": i, "epoch": 0, "loss": 4.0 - 0.4 * i,
                                "token_acc": 0.05 * i} for i in range(1, 6)]}
        path = plots.plot_loss_curve(metrics, self.dir / "loss_curve.png")
        self._assert_png(Path(path))

    def test_loss_curve_handles_parallel_lists(self) -> None:
        metrics = {"loss": [4.0, 3.0, 2.0], "token_acc": [0.1, 0.3, 0.5]}
        path = plots.plot_loss_curve(metrics, self.dir / "loss2.png")
        self._assert_png(Path(path))

    def test_loss_curve_handles_empty_metrics(self) -> None:
        path = plots.plot_loss_curve({}, self.dir / "loss_empty.png")
        self._assert_png(Path(path))

    def test_figures_are_deterministic(self) -> None:
        """Same input -> identical bytes (no random layout, no timestamps)."""
        a = plots.plot_cost_vs_quality(_ablation_payload()["rows"], self.dir / "det_a.png")
        b = plots.plot_cost_vs_quality(_ablation_payload()["rows"], self.dir / "det_b.png")
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())

    def test_make_all_figures_skips_missing_inputs(self) -> None:
        made = plots.make_all_figures(self.dir / "all", ablation=None, scale=None,
                                      train_metrics=None)
        self.assertEqual(made, {})
        made = plots.make_all_figures(
            self.dir / "all2", ablation=_ablation_payload(),
            scale=_scale_payload(),
            train_metrics={"history": [{"step": 1, "loss": 3.0, "token_acc": 0.2}]})
        self.assertEqual(set(made), {"cost_vs_quality", "scale_curve", "loss_curve"})
        for path in made.values():
            self._assert_png(Path(path))


# --------------------------------------------------------------------------- #
# the full report
# --------------------------------------------------------------------------- #
class TestBuildReport(unittest.TestCase):
    """End-to-end: one call produces md + csv + png + summary."""

    def setUp(self) -> None:
        self.dir = scratch_dir("report_full")
        self.cfg = _cfg(self.dir)
        self.out = self.dir / "reports"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _build(self, ablation: Any = None, scale: Any = None,
               ledger: Any = None, **kwargs: Any) -> dict[str, Any]:
        return build_report(
            self.cfg,
            _ablation_payload() if ablation is None else ablation,
            _scale_payload() if scale is None else scale,
            _ledger_payload() if ledger is None else ledger,
            self.out, **kwargs)

    def test_writes_all_artifacts(self) -> None:
        summary = self._build(artifacts=_artifacts_payload(),
                              train_metrics={"history": [
                                  {"step": 1, "loss": 3.5, "token_acc": 0.1},
                                  {"step": 2, "loss": 2.0, "token_acc": 0.4}]})
        self.assertTrue(Path(summary["markdown"]).exists())
        self.assertEqual(Path(summary["markdown"]).name, "README_TABLES.md")
        self.assertTrue((self.out / "cost_vs_quality.png").exists())
        self.assertTrue((self.out / "scale_curve.png").exists())
        self.assertTrue((self.out / "loss_curve.png").exists())
        self.assertTrue((self.out / "report_summary.json").exists())
        self.assertIn("ablation.csv", summary["csv"])
        self.assertIn("scale_curve.csv", summary["csv"])

    def test_markdown_contains_every_required_section(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        for heading in (
            "token investment",
            "Headline",
            "Figures",
            "Ablation",
            "Data-scale curve",
            "Validation funnel",
            "Token ledger by stage and role",
            "Extrapolation to Spider scale",
        ):
            self.assertIn(heading, md, f"missing section: {heading}")
        # every required table is present and well formed
        self.assertIn("| Config | SFT records |", md)
        self.assertIn("| Synth fraction | SFT records |", md)
        self.assertIn("| Stage | Records in |", md)
        self.assertIn("| Stage / role | Calls |", md)
        self.assertIn("| Metric | Value |", md)

    def test_markdown_tables_are_not_ragged(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        block: list[str] = []
        for line in md.splitlines() + [""]:
            if line.strip().startswith("|"):
                block.append(line)
                continue
            if block:
                widths = {b.count("|") for b in block}
                self.assertEqual(len(widths), 1, "ragged table:\n" + "\n".join(block))
                block = []

    def test_headline_section_states_the_ratio(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertIn("accuracy points per 1M synthesis tokens", md)
        self.assertIn("120,000", md)
        self.assertIn("+33.3", md)
        h = summary["headline"]
        self.assertTrue(h["available"])
        self.assertAlmostEqual(h["points_per_mtok"], 277.75, places=2)

    def test_quick_run_is_labelled_as_smoke(self) -> None:
        """A --quick number must never look like a reportable result."""
        summary = self._build(ablation=_ablation_payload(quick=True),
                              scale=_scale_payload(quick=True),
                              artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertIn("Smoke run", md)
        self.assertIn("not the achievable model quality", md)

    def test_estimated_cost_is_disclosed(self) -> None:
        """When cost did not come from the ledger the table must say so."""
        ablation = _ablation_payload()
        ablation["rows"][1]["cost_from_ledger"] = False
        summary = self._build(ablation=ablation, artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertTrue(
            "estimated" in md.lower() or "ledger" in md.lower(),
            "estimated cost must be disclosed in the report")

    def test_report_survives_missing_inputs(self) -> None:
        """No ablation, no scale, no ledger: the report still renders."""
        summary = build_report(self.cfg, None, None, None, self.out)
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertIn("not run", md)
        self.assertIn("unavailable", md)
        self.assertFalse(summary["headline"]["available"])
        self.assertEqual(summary["n_ablation_rows"], 0)

    def test_extrapolation_section_absent_by_default(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertIn("Token extrapolation not available", md)

    def test_extrapolation_section_rendered_when_present(self) -> None:
        """The Lead's projection table must appear when its JSON exists."""
        self.out.mkdir(parents=True, exist_ok=True)
        payload = {
            "measured": {"n_source_examples": 98, "n_synth_records": 322,
                         "synth_prompt_tokens": 80_000,
                         "synth_completion_tokens": 24_000,
                         "judge_prompt_tokens": 12_000,
                         "judge_completion_tokens": 4_000,
                         "synth_cost_usd": 0.0078, "judge_cost_usd": 0.0013},
            "projections": [
                {"label": "Spider train (7k)", "n_source_examples": 7_000,
                 "n_synth_records": 23_000, "plan": {"cot_per_example": 1,
                                                     "paraphrase_per_example": 2,
                                                     "hard_negative_per_example": 1},
                 "synth_tokens": 5_700_000, "judge_tokens": 1_140_000,
                 "total_tokens": 6_840_000, "synth_cost_usd": 0.56,
                 "judge_cost_usd": 0.09, "total_cost_usd": 0.65},
            ],
            "headline": {"largest_projection": "Spider train (7k)",
                         "total_tokens": 6_840_000},
        }
        (self.out / "token_extrapolation.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8")
        summary = self._build(artifacts=_artifacts_payload())
        md = Path(summary["markdown"]).read_text(encoding="utf-8")
        self.assertNotIn("Token extrapolation not available", md)
        self.assertIn("Spider train", md)

    def test_csv_roundtrip(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        path = Path(summary["csv"]["ablation.csv"])
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["config"], "raw_only")
        self.assertEqual(float(rows[2]["exec_acc"]), 0.5833)
        self.assertIn("synth_tokens", rows[0])
        # nested values are flattened to JSON, never a Python repr
        self.assertNotIn("'raw'", rows[0]["kind_counts"])

    def test_report_summary_json_is_written(self) -> None:
        summary = self._build(artifacts=_artifacts_payload())
        data = read_json(self.out / "report_summary.json")
        self.assertEqual(data["n_ablation_rows"], 3)
        self.assertEqual(data["n_scale_points"], 3)
        self.assertTrue(data["headline"]["available"])

    def test_plots_can_be_disabled(self) -> None:
        summary = self._build(artifacts=_artifacts_payload(), make_plots=False)
        self.assertEqual(summary["figures"], {})
        self.assertFalse((self.out / "cost_vs_quality.png").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
