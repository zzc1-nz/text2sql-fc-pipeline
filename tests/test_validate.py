"""Unit tests for the validation stage (format checks + semantic judge).

Self-contained: a two-table toy Spider-format database is built into a scratch
directory (``fcdata.config.scratch_dir`` -- ``tempfile.mkdtemp`` is unwritable in
this sandbox) and the judge is a local stub, so nothing here needs a network, an
API key, or the synthesis stage.

Run with::

    $env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_validate.py' -v
"""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fcdata.config import Config, load_config, scratch_dir
from fcdata.dbexec import build_sqlite
from fcdata.records import SynthRecord, read_validated
from fcdata.schema import Database
from fcdata.spider import SpiderData
from fcdata.validate import (build_validation_report, check_format, format_ok,
                             parse_judge_json, select_semantic_sample,
                             validate_and_write, validate_records)
from fcdata.validate.format_check import BLOCKING_CODES

# --------------------------------------------------------------------------- #
# tiny fixture
# --------------------------------------------------------------------------- #
TABLES_JSON: dict[str, Any] = {
    "db_id": "toy",
    "table_names_original": ["student", "department"],
    "table_names": ["student", "department"],
    "column_names_original": [[-1, "*"], [0, "stu_id"], [0, "stu_name"],
                              [0, "dept_id"], [1, "dept_id"], [1, "dept_name"]],
    "column_types": ["number", "number", "text", "number", "number", "text"],
    "primary_keys": [1, 4],
    "foreign_keys": [[3, 4]],
}

ROWS: dict[str, list[dict[str, Any]]] = {
    "student": [
        {"stu_id": 1, "stu_name": "Alice", "dept_id": 1},
        {"stu_id": 2, "stu_name": "Bob", "dept_id": 1},
        {"stu_id": 3, "stu_name": "Carol", "dept_id": 2},
        {"stu_id": 4, "stu_name": "Dan", "dept_id": 2},
    ],
    "department": [
        {"dept_id": 1, "dept_name": "CS"},
        {"dept_id": 2, "dept_name": "Math"},
    ],
}

GOOD_SQL = "SELECT stu_name FROM student WHERE dept_id = 1"
BAD_SQL_EXEC_ERROR = "SELECT nickname FROM student"
BAD_SQL_WRONG_ROWS = "SELECT stu_name FROM student WHERE dept_id = 2"
BAD_SQL_SAME_ROWS = "SELECT stu_name FROM student WHERE dept_id = 1 AND 1 = 1"


def make_db() -> Database:
    db = Database.from_spider_json(TABLES_JSON)
    db.rows = {k: [dict(r) for r in v] for k, v in ROWS.items()}
    return db


class FixtureCase(unittest.TestCase):
    """Builds the toy sqlite once per test class into a scratch directory."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = scratch_dir("validate_test")
        cls.db = make_db()
        cls.sqlite_dir = cls.root / "databases"
        cls.sqlite_dir.mkdir(parents=True, exist_ok=True)
        cls.db_path = build_sqlite(cls.db, cls.sqlite_dir / "toy.sqlite")
        cls.spider = SpiderData(cls.root, {"toy": cls.db}, [], [], cls.sqlite_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    # -- helpers --------------------------------------------------------
    def rec(self, sql: str = GOOD_SQL, kind: str = "cot", **kw: Any) -> SynthRecord:
        base = dict(id=kw.pop("id", "rec_1"), source_id="src_1", db_id="toy",
                    kind=kind, question="Which students are in CS?",
                    sql=sql, reasoning="I will read student rows.", difficulty=0.3)
        base.update(kw)
        return SynthRecord(**base)

    def cfg(self, **overrides: Any) -> Config:
        cfg = load_config("configs/demo.yaml", root=".") if Path("configs/demo.yaml").exists() \
            else Config()
        for key, value in overrides.items():
            section, field = key.split(".", 1)
            setattr(getattr(cfg, section), field, value)
        cfg.paths.root = str(self.root)
        return cfg


# --------------------------------------------------------------------------- #
# format stage
# --------------------------------------------------------------------------- #
class TestFormatCheck(FixtureCase):
    def test_good_sql_passes_all_three_checks(self) -> None:
        rep = check_format(self.rec(), self.db, self.db_path)
        self.assertTrue(rep.sql_parses)
        self.assertTrue(rep.schema_refs_ok)
        self.assertTrue(rep.executes)
        self.assertEqual(rep.exec_error, "")
        self.assertTrue(rep.passed)
        self.assertTrue(format_ok(rep))
        self.assertFalse(set(rep.issue_codes) & BLOCKING_CODES)

    def test_unknown_column_is_blocking_but_parses(self) -> None:
        rep = check_format(self.rec("SELECT nickname FROM student"), self.db, self.db_path)
        self.assertTrue(rep.sql_parses)
        self.assertFalse(rep.schema_refs_ok)
        self.assertFalse(rep.executes)
        self.assertIn("unknown_column", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_unknown_table_is_blocking(self) -> None:
        rep = check_format(self.rec("SELECT name FROM aliens"), self.db, self.db_path)
        self.assertIn("unknown_table", rep.issue_codes)
        self.assertFalse(rep.schema_refs_ok)
        self.assertFalse(rep.passed)

    def test_unbalanced_paren_fails_parse(self) -> None:
        rep = check_format(self.rec("SELECT * FROM student WHERE ("), self.db, self.db_path)
        self.assertFalse(rep.sql_parses)
        self.assertIn("unbalanced_paren", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_destructive_statement_rejected(self) -> None:
        rep = check_format(self.rec("DELETE FROM student"), self.db, self.db_path)
        self.assertFalse(rep.sql_parses)
        self.assertIn("destructive_statement", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_empty_sql_rejected(self) -> None:
        rep = check_format(self.rec("   "), self.db, self.db_path)
        self.assertIn("empty_sql", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_prose_wrapped_sql_is_repaired_and_flagged(self) -> None:
        rec = self.rec("Sure, here is the query: " + GOOD_SQL)
        rep = check_format(rec, self.db, self.db_path)
        self.assertEqual(rec.sql, GOOD_SQL, "validation must repair record.sql in place")
        self.assertIn("sql_normalized_from_text", rep.issue_codes)
        self.assertTrue(rep.passed)

    def test_missing_db_path_downgrades_execution_to_warning(self) -> None:
        rep = check_format(self.rec(), self.db, None)
        self.assertFalse(rep.executes)
        self.assertIn("exec_check_unavailable", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_schema_check_is_not_claimed_without_a_database(self) -> None:
        rep = check_format(self.rec(), None, self.db_path)
        self.assertFalse(rep.schema_refs_ok)
        self.assertIn("schema_check_unavailable", rep.issue_codes)

    def test_cot_without_reasoning_is_a_warning_only(self) -> None:
        rep = check_format(self.rec(reasoning=""), self.db, self.db_path)
        self.assertIn("cot_reasoning_missing", rep.issue_codes)
        self.assertTrue(rep.passed)

    def test_paraphrase_identical_question_is_a_warning(self) -> None:
        rec = self.rec(kind="paraphrase", question="Same question?",
                       original_question="Same question!")
        rep = check_format(rec, self.db, self.db_path)
        self.assertIn("paraphrase_identical", rep.issue_codes)
        self.assertTrue(rep.passed)


class TestHardNegativePremise(FixtureCase):
    def hn(self, bad_sql: str, error_kind: str = "wrong_column") -> SynthRecord:
        return self.rec(kind="hard_negative", bad_sql=bad_sql, error_kind=error_kind,
                        correction_note="that column belongs to another table")

    def test_missing_bad_sql_is_blocking(self) -> None:
        rep = check_format(self.hn(""), self.db, self.db_path)
        self.assertIn("hn_bad_sql_missing", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_bad_sql_identical_to_gold_is_blocking(self) -> None:
        rep = check_format(self.hn(GOOD_SQL), self.db, self.db_path)
        self.assertIn("hn_bad_sql_identical", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_bad_sql_identical_modulo_whitespace_is_blocking(self) -> None:
        rep = check_format(self.hn("select  stu_name  from student where dept_id=1"),
                           self.db, self.db_path)
        self.assertIn("hn_bad_sql_identical", rep.issue_codes)

    def test_execution_error_symptom_is_recorded_and_passes(self) -> None:
        rep = check_format(self.hn(BAD_SQL_EXEC_ERROR), self.db, self.db_path)
        self.assertIn("hn_symptom_exec_error", rep.issue_codes)
        self.assertTrue(rep.executes, "the GOLD query must execute for a hard negative")
        self.assertTrue(rep.passed)

    def test_wrong_rows_symptom_is_recorded_and_passes(self) -> None:
        rep = check_format(self.hn(BAD_SQL_WRONG_ROWS), self.db, self.db_path)
        self.assertIn("hn_symptom_wrong_result", rep.issue_codes)
        self.assertTrue(rep.passed)

    def test_negative_with_identical_result_set_is_blocking(self) -> None:
        rep = check_format(self.hn(BAD_SQL_SAME_ROWS), self.db, self.db_path)
        self.assertIn("hn_bad_sql_same_result", rep.issue_codes)
        self.assertFalse(rep.passed)

    def test_unknown_error_kind_is_only_a_warning(self) -> None:
        rep = check_format(self.hn(BAD_SQL_EXEC_ERROR, error_kind="not_a_real_kind"),
                           self.db, self.db_path)
        self.assertIn("hn_error_kind_unknown", rep.issue_codes)
        self.assertTrue(rep.passed)

    def test_bad_sql_does_not_replace_gold_sql(self) -> None:
        rec = self.hn(BAD_SQL_EXEC_ERROR)
        check_format(rec, self.db, self.db_path)
        self.assertEqual(rec.sql, GOOD_SQL)


# --------------------------------------------------------------------------- #
# semantic stage: JSON robustness
# --------------------------------------------------------------------------- #
class TestJudgeJsonParsing(unittest.TestCase):
    def test_clean_json(self) -> None:
        parsed = parse_judge_json('{"score": 0.9, "verdict": "consistent", "rationale": "ok"}')
        self.assertEqual(parsed, {"score": 0.9, "verdict": "consistent", "rationale": "ok"})

    def test_fenced_json(self) -> None:
        parsed = parse_judge_json(
            '```json\n{"score": 0.25, "verdict": "inconsistent", "rationale": "no"}\n```')
        self.assertEqual(parsed["score"], 0.25)
        self.assertEqual(parsed["verdict"], "inconsistent")

    def test_prose_prefix(self) -> None:
        parsed = parse_judge_json(
            'Sure! Here is my verdict: {"score": 0.8, "verdict": "consistent", "rationale": "yes"}')
        self.assertEqual(parsed["score"], 0.8)

    def test_score_as_string(self) -> None:
        parsed = parse_judge_json('{"score": "0.75", "verdict": "consistent"}')
        self.assertAlmostEqual(parsed["score"], 0.75)

    def test_score_on_0_100_scale_is_rescaled(self) -> None:
        parsed = parse_judge_json('{"score": 90, "verdict": "consistent"}')
        self.assertAlmostEqual(parsed["score"], 0.9)

    def test_score_is_clamped(self) -> None:
        self.assertEqual(parse_judge_json('{"score": 500, "verdict": "consistent"}')["score"], 1.0)
        self.assertEqual(parse_judge_json('{"score": -3, "verdict": "inconsistent"}')["score"], 0.0)

    def test_verdict_without_score_derives_one(self) -> None:
        self.assertEqual(parse_judge_json('{"verdict": "consistent"}')["score"], 1.0)
        self.assertEqual(parse_judge_json('{"verdict": "inconsistent"}')["score"], 0.0)

    def test_invalid_verdict_label_is_normalised_from_score(self) -> None:
        parsed = parse_judge_json('{"score": 0.9, "verdict": "great"}')
        self.assertEqual(parsed["verdict"], "consistent")

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(parse_judge_json("I cannot answer that."))
        self.assertIsNone(parse_judge_json(""))
        self.assertIsNone(parse_judge_json("{not json}"))

    def test_nested_braces_inside_rationale(self) -> None:
        parsed = parse_judge_json(
            '{"score": 0.4, "verdict": "uncertain", "rationale": "uses {a} weird syntax"}')
        self.assertEqual(parsed["score"], 0.4)
        self.assertIn("{a}", parsed["rationale"])


# --------------------------------------------------------------------------- #
# stub provider (the providers stage may not exist yet)
# --------------------------------------------------------------------------- #
@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int = 40
    completion_tokens: int = 12
    ok: bool = True
    error: str = ""
    cached: bool = False
    latency_ms: float = 1.0


class StubJudge:
    """Minimal object exposing the frozen provider contract."""

    name = "stub"
    model = "stub-judge"

    def __init__(self, score: float = 0.9, verdict: str = "consistent",
                 ok: bool = True, raw: str | None = None):
        self.score = score
        self.verdict = verdict
        self.ok = ok
        self.raw = raw
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], *, task: str = "",
                 sample_id: str = "", temperature: float | None = None,
                 max_tokens: int | None = None, json_mode: bool = False) -> FakeCompletion:
        self.calls.append({"messages": messages, "task": task,
                           "sample_id": sample_id, "json_mode": json_mode})
        if not self.ok:
            return FakeCompletion(text="", ok=False, error="simulated provider outage")
        text = self.raw if self.raw is not None else (
            '```json\n{"score": %s, "verdict": "%s", "rationale": "stub verdict"}\n```'
            % (self.score, self.verdict))
        return FakeCompletion(text=text)


class TestSemanticSampling(unittest.TestCase):
    def test_sample_zero_means_all(self) -> None:
        ids = [f"r{i}" for i in range(10)]
        self.assertEqual(select_semantic_sample(ids, 0, 17), set(ids))

    def test_sample_is_deterministic_and_order_independent(self) -> None:
        ids = [f"r{i}" for i in range(20)]
        a = select_semantic_sample(ids, 5, 17)
        b = select_semantic_sample(list(reversed(ids)), 5, 17)
        self.assertEqual(len(a), 5)
        self.assertEqual(a, b)

    def test_different_seed_picks_a_different_sample(self) -> None:
        ids = [f"r{i}" for i in range(50)]
        self.assertNotEqual(select_semantic_sample(ids, 5, 17),
                            select_semantic_sample(ids, 5, 99))


class TestValidateRecords(FixtureCase):
    def _records(self) -> list[SynthRecord]:
        return [
            self.rec(GOOD_SQL, id="r_good"),
            self.rec("SELECT nickname FROM student", id="r_bad_col"),
            self.rec(kind="hard_negative", bad_sql=BAD_SQL_EXEC_ERROR,
                     error_kind="wrong_column", id="r_hn"),
            self.rec(kind="paraphrase", question="Which pupils study CS?",
                     original_question="Which students are in CS?", id="r_para"),
        ]

    def test_one_validated_record_per_input(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider, provider=StubJudge())
        self.assertEqual(len(out), 4)
        self.assertEqual([v.id.startswith("val") for v in out], [True] * 4)

    def test_format_failure_is_kept_but_not_passed(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider, provider=StubJudge())
        by_id = {v.record.id: v for v in out}
        self.assertFalse(by_id["r_bad_col"].report.passed)
        self.assertTrue(by_id["r_good"].report.passed)

    def test_semantic_fields_populated(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider, provider=StubJudge(0.9))
        good = [v for v in out if v.record.id == "r_good"][0]
        self.assertTrue(good.report.semantic_checked)
        self.assertEqual(good.report.semantic_score, 0.9)
        self.assertEqual(good.report.semantic_verdict, "consistent")
        self.assertGreater(good.report.judge_tokens, 0)
        self.assertTrue(good.report.passed)

    def test_low_semantic_score_fails_the_record(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider,
                               provider=StubJudge(0.1, "inconsistent"))
        by_id = {v.record.id: v for v in out}
        self.assertFalse(by_id["r_good"].report.passed)
        self.assertIn("semantic_inconsistent", by_id["r_good"].report.issue_codes)

    def test_threshold_is_respected(self) -> None:
        cfg = self.cfg(**{"validate.semantic_threshold": 0.95})
        out = validate_records(cfg, self._records(), self.spider, provider=StubJudge(0.9))
        by_id = {v.record.id: v for v in out}
        self.assertFalse(by_id["r_good"].report.passed)

    def test_semantic_disabled_skips_the_judge(self) -> None:
        cfg = self.cfg(**{"validate.enable_semantic": False})
        stub = StubJudge()
        out = validate_records(cfg, self._records(), self.spider, provider=stub)
        self.assertEqual(stub.calls, [])
        self.assertFalse(any(v.report.semantic_checked for v in out))
        self.assertTrue(all(v.report.passed == format_ok(v.report) for v in out))

    def test_judge_only_sees_format_passing_records(self) -> None:
        cfg = self.cfg()
        stub = StubJudge()
        validate_records(cfg, self._records(), self.spider, provider=stub)
        judged = {c["sample_id"] for c in stub.calls}
        self.assertNotIn("r_bad_col", judged)
        self.assertIn("r_good", judged)

    def test_provider_outage_does_not_fail_a_format_valid_record(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider,
                               provider=StubJudge(ok=False))
        by_id = {v.record.id: v for v in out}
        self.assertTrue(by_id["r_good"].report.passed)
        self.assertIn("judge_error", by_id["r_good"].report.issue_codes)

    def test_unparseable_judge_reply_is_reported(self) -> None:
        cfg = self.cfg()
        out = validate_records(cfg, self._records(), self.spider,
                               provider=StubJudge(raw="I refuse to answer"))
        by_id = {v.record.id: v for v in out}
        self.assertTrue(by_id["r_good"].report.passed)
        self.assertIn("judge_parse_error", by_id["r_good"].report.issue_codes)

    def test_budget_stops_the_judge(self) -> None:
        cfg = self.cfg(**{"validate.max_total_tokens": 50})
        stub = StubJudge()
        validate_records(cfg, self._records(), self.spider, provider=stub)
        self.assertEqual(len(stub.calls), 1, "52 tokens per call exceeds a 50-token budget")

    def test_semantic_sample_limits_calls(self) -> None:
        cfg = self.cfg(**{"validate.semantic_sample": 1})
        stub = StubJudge()
        validate_records(cfg, self._records(), self.spider, provider=stub)
        self.assertEqual(len(stub.calls), 1)

    def test_unknown_db_id_is_a_blocking_code(self) -> None:
        rec = self.rec(id="r_nodb", db_id="ghost_db")
        cfg = self.cfg(**{"validate.enable_semantic": False})
        out = validate_records(cfg, [rec], self.spider, provider=StubJudge())
        self.assertFalse(out[0].report.passed)
        self.assertIn("unknown_db_id", out[0].report.issue_codes)

    def test_missing_sqlite_file_is_a_blocking_code(self) -> None:
        # schema is known but the database file was never built
        empty_dir = scratch_dir("validate_no_sqlite")
        try:
            spider = SpiderData(empty_dir, {"toy": self.db}, [], [], empty_dir)
            cfg = self.cfg(**{"validate.enable_semantic": False})
            out = validate_records(cfg, [self.rec(id="r_nofile")], spider,
                                   provider=StubJudge())
            self.assertFalse(out[0].report.passed)
            self.assertIn("missing_database", out[0].report.issue_codes)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_judge_prompt_contains_question_sql_and_ddl(self) -> None:
        cfg = self.cfg()
        stub = StubJudge()
        validate_records(cfg, [self.rec()], self.spider, provider=stub)
        user = stub.calls[0]["messages"][-1]["content"]
        self.assertIn("Which students are in CS?", user)
        self.assertIn(GOOD_SQL, user)
        self.assertIn("CREATE TABLE student", user)
        self.assertTrue(stub.calls[0]["json_mode"])


class TestJudgeProviderIntegration(FixtureCase):
    """Behaviours that only show up against a real provider implementation.

    These are the regressions the first real integration run caught: a prose
    prompt that the offline teacher cannot parse, and judge usage landing in the
    wrong ledger stage.
    """

    class TagAwareJudge(StubJudge):
        """Mimics the mock teacher: requires the provider's tag grammar."""

        def __init__(self) -> None:
            super().__init__(0.9)
            self.received: list[str] = []

        def complete(self, messages, **kw):  # type: ignore[override]
            self.received.append(messages[-1]["content"])
            return super().complete(messages, **kw)

    def test_judge_prompt_uses_the_provider_tag_grammar_when_available(self) -> None:
        try:
            from fcdata.providers import prompts as provider_prompts  # noqa: F401
        except ImportError:
            self.skipTest("fcdata.providers not importable yet")
        cfg = self.cfg()
        stub = self.TagAwareJudge()
        validate_records(cfg, [self.rec()], self.spider, provider=stub)
        prompt = stub.received[0]
        for tag in ("schema", "question", "candidate_sql"):
            self.assertIn(f"<{tag}>", prompt, f"judge prompt is missing <{tag}>")
        self.assertIn(GOOD_SQL, prompt)
        self.assertIn("Which students are in CS?", prompt)
        # the judge must not be told the answer: that would make it circular
        self.assertNotIn("<gold_sql>", prompt)

    def test_judge_forwards_the_validate_stage_to_the_provider(self) -> None:
        seen: list[str] = []

        class StageJudge:
            def complete(self, messages, *, task="", sample_id="", temperature=None,
                         max_tokens=None, json_mode=False, stage=""):
                seen.append(stage)
                return FakeCompletion(
                    text='{"score": 0.9, "verdict": "consistent", "rationale": "ok"}')

        validate_records(self.cfg(), [self.rec()], self.spider, provider=StageJudge())
        self.assertEqual(seen, ["validate"])

    def test_provider_without_stage_kwarg_still_works(self) -> None:
        class LegacyJudge:
            def complete(self, messages, *, task="", sample_id="",
                         temperature=None, max_tokens=None, json_mode=False):
                return FakeCompletion(
                    text='{"score": 0.9, "verdict": "consistent", "rationale": "ok"}')

        out = validate_records(self.cfg(), [self.rec()], self.spider,
                               provider=LegacyJudge())
        self.assertTrue(out[0].report.passed)
        self.assertEqual(out[0].report.semantic_score, 0.9)

    def test_judge_usage_can_be_recorded_once_when_asked(self) -> None:
        from fcdata.tokens import TokenLedger

        ledger = TokenLedger(None)
        validate_records(self.cfg(), [self.rec()], self.spider,
                         ledger, provider=StubJudge(), record_usage_in_ledger=True)
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0].stage, "validate")
        self.assertEqual(ledger.records[0].role, "judge")

    def test_judge_usage_is_not_double_counted_by_default(self) -> None:
        from fcdata.tokens import TokenLedger

        ledger = TokenLedger(None)
        validate_records(self.cfg(), [self.rec()], self.spider, ledger,
                         provider=StubJudge())
        self.assertEqual(len(ledger.records), 0)


class TestGroundTruthAudit(FixtureCase):
    """A drop rate is only meaningful if you know whether the drops were wrong.

    ``SynthRecord.sql`` is the gold SQL by contract, so a record whose SQL still
    executes to the same result as its source example was a *correct* record that
    the judge wrongly deleted. The report must measure that, not imply it.
    """

    def _source_example(self):
        from fcdata.records import Example

        return Example(id="src_1", db_id="toy",
                       question="Which students are in CS?",
                       sql=GOOD_SQL, split="train")

    def test_correct_but_rejected_record_is_counted_as_a_false_negative(self) -> None:
        cfg = self.cfg()
        example = self._source_example()
        spider = SpiderData(self.root, {"toy": self.db}, [example], [], self.sqlite_dir)
        rec = self.rec(GOOD_SQL, id="rec_1")
        rec.source_id = example.id
        validated = validate_records(cfg, [rec], spider,
                                     provider=StubJudge(0.1, "inconsistent"))
        self.assertFalse(validated[0].report.passed)
        report = build_validation_report(cfg, validated, None, None,
                                         spider_data=spider)
        audit = report["ground_truth_audit"]
        self.assertTrue(audit["available"])
        self.assertEqual(audit["n_dropped"], 1)
        self.assertEqual(audit["n_correct"], 1)
        self.assertEqual(audit["n_false_negative"], 1)
        self.assertEqual(audit["n_false_negative_rate"], 1.0)
        self.assertEqual(audit["by_kind"]["cot"]["provably_correct"], 1)
        self.assertEqual(len(audit["examples"]), 1)

    def test_genuinely_wrong_drop_is_not_a_false_negative(self) -> None:
        cfg = self.cfg()
        example = self._source_example()
        spider = SpiderData(self.root, {"toy": self.db}, [example], [], self.sqlite_dir)
        # executes fine, but answers a different question -> a legitimate drop
        rec = self.rec(BAD_SQL_WRONG_ROWS, id="rec_bad")
        rec.source_id = example.id
        validated = validate_records(cfg, [rec], spider,
                                     provider=StubJudge(0.1, "inconsistent"))
        self.assertFalse(validated[0].report.passed)
        report = build_validation_report(cfg, validated, None, None, spider_data=spider)
        audit = report["ground_truth_audit"]
        self.assertEqual(audit["n_dropped"], 1)
        self.assertEqual(audit["n_correct"], 0)
        self.assertEqual(audit["n_false_negative"], 0)
        self.assertEqual(audit["n_false_negative_rate"], 0.0)

    def test_format_failures_are_excluded_from_the_semantic_audit(self) -> None:
        cfg = self.cfg()
        example = self._source_example()
        spider = SpiderData(self.root, {"toy": self.db}, [example], [], self.sqlite_dir)
        rec = self.rec("SELECT nickname FROM student", id="rec_bad")
        rec.source_id = example.id
        validated = validate_records(cfg, [rec], spider, provider=StubJudge())
        report = build_validation_report(cfg, validated, None, None, spider_data=spider)
        # the record was dropped by the FORMAT gate, so the judge never saw it
        self.assertEqual(report["ground_truth_audit"]["n_dropped"], 0)
        self.assertEqual(report["stages"]["format"]["passed"], 0)

    def test_audit_is_marked_unavailable_without_spider_data(self) -> None:
        cfg = self.cfg()
        validated = validate_records(cfg, [self.rec()], self.spider, provider=StubJudge())
        report = build_validation_report(cfg, validated, None, None)
        self.assertFalse(report["ground_truth_audit"]["available"])

    def test_audit_reports_zero_drops_cleanly(self) -> None:
        cfg = self.cfg()
        example = self._source_example()
        spider = SpiderData(self.root, {"toy": self.db}, [example], [], self.sqlite_dir)
        rec = self.rec(GOOD_SQL, id="rec_1")
        rec.source_id = example.id
        validated = validate_records(cfg, [rec], spider, provider=StubJudge(0.9))
        report = build_validation_report(cfg, validated, None, None, spider_data=spider)
        audit = report["ground_truth_audit"]
        self.assertEqual(audit["n_dropped"], 0)
        self.assertEqual(audit["n_false_negative"], 0)
        self.assertEqual(audit["n_false_negative_rate"], 0.0)
        self.assertEqual(audit["precision_of_kept"], 1.0)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
class TestValidationReport(FixtureCase):
    def _run(self) -> tuple[list, dict[str, Any]]:
        cfg = self.cfg()
        records = [
            self.rec(GOOD_SQL, id="r_good"),
            self.rec("SELECT nickname FROM student", id="r_bad_col"),
            self.rec(kind="hard_negative", bad_sql=BAD_SQL_EXEC_ERROR,
                     error_kind="wrong_column", id="r_hn1"),
            self.rec(kind="hard_negative", bad_sql=BAD_SQL_SAME_ROWS,
                     error_kind="wrong_operator", id="r_hn2"),
            self.rec(kind="paraphrase", question="Which pupils study CS?",
                     original_question="Which students are in CS?", id="r_para"),
        ]
        validated = validate_records(cfg, records, self.spider, provider=StubJudge(0.8))
        out_dir = scratch_dir("validate_report")
        try:
            report = build_validation_report(cfg, validated, None, out_dir)
            return validated, report
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_report_counts_and_rates(self) -> None:
        validated, report = self._run()
        self.assertEqual(report["counts"]["n"], 5)
        self.assertEqual(report["counts"]["by_kind"]["cot"], 2)
        self.assertEqual(report["counts"]["by_kind"]["hard_negative"], 2)
        fmt = report["stages"]["format"]
        self.assertEqual(fmt["checked"], 5)
        # passes: good cot, hn_exec_error, paraphrase; fails: bad column, hn_same_result
        self.assertEqual(fmt["passed"], 3)
        self.assertAlmostEqual(fmt["pass_rate"], 0.6, places=4)
        self.assertEqual(report["stages"]["combined"]["passed"], 3)

    def test_funnel_is_monotonic_after_the_amplification_step(self) -> None:
        _, report = self._run()
        rows = report["funnel"]
        # raw -> synth is amplification (may exceed 100%); everything after must not
        amp = [r for r in rows if r.get("is_amplification")]
        self.assertEqual([r["stage"] for r in amp], ["synth"])
        survival = [r["n"] for r in rows if r["stage"] != "synth"]
        self.assertEqual(survival, sorted(survival, reverse=True))
        self.assertEqual(report["funnel"][0]["n"], 5)

    def test_funnel_separates_format_from_semantic_pass(self) -> None:
        _, report = self._run()
        stages = {row["stage"]: row["n"] for row in report["funnel"]}
        self.assertIn("format_pass", stages)
        self.assertIn("semantic_pass", stages)
        self.assertIn("sft", stages)
        self.assertEqual(stages["format_pass"], 3)
        self.assertEqual(stages["semantic_pass"], 3)
        self.assertEqual(stages["sft"], 3)

    def test_funnel_starts_at_raw_when_the_raw_count_is_known(self) -> None:
        cfg = self.cfg()
        validated = validate_records(cfg, [self.rec()], self.spider, provider=StubJudge())
        report = build_validation_report(cfg, validated, None, None, n_raw=98)
        stages = [row["stage"] for row in report["funnel"]]
        self.assertEqual(stages, ["raw", "synth", "format_pass", "semantic_checked",
                                  "semantic_pass", "sft"])
        self.assertEqual(report["funnel"][0]["n"], 98)
        self.assertEqual(report["funnel"][1]["n"], 1)
        # amplification: 1 synth record from 98 raw examples
        self.assertAlmostEqual(report["funnel"][1]["pct_of_raw"], 1.02, places=2)

    def test_funnel_omits_raw_when_unknown(self) -> None:
        _, report = self._run()
        self.assertNotIn("raw", [row["stage"] for row in report["funnel"]])

    def test_funnel_marks_the_amplification_step(self) -> None:
        cfg = self.cfg()
        validated = validate_records(cfg, [self.rec()], self.spider, provider=StubJudge())
        report = build_validation_report(cfg, validated, None, None, n_raw=98)
        by_stage = {row["stage"]: row for row in report["funnel"]}
        self.assertTrue(by_stage["synth"]["is_amplification"])
        self.assertFalse(by_stage["format_pass"]["is_amplification"])
        self.assertLess(by_stage["synth"]["pct"], 100.0)   # 1 record from 98 seeds

    def test_per_kind_format_and_semantic_rates_are_separate(self) -> None:
        _, report = self._run()
        hn = report["by_kind"]["hard_negative"]
        self.assertEqual(hn["format_pass_rate"], 0.5)   # one of two premises is false
        self.assertEqual(hn["semantic_pass_rate"], 1.0)
        para = report["by_kind"]["paraphrase"]
        self.assertEqual(para["format_pass_rate"], 1.0)
        self.assertEqual(para["semantic_pass_rate"], 1.0)

    def test_judge_score_distribution_is_reported(self) -> None:
        validated, report = self._run()
        dist = report["judge_score_distribution"]
        self.assertEqual(dist["threshold"], self.cfg().validate.semantic_threshold)
        self.assertEqual(sum(dist["histogram"].values()), 3)   # 3 judged records
        self.assertEqual(dist["by_verdict"]["consistent"]["n"], 3)
        self.assertAlmostEqual(dist["by_verdict"]["consistent"]["mean"], 0.8, places=3)
        self.assertEqual(dist["n_at_or_above_threshold"], 3)

    def test_threshold_sweep_is_self_documenting(self) -> None:
        validated, report = self._run()
        sweep = report["threshold_sweep"]
        self.assertTrue(sweep["available"])
        self.assertEqual(sweep["n_judged"], 3)
        rows = {row["threshold"]: row for row in sweep["rows"]}
        self.assertIn(0.4, rows)
        self.assertIn(0.5, rows)
        configured = [r for r in sweep["rows"] if r["is_configured"]]
        self.assertEqual(len(configured), 1)
        self.assertEqual(configured[0]["threshold"],
                         self.cfg().validate.semantic_threshold)
        # a higher threshold can never keep more records
        ordered = [rows[t]["kept"] for t in sorted(rows)]
        self.assertEqual(ordered, sorted(ordered, reverse=True))

    def test_threshold_sweep_reports_precision_against_ground_truth(self) -> None:
        cfg = self.cfg()
        from fcdata.records import Example

        example = Example(id="src_1", db_id="toy",
                          question="Which students are in CS?",
                          sql=GOOD_SQL, split="train")
        spider = SpiderData(self.root, {"toy": self.db}, [example], [], self.sqlite_dir)
        rec = self.rec(GOOD_SQL, id="rec_1")
        rec.source_id = example.id
        validated = validate_records(cfg, [rec], spider,
                                     provider=StubJudge(0.1, "inconsistent"))
        report = build_validation_report(cfg, validated, None, None, spider_data=spider)
        sweep = report["threshold_sweep"]
        self.assertTrue(sweep["precision_defined"])
        row = {r["threshold"]: r for r in sweep["rows"]}[0.4]
        # the only record is provably correct and dropped, so recall is 0 at any
        # threshold above its score -- an honest number, not a flattering one
        self.assertEqual(row["kept"], 0)
        self.assertEqual(row["false_negatives"], 1)
        self.assertEqual(row["recall"], 0.0)
        self.assertIsNone(row["precision"])

    def test_threshold_sweep_precision_is_none_without_ground_truth(self) -> None:
        _, report = self._run()          # _run() supplies no spider_data to the report
        sweep = report["threshold_sweep"]
        self.assertFalse(sweep["precision_defined"])
        self.assertTrue(all(r["precision"] is None for r in sweep["rows"]))

    def test_success_codes_are_not_listed_as_failure_codes(self) -> None:
        _, report = self._run()
        codes = [c["code"] for c in report["top_failure_codes"]]
        self.assertNotIn("semantic_consistent", codes)

    def test_judge_token_accounting(self) -> None:
        _, report = self._run()
        judge = report["judge"]
        self.assertEqual(judge["calls"], 3, "the judge only sees format-passing records")
        self.assertEqual(judge["total_tokens"], 3 * (40 + 12))
        self.assertGreater(judge["avg_tokens_per_call"], 0)

    def test_failure_codes_are_ranked(self) -> None:
        _, report = self._run()
        codes = {c["code"]: c for c in report["top_failure_codes"]}
        self.assertIn("unknown_column", codes)
        self.assertTrue(codes["unknown_column"]["blocking"])
        self.assertIn("hn_bad_sql_same_result", codes)

    def test_hard_negative_breakdown_by_error_kind(self) -> None:
        _, report = self._run()
        hn = report["hard_negatives"]
        self.assertEqual(hn["n"], 2)
        self.assertEqual(hn["by_error_kind"]["wrong_column"]["n"], 1)
        self.assertEqual(hn["by_error_kind"]["wrong_column"]["kept"], 1)
        self.assertEqual(hn["by_error_kind"]["wrong_operator"]["kept"], 0)
        self.assertEqual(hn["by_error_kind"]["wrong_operator"]["bad_sql_same_result"], 1)

    def test_report_written_to_disk(self) -> None:
        cfg = self.cfg()
        validated = validate_records(cfg, [self.rec()], self.spider, provider=StubJudge())
        out_dir = scratch_dir("validate_report_disk")
        try:
            build_validation_report(cfg, validated, None, out_dir)
            path = out_dir / "validation_report.json"
            self.assertTrue(path.exists())
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["counts"]["n"], 1)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


class TestValidateAndWrite(FixtureCase):
    def test_drop_failed_controls_the_artifact_but_not_the_report(self) -> None:
        cfg = self.cfg()
        out_dir = scratch_dir("validate_write")
        try:
            records = [self.rec(GOOD_SQL, id="r_good"),
                       self.rec("SELECT nickname FROM student", id="r_bad")]
            report = validate_and_write(cfg, records, self.spider, None,
                                        provider=StubJudge(), out_dir=out_dir)
            written = read_validated(out_dir / "validated.jsonl")
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].record.id, "r_good")
            self.assertEqual(report["counts"]["n"], 2)
            self.assertEqual(report["stages"]["combined"]["passed"], 1)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_keep_failed_writes_everything(self) -> None:
        cfg = self.cfg(**{"validate.drop_failed": False})
        out_dir = scratch_dir("validate_write_keep")
        try:
            records = [self.rec(GOOD_SQL, id="r_good"),
                       self.rec("SELECT nickname FROM student", id="r_bad")]
            validate_and_write(cfg, records, self.spider, None,
                               provider=StubJudge(), out_dir=out_dir)
            self.assertEqual(len(read_validated(out_dir / "validated.jsonl")), 2)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
