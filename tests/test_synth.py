"""Tests for ``fcdata.synth``: CoT, paraphrase and verified hard negatives.

What is asserted (and why)
--------------------------
The synthesis stage is the *expensive* stage, so its invariants are about not
wasting tokens and not poisoning the corpus:

* **Contract shape.** Every record has the fields CONTRACTS.md promises, the gold
  SQL as ``sql``, and a real non-zero token count. A record with zero tokens would
  make the token-investment report wrong.
* **Hard negatives are real negatives.** For every produced record the test
  *executes* both queries against a real sqlite fixture and asserts the corrupted
  query is observably different (errors or returns a different result set). This
  is the exact failure the Lead measured: ``inject_error`` no-ops on most kinds for
  most queries, so a naive implementation emits ``bad_sql == gold``.
* **Difficulty stratification.** With the flag on, the negative budget must be
  concentrated on harder examples; the test compares the mean difficulty of
  covered examples against the corpus mean.
* **Determinism.** Two runs must produce identical ids and text (the reproducibility
  claim in the README).
* **Budget.** A token cap must truncate the corpus, not overrun it.
* **Robustness.** An example whose every error kind is a no-op must be skipped,
  not emitted.

Tests build their own tiny Spider fixture (no dependency on ``data/fixture``).
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from typing import Any

from fcdata.config import Config, ProviderCfg, scratch_dir
from fcdata.dbexec import build_sqlite, evaluate_pair, execute_sql
from fcdata.providers import MockProvider
from fcdata.records import Example, SynthRecord
from fcdata.schema import Database
from fcdata.spider import SpiderData, load_spider
from fcdata.sqlutils import ERROR_KINDS, analyze_sql, normalize_sql
from fcdata.synth import KIND_ORDER, SYNTH_STAGE, build_synth_records, synthesize
from fcdata.synth.common import (build_sources, dedupe_questions, normalize_question,
                                 record_id, stable_rng)
from fcdata.synth.cot import synth_cot, validate_trace
from fcdata.synth.hard_negative import (PREFERRED_KINDS, corrupt_candidate,
                                        pick_negatives, verify_negative)
from fcdata.synth.paraphrase import (SAFE_STRATEGIES, clean_rewrite,
                                     pick_strategies, synth_paraphrase)
from fcdata.tokens import TokenLedger


# --------------------------------------------------------------------------- #
# tiny Spider fixture
# --------------------------------------------------------------------------- #
def fixture_tables() -> dict[str, Any]:
    return {
        "db_id": "library",
        "table_names_original": ["author", "book"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "name"],
                                  [1, "id"], [1, "author_id"], [1, "title"],
                                  [1, "year"]],
        "column_types": ["text", "number", "text", "number", "number",
                         "text", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    }


def fixture_examples() -> list[dict[str, Any]]:
    """Questions chosen so every hard-negative kind has something to corrupt."""
    return [
        {"id": "ex0", "db_id": "library",
         "question": "Show the names of authors who wrote a book after 2000.",
         "query": ("SELECT T1.name FROM author AS T1 JOIN book AS T2 "
                   "ON T1.id = T2.author_id WHERE T2.year > 2000")},
        {"id": "ex1", "db_id": "library",
         "question": "How many books are there?",
         "query": "SELECT COUNT(*) FROM book"},
        {"id": "ex2", "db_id": "library",
         "question": "List each author and the number of books they wrote.",
         "query": ("SELECT T1.name, COUNT(*) FROM author AS T1 JOIN book AS T2 "
                   "ON T1.id = T2.author_id GROUP BY T1.name")},
        {"id": "ex3", "db_id": "library",
         "question": "Show the titles of books published after 1990, newest first.",
         "query": ("SELECT title FROM book WHERE year > 1990 "
                   "ORDER BY year DESC")},
        {"id": "ex4", "db_id": "library",
         "question": "List the distinct years in which books were published.",
         "query": "SELECT DISTINCT year FROM book"},
        {"id": "ex5", "db_id": "library",
         "question": "Which books have never been borrowed?",
         "query": ("SELECT title FROM book WHERE id NOT IN "
                   "(SELECT id FROM book WHERE year IS NULL)")},
    ]


class FixtureMixin:
    _dirs: list[Path]

    def _make_data(self, name: str = "synth") -> SpiderData:
        d = scratch_dir(name)
        self._dirs.append(d)
        tables = fixture_tables()
        examples = fixture_examples()
        db = Database.from_spider_json(tables)
        db.rows = {
            "author": [{"id": 1, "name": "Ann"}, {"id": 2, "name": "Bob"},
                       {"id": 3, "name": "Cid"}],
            "book": [{"id": 10, "author_id": 1, "title": "A", "year": 2001},
                     {"id": 11, "author_id": 1, "title": "B", "year": 2005},
                     {"id": 12, "author_id": 2, "title": "C", "year": 1999},
                     {"id": 13, "author_id": 3, "title": "D", "year": 2010}],
        }
        db_dir = d / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        build_sqlite(db, db_dir / "library.sqlite")
        (d / "tables.json").write_text(json.dumps([tables]), encoding="utf-8")
        (d / "train.json").write_text(json.dumps(examples), encoding="utf-8")
        (d / "dev.json").write_text(json.dumps(examples[:2]), encoding="utf-8")
        return load_spider(d)

    def tearDown(self) -> None:  # noqa: D401 - unittest hook
        for d in getattr(self, "_dirs", []):
            shutil.rmtree(d, ignore_errors=True)
        self._dirs = []


def make_cfg(**synth: Any) -> Config:
    cfg = Config()
    cfg.providers["cheap"] = ProviderCfg(name="mock", model="mock-cheap-1",
                                         concurrency=4, temperature=0.7,
                                         max_tokens=512, price_in_per_mtok=0.05,
                                         price_out_per_mtok=0.10)
    cfg.synth.cot_per_example = 1
    cfg.synth.paraphrase_per_example = 2
    cfg.synth.hard_negative_per_example = 1
    cfg.synth.seed = 17
    for key, value in synth.items():
        setattr(cfg.synth, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# planning helpers
# --------------------------------------------------------------------------- #
class TestPlanning(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.data = self._make_data("synth_plan")
        self.cfg = make_cfg()

    def test_build_sources_grounds_every_plan(self) -> None:
        plans = build_sources(self.cfg, self.data)
        self.assertEqual(len(plans), len(self.data.train))
        for plan in plans:
            self.assertIn("CREATE TABLE", plan.ddl)
            self.assertFalse(plan.schema.is_empty)
            self.assertTrue(plan.facts.select_items)
            self.assertEqual(plan.gold_sql, plan.example.sql)

    def test_source_selection_is_deterministic(self) -> None:
        a = [p.example.id for p in build_sources(self.cfg, self.data)]
        b = [p.example.id for p in build_sources(self.cfg, self.data)]
        self.assertEqual(a, b)

    def test_max_source_examples_caps_and_stays_seeded(self) -> None:
        cfg = make_cfg(max_source_examples=3)
        a = [p.example.id for p in build_sources(cfg, self.data)]
        b = [p.example.id for p in build_sources(cfg, self.data)]
        self.assertEqual(len(a), 3)
        self.assertEqual(a, b)

    def test_record_id_is_deterministic_and_safe(self) -> None:
        self.assertEqual(record_id("cot", "ex 0/x", 1), record_id("cot", "ex 0/x", 1))
        self.assertNotEqual(record_id("cot", "ex0", 0), record_id("cot", "ex0", 1))
        self.assertNotIn(" ", record_id("cot", "a b c", 0))

    def test_stable_rng_is_process_independent(self) -> None:
        self.assertEqual(stable_rng("a", "b").random(), stable_rng("a", "b").random())

    def test_normalize_question_collapses_style(self) -> None:
        self.assertEqual(normalize_question("Please list the books?"),
                         normalize_question("List the books."))
        self.assertNotEqual(normalize_question("List the books"),
                            normalize_question("List the authors"))

    def test_dedupe_questions(self) -> None:
        recs = [
            SynthRecord(id="a", source_id="s", db_id="d", kind="paraphrase",
                        question="List the books.", sql="SELECT 1"),
            SynthRecord(id="b", source_id="s", db_id="d", kind="paraphrase",
                        question="Please list the books?", sql="SELECT 1"),
            SynthRecord(id="c", source_id="s", db_id="d", kind="paraphrase",
                        question="Show the authors.", sql="SELECT 2"),
        ]
        kept, dropped = dedupe_questions(recs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 1)


# --------------------------------------------------------------------------- #
# hard-negative verification (the critical invariant)
# --------------------------------------------------------------------------- #
class TestHardNegativeVerification(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.data = self._make_data("synth_hn")
        self.cfg = make_cfg()
        self.plans = {p.example.id: p for p in build_sources(self.cfg, self.data)}

    def test_corrupt_candidate_rejects_noops(self) -> None:
        """``inject_error`` is a no-op for most kinds; those must be rejected."""
        plan = self.plans["ex1"]          # SELECT COUNT(*) FROM book: no JOIN/GROUP BY
        accepted, bad, reason = corrupt_candidate(
            plan, "missing_join", self.data.db_path(plan.db_id))
        self.assertFalse(accepted)
        self.assertEqual(bad, "")
        self.assertTrue(reason)

    def test_corrupt_candidate_accepts_a_real_corruption(self) -> None:
        plan = self.plans["ex1"]
        accepted, bad, reason = corrupt_candidate(
            plan, "wrong_aggregate", self.data.db_path(plan.db_id))
        self.assertTrue(accepted, reason)
        self.assertNotEqual(normalize_sql(bad), normalize_sql(plan.gold_sql))
        self.assertTrue(analyze_sql(bad, plan.db).ok)

    def test_every_accepted_negative_is_observably_wrong(self) -> None:
        """The core invariant: a negative must differ *in execution*, not just text."""
        checked = 0
        for plan in self.plans.values():
            db_path = self.data.db_path(plan.db_id)
            for kind in PREFERRED_KINDS:
                accepted, bad, reason = corrupt_candidate(plan, kind, db_path)
                if not accepted:
                    continue
                checked += 1
                self.assertNotEqual(normalize_sql(bad), normalize_sql(plan.gold_sql))
                pair = evaluate_pair(db_path, bad, plan.gold_sql)
                self.assertTrue(pair["gold_ok"], pair)
                # a schema-invalid statement cannot execute, which *is* an
                # observable difference; a schema-valid one must return a
                # different result set.
                self.assertFalse(pair["exec_match"],
                                 f"{kind} produced an equivalent query: {bad}")
        self.assertGreater(checked, 5, "verification test covered too few negatives")

    def test_verify_negative_helper_reports_differences(self) -> None:
        plan = self.plans["ex0"]
        db_path = self.data.db_path(plan.db_id)
        accepted, bad, reason = corrupt_candidate(plan, "wrong_table", db_path)
        self.assertTrue(accepted, reason)
        verdict = verify_negative(db_path, plan.gold_sql, bad)
        self.assertFalse(verdict["identical"])
        self.assertTrue(verdict["exec_differs"])

    def test_schema_broken_corruption_is_still_a_valid_negative(self) -> None:
        """A hallucinated column is a realistic mistake, not an invalid negative."""
        plan = self.plans["ex0"]
        db_path = self.data.db_path(plan.db_id)
        # wrong_table on a JOIN query yields a statement referencing a column the
        # swapped-in table does not have -- schema-invalid but exactly the kind of
        # error a text2sql model really makes.
        accepted, bad, reason = corrupt_candidate(plan, "wrong_table", db_path)
        self.assertTrue(accepted, reason)
        analysis = analyze_sql(bad, plan.db)
        self.assertFalse(analysis.ok)
        self.assertTrue(any(i.code.startswith("unknown_") for i in analysis.errors))

    def test_pick_negatives_returns_only_viable_kinds(self) -> None:
        plan = self.plans["ex2"]          # has JOIN + GROUP BY
        db_path = self.data.db_path(plan.db_id)
        found = pick_negatives(plan, PREFERRED_KINDS, db_path)
        self.assertGreaterEqual(len(found), 3)
        for kind, bad, reason in found:
            self.assertIn(kind, ERROR_KINDS)
            self.assertNotEqual(normalize_sql(bad), normalize_sql(plan.gold_sql))
            self.assertTrue(reason)


# --------------------------------------------------------------------------- #
# CoT
# --------------------------------------------------------------------------- #
class TestCot(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.data = self._make_data("synth_cot")
        self.cfg = make_cfg()
        self.ledger = TokenLedger(None)

    def test_cot_records_are_grounded_and_counted(self) -> None:
        records, report = synth_cot(self.cfg, self.data, self.ledger)
        self.assertGreater(len(records), 0)
        self.assertEqual(report.ok, len(records))
        plans = self.plans()
        for rec in records:
            self.assertEqual(rec.kind, "cot")
            self.assertEqual(rec.sql, plans[rec.source_id].gold_sql)
            self.assertIn("Final SQL:", rec.reasoning)
            self.assertGreater(rec.prompt_tokens, 0)
            self.assertGreater(rec.completion_tokens, 0)
            self.assertEqual(rec.generator_role, "cheap")
            self.assertEqual(rec.generator_model, "mock-cheap-1")
            # grounded: the trace names a real table of *this* example's schema
            tables = plans[rec.source_id].schema.table_names
            self.assertTrue(any(t in rec.reasoning for t in tables),
                            f"trace names no real table: {rec.reasoning!r}")

    def plans(self) -> dict[str, Any]:
        return {p.example.id: p for p in build_sources(self.cfg, self.data)}

    def test_cot_reasoning_ends_with_the_gold_sql(self) -> None:
        records, _ = synth_cot(self.cfg, self.data, self.ledger)
        for rec in records:
            final = rec.reasoning.split("Final SQL:")[-1].strip()
            self.assertEqual(normalize_sql(final), normalize_sql(rec.sql))

    def test_cot_validation_rejects_wrong_final_sql(self) -> None:
        plan = build_sources(self.cfg, self.data)[0]
        bad = "1. Read the table.\n2. Filter rows.\nFinal SQL: SELECT 1 FROM nowhere"
        ok, reason = validate_trace(bad, plan)
        self.assertFalse(ok)
        self.assertEqual(reason, "final_sql_mismatch")

    def test_cot_validation_rejects_missing_final_sql(self) -> None:
        plan = build_sources(self.cfg, self.data)[0]
        ok, reason = validate_trace("1. Read the book table and filter it.", plan)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_final_sql")

    def test_cot_validation_rejects_empty(self) -> None:
        plan = build_sources(self.cfg, self.data)[0]
        self.assertEqual(validate_trace("", plan), (False, "empty_trace"))
        self.assertEqual(validate_trace("   ", plan), (False, "empty_trace"))

    def test_cot_validation_rejects_hallucinated_table(self) -> None:
        plan = build_sources(self.cfg, self.data)[0]
        # a table name used directly in table position, not in the schema
        text = ("1. Read from author and join publisher on id. "
                "2. Filter the rows to the ones the question asks about.\n"
                "Final SQL: " + plan.gold_sql)
        ok, reason = validate_trace(text, plan)
        self.assertFalse(ok)
        self.assertIn("hallucinated_table:publisher", reason)

    def test_cot_validation_allows_prose_referring_to_the_schema(self) -> None:
        """Pointing at "the table" in prose is not a hallucinated relation."""
        plan = build_sources(self.cfg, self.data)[0]
        text = ("1. Read from author, keeping the name column and filtering the rows "
                "that table holds. "
                "2. Keep only the records the question asks about.\n"
                "Final SQL: " + plan.gold_sql)
        ok, reason = validate_trace(text, plan)
        self.assertTrue(ok, reason)

    def test_cot_tokens_hit_the_ledger(self) -> None:
        records, report = synth_cot(self.cfg, self.data, self.ledger)
        self.assertEqual(len(self.ledger), report.planned)
        self.assertEqual(self.ledger.spent(SYNTH_STAGE), report.total_tokens)
        # records carry the tokens of *accepted* calls; a dropped trace still cost
        # tokens, which is why the report (not the records) is the authoritative
        # spend figure. Assert the relationship rather than equality.
        self.assertLessEqual(sum(r.total_tokens for r in records), report.total_tokens)
        if report.dropped_invalid == 0:
            self.assertEqual(sum(r.total_tokens for r in records),
                             report.total_tokens)


# --------------------------------------------------------------------------- #
# paraphrase
# --------------------------------------------------------------------------- #
class TestParaphrase(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.data = self._make_data("synth_para")
        self.cfg = make_cfg()
        self.ledger = TokenLedger(None)

    def test_paraphrase_records_preserve_meaning(self) -> None:
        records, report = synth_paraphrase(self.cfg, self.data, self.ledger)
        self.assertGreater(len(records), 0)
        plans = {p.example.id: p for p in build_sources(self.cfg, self.data)}
        from fcdata.providers.mockwriters import paraphrase_preserves
        from fcdata.synth.paraphrase import _same_surface
        for rec in records:
            original = plans[rec.source_id].question
            self.assertEqual(rec.kind, "paraphrase")
            self.assertEqual(rec.original_question, original)
            self.assertEqual(rec.sql, plans[rec.source_id].gold_sql)
            self.assertTrue(paraphrase_preserves(original, rec.question),
                            f"drift: {rec.question!r}")
            # the question must be rewritten, but a conversational frame is a
            # legitimate rewrite even though it normalizes to the same content
            self.assertFalse(_same_surface(rec.question, original),
                             f"not a rewrite: {rec.question!r}")
            self.assertGreater(rec.total_tokens, 0)

    def test_strategies_are_distinct_per_source(self) -> None:
        strategies = pick_strategies("ex0", 3)
        self.assertEqual(len(strategies), 3)
        self.assertEqual(len(set(strategies)), 3)
        self.assertTrue(all(s in SAFE_STRATEGIES for s in strategies))
        self.assertEqual(strategies, pick_strategies("ex0", 3))

    def test_strategy_rotation_differs_between_sources(self) -> None:
        starts = {pick_strategies(f"ex{i}", 1)[0] for i in range(6)}
        self.assertGreater(len(starts), 1)

    def test_clean_rewrite_strips_model_wrapping(self) -> None:
        self.assertEqual(clean_rewrite("```\nList the books.\n```"), "List the books.")
        self.assertEqual(clean_rewrite('"List the books."'), "List the books.")
        self.assertEqual(clean_rewrite("Rewritten question: List the books."),
                         "List the books.")
        self.assertEqual(clean_rewrite(""), "")

    def test_paraphrase_dedupes_within_a_source(self) -> None:
        """No two records for one source may carry the same normalized question."""
        cfg = make_cfg(paraphrase_per_example=4)
        records, report = synth_paraphrase(cfg, self.data, TokenLedger(None))
        seen: set[tuple[str, str]] = set()
        for rec in records:
            key = (rec.source_id, normalize_question(rec.question))
            self.assertNotIn(key, seen, f"duplicate paraphrase: {rec.question!r}")
            seen.add(key)
        # drops are allowed (that is the point of dedup), but they must be counted
        self.assertGreaterEqual(report.dropped_duplicate + report.dropped_invalid, 0)


# --------------------------------------------------------------------------- #
# end-to-end stage
# --------------------------------------------------------------------------- #
class TestSynthStage(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.data = self._make_data("synth_stage")
        self.cfg = make_cfg()

    def test_build_synth_records_contract(self) -> None:
        ledger = TokenLedger(None)
        records = build_synth_records(self.cfg, self.data, ledger)
        self.assertGreater(len(records), 0)
        kinds = {r.kind for r in records}
        self.assertEqual(kinds, set(KIND_ORDER))
        for rec in records:
            self.assertIsInstance(rec, SynthRecord)
            self.assertTrue(rec.id)
            self.assertTrue(rec.source_id)
            self.assertEqual(rec.db_id, "library")
            self.assertTrue(rec.sql.strip())
            self.assertGreater(rec.prompt_tokens, 0)
            self.assertGreater(rec.completion_tokens, 0)
            # difficulty is inherited from the Spider example; simple fixture
            # queries score 0.0, so assert the field is *populated* (>= 0) and
            # that it matches the source rather than assuming a positive value.
            source = next(e for e in self.data.train if e.id == rec.source_id)
            self.assertEqual(rec.difficulty, source.difficulty)
            # every gold SQL must still be schema-valid
            db = self.data.databases[rec.db_id]
            self.assertTrue(analyze_sql(rec.sql, db).ok, rec.sql)

    def test_kind_specific_payloads(self) -> None:
        records = build_synth_records(self.cfg, self.data, TokenLedger(None))
        for rec in records:
            if rec.kind == "cot":
                self.assertTrue(rec.reasoning)
                self.assertEqual(rec.bad_sql, "")
            elif rec.kind == "paraphrase":
                self.assertTrue(rec.original_question)
                self.assertEqual(rec.bad_sql, "")
            elif rec.kind == "hard_negative":
                self.assertTrue(rec.bad_sql)
                self.assertTrue(rec.correction_note)
                self.assertIn(rec.error_kind, ERROR_KINDS)
                self.assertNotEqual(normalize_sql(rec.bad_sql),
                                    normalize_sql(rec.sql))

    def test_records_are_sorted_for_stable_diffs(self) -> None:
        records = build_synth_records(self.cfg, self.data, TokenLedger(None))
        keys = [(r.kind, str(r.source_id), r.id) for r in records]
        self.assertEqual(keys, sorted(keys))

    def test_deterministic_across_runs(self) -> None:
        first = build_synth_records(self.cfg, self.data, TokenLedger(None))
        second = build_synth_records(self.cfg, self.data, TokenLedger(None))
        self.assertEqual([r.id for r in first], [r.id for r in second])
        self.assertEqual([r.question for r in first], [r.question for r in second])
        self.assertEqual([r.reasoning for r in first], [r.reasoning for r in second])
        self.assertEqual([r.bad_sql for r in first], [r.bad_sql for r in second])
        self.assertEqual([r.total_tokens for r in first],
                         [r.total_tokens for r in second])

    def test_summary_accounts_for_every_token(self) -> None:
        ledger = TokenLedger(None)
        records, summary = synthesize(self.cfg, self.data, ledger, verbose=False)
        self.assertEqual(summary["records"], len(records))
        self.assertEqual(summary["total_tokens"], ledger.spent(SYNTH_STAGE))
        self.assertEqual(summary["by_kind"]["cot"],
                         sum(1 for r in records if r.kind == "cot"))
        # every planned job is accounted for exactly once:
        #   ok + failed + skipped_budget + dropped_invalid + dropped_duplicate
        #   == planned
        for kind_report in summary["kinds"]:
            planned = kind_report["planned"]
            accounted = (kind_report["ok"] + kind_report["failed"]
                         + kind_report["skipped_budget"]
                         + kind_report["dropped_invalid"]
                         + kind_report["dropped_duplicate"])
            self.assertEqual(planned, accounted,
                             f"{kind_report['kind']}: {kind_report}")

    def test_kind_restriction_for_ablation(self) -> None:
        records, summary = synthesize(self.cfg, self.data, TokenLedger(None),
                                      kinds=("cot",))
        self.assertEqual({r.kind for r in records}, {"cot"})
        self.assertEqual(len(summary["kinds"]), 1)

    def test_budget_cap_truncates_without_overrunning(self) -> None:
        cfg = make_cfg(max_total_tokens=6000)
        ledger = TokenLedger(None)
        records, summary = synthesize(cfg, self.data, ledger)
        self.assertLessEqual(ledger.spent(SYNTH_STAGE), cfg.synth.max_total_tokens)
        skipped = sum(k["skipped_budget"] for k in summary["kinds"])
        self.assertGreater(skipped, 0)
        self.assertGreater(len(records), 0)

    def test_zero_per_example_produces_nothing_for_that_kind(self) -> None:
        cfg = make_cfg(cot_per_example=0, paraphrase_per_example=0,
                       hard_negative_per_example=0)
        records, summary = synthesize(cfg, self.data, TokenLedger(None))
        self.assertEqual(records, [])
        self.assertEqual(summary["records"], 0)

    def test_max_source_examples_limits_the_corpus(self) -> None:
        cfg = make_cfg(max_source_examples=2)
        records, _ = synthesize(cfg, self.data, TokenLedger(None))
        self.assertLessEqual(len({r.source_id for r in records}), 2)

    def test_difficulty_stratification_concentrates_on_hard_examples(self) -> None:
        """With stratification on, covered examples must skew harder than average."""
        cfg = make_cfg(hard_negative_per_example=2,
                       hard_negative_use_difficulty=True,
                       hard_negative_hard_fraction=0.8)
        records, _ = synthesize(cfg, self.data, TokenLedger(None),
                                kinds=("hard_negative",))
        plans = build_sources(cfg, self.data)
        by_id = {p.example.id: p for p in plans}
        covered = [by_id[r.source_id].difficulty for r in records]
        all_diff = [p.difficulty for p in plans]
        self.assertGreater(len(covered), 0)
        self.assertGreater(sum(covered) / len(covered),
                           sum(all_diff) / len(all_diff))

    def test_uniform_negatives_when_stratification_off(self) -> None:
        cfg = make_cfg(hard_negative_per_example=1,
                       hard_negative_use_difficulty=False)
        records, _ = synthesize(cfg, self.data, TokenLedger(None),
                                kinds=("hard_negative",))
        covered = {r.source_id for r in records}
        self.assertEqual(len(covered), len(self.data.train))

    def test_hard_negatives_pass_execution_verification(self) -> None:
        """End-to-end: every emitted negative must really be wrong on sqlite."""
        records = build_synth_records(self.cfg, self.data, TokenLedger(None))
        negatives = [r for r in records if r.kind == "hard_negative"]
        self.assertGreater(len(negatives), 0)
        for rec in negatives:
            db_path = self.data.db_path(rec.db_id)
            verdict = verify_negative(db_path, rec.sql, rec.bad_sql)
            self.assertFalse(verdict["identical"], rec.bad_sql)
            self.assertTrue(verdict["exec_differs"],
                            f"equivalent negative: {rec.bad_sql} vs {rec.sql}")

    def test_example_with_no_viable_error_kind_is_skipped(self) -> None:
        """A query with nothing to corrupt is skipped, not faked.

        The table has a *single* column, so ``wrong_column`` has no sibling to
        swap in; ``wrong_table`` finds no alternative table; and the query has no
        aggregate, join, grouping, ordering, distinct or literal to corrupt. Every
        kind is therefore a no-op, and the example must be skipped with a counted
        reason instead of emitting ``bad_sql == gold``.
        """
        d = scratch_dir("synth_noop")
        self._dirs.append(d)
        single = {
            "db_id": "notes",
            "table_names_original": ["note"],
            "column_names_original": [[-1, "*"], [0, "id"]],
            "column_types": ["text", "number"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
        examples = [{"id": "noop", "db_id": "notes",
                     "question": "List the note ids.",
                     "query": "SELECT id FROM note"}]
        db = Database.from_spider_json(single)
        db.rows = {"note": [{"id": 1}]}
        db_dir = d / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        build_sqlite(db, db_dir / "notes.sqlite")
        (d / "tables.json").write_text(json.dumps([single]), encoding="utf-8")
        (d / "train.json").write_text(json.dumps(examples), encoding="utf-8")
        (d / "dev.json").write_text("[]", encoding="utf-8")
        data = load_spider(d)
        cfg = make_cfg(hard_negative_per_example=1, cot_per_example=0,
                       paraphrase_per_example=0)
        records, summary = synthesize(cfg, data, TokenLedger(None))
        negatives = [r for r in records if r.kind == "hard_negative"]
        self.assertEqual(negatives, [])
        report = next(k for k in summary["kinds"] if k["kind"] == "hard_negative")
        self.assertEqual(report["planned"], 0)
        self.assertEqual(report["examples_skipped"], 1)
        self.assertEqual(report["examples_covered"], 0)
        self.assertIn("no_viable_error_kind", report["errors"])

    def test_synthesis_writes_nothing_to_the_dataset(self) -> None:
        """The stage must not mutate SpiderData (it is shared across stages)."""
        before = [(e.id, e.question, e.sql) for e in self.data.train]
        build_synth_records(self.cfg, self.data, TokenLedger(None))
        after = [(e.id, e.question, e.sql) for e in self.data.train]
        self.assertEqual(before, after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
