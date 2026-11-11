"""Tests for partial-credit metrics (the learning signal behind 0% exec accuracy).

WHY THESE TESTS MATTER
----------------------
Strict execution accuracy is all-or-nothing, so a working model can legitimately
score 0.0% exec_acc while having learned most of the task. The partial-credit
metrics exist to show that signal, which means they must be *provably* correct
and *provably* discriminating: if they were also all-zero they would be
decoration rather than evidence.

So the tests below check three things:
1. the metrics are correct on hand-computed cases (exact expected values);
2. they *discriminate* -- a near-miss scores far above a total miss;
3. they are computed from the real SQL text via the project's own tokenizer and
   analyzer, with no fabricated values.
"""

from __future__ import annotations

import unittest

from fcdata.dbexec import build_sqlite, evaluate_pair
from fcdata.eval.metrics import summarize
from fcdata.eval.partial import (
    CLAUSE_FLAGS,
    PARTIAL_FAILURE_TAXONOMY,
    aggregate_partial,
    classify_partial_failure,
    clause_flags,
    clause_match,
    column_f1,
    edit_distance,
    prefix_match_ratio,
    score_prediction,
    token_accuracy,
)
from fcdata.records import Prediction, new_id
from fcdata.schema import Column, Database, Table
from fcdata.sqlutils import analyze_sql
from fcdata.config import scratch_dir
import shutil


def _db() -> Database:
    dept = Table(index=0, name="department", columns=[
        Column(0, "dept_id", "number", True),
        Column(0, "dept_name", "text"),
    ])
    emp = Table(index=1, name="employee", columns=[
        Column(1, "emp_id", "number", True),
        Column(1, "emp_name", "text"),
        Column(1, "dept_id", "number"),
        Column(1, "salary", "number"),
    ])
    return Database(db_id="tiny", tables=[dept, emp])


GOLD = "SELECT COUNT(*) FROM employee"


def _pred(pred_sql: str, gold: str = GOLD, *, executable: bool = False,
          exec_match: bool = False, db_id: str = "tiny") -> Prediction:
    return Prediction(id=new_id("p"), example_id="e1", db_id=db_id,
                      question="q", gold_sql=gold, pred_sql=pred_sql,
                      executable=executable, exec_match=exec_match,
                      exact_match=(pred_sql == gold))


# --------------------------------------------------------------------------- #
# token similarity
# --------------------------------------------------------------------------- #
class TestTokenSimilarity(unittest.TestCase):
    """Token edit similarity must be exact and monotone."""

    def test_identical_is_one(self) -> None:
        self.assertEqual(token_accuracy(GOLD, GOLD), 1.0)

    def test_completely_different_is_near_zero(self) -> None:
        self.assertLess(token_accuracy("SELECT 1", "SELECT COUNT(*) FROM employee"),
                        0.5)

    def test_one_token_wrong_is_high(self) -> None:
        """A single wrong identifier on a long query stays high: that is the point."""
        gold = "SELECT emp_name FROM employee WHERE dept_id = 1 AND salary > 100"
        pred = "SELECT emp_name FROM employee WHERE dept_id = 2 AND salary > 100"
        score = token_accuracy(pred, gold)
        self.assertGreater(score, 0.85)
        self.assertLess(score, 1.0)

    def test_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(token_accuracy("select count(*) from employee",
                                        "SELECT COUNT(*) FROM employee"), 1.0)

    def test_empty_cases(self) -> None:
        self.assertEqual(token_accuracy("", ""), 1.0)
        self.assertEqual(token_accuracy("", GOLD), 0.0)
        self.assertEqual(token_accuracy(GOLD, ""), 0.0)

    def test_edit_distance_symmetric(self) -> None:
        a = ["select", "count", "(", "*", ")"]
        b = ["select", "sum", "(", "*", ")"]
        self.assertEqual(edit_distance(a, b), edit_distance(b, a))
        self.assertEqual(edit_distance(a, a), 0)
        self.assertEqual(edit_distance([], a), len(a))

    def test_prefix_match(self) -> None:
        gold = "SELECT emp_name FROM employee"
        self.assertEqual(prefix_match_ratio(gold, gold), 1.0)
        # diverges at the second token -> 1 of 4 gold tokens matched
        self.assertAlmostEqual(
            prefix_match_ratio("SELECT salary FROM employee", gold), 1 / 4)

    def test_prefix_match_empty_gold(self) -> None:
        self.assertEqual(prefix_match_ratio("", ""), 1.0)
        self.assertEqual(prefix_match_ratio("SELECT 1", ""), 0.0)


# --------------------------------------------------------------------------- #
# schema-linking metrics
# --------------------------------------------------------------------------- #
class TestSchemaLinking(unittest.TestCase):
    """Table-set and column-F1 are the interpretable schema-grounding signals."""

    def setUp(self) -> None:
        self.db = _db()

    def test_table_set_match(self) -> None:
        gold = "SELECT COUNT(*) FROM employee"
        good = _pred("SELECT COUNT(emp_id) FROM employee", gold)
        bad = _pred("SELECT COUNT(*) FROM department", gold)
        self.assertTrue(score_prediction(good, self.db).table_set_match)
        self.assertFalse(score_prediction(bad, self.db).table_set_match)

    def test_hallucinated_table_is_detected(self) -> None:
        pred = _pred("SELECT COUNT(*) FROM nonexistent_table")
        score = score_prediction(pred, self.db)
        self.assertEqual(score.failure, "wrong_table")
        self.assertFalse(score.table_set_match)

    def test_hallucinated_column_is_detected(self) -> None:
        pred = _pred("SELECT bogus_column FROM employee")
        score = score_prediction(pred, self.db)
        self.assertEqual(score.failure, "wrong_column")

    def test_column_f1_perfect(self) -> None:
        a = analyze_sql("SELECT emp_name FROM employee", self.db)
        b = analyze_sql("SELECT emp_name FROM employee", self.db)
        self.assertEqual(column_f1(a, b), (1.0, 1.0, 1.0))

    def test_column_f1_partial(self) -> None:
        pred = analyze_sql("SELECT emp_name, salary FROM employee", self.db)
        gold = analyze_sql("SELECT emp_name FROM employee", self.db)
        p, r, f = column_f1(pred, gold)
        self.assertAlmostEqual(p, 0.5)     # 1 of 2 predicted columns correct
        self.assertAlmostEqual(r, 1.0)     # found the one gold column
        self.assertGreater(f, 0.5)
        self.assertLess(f, 1.0)

    def test_column_f1_zero_when_disjoint(self) -> None:
        pred = analyze_sql("SELECT salary FROM employee", self.db)
        gold = analyze_sql("SELECT emp_name FROM employee", self.db)
        p, r, f = column_f1(pred, gold)
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_clause_flags_and_match(self) -> None:
        a = analyze_sql("SELECT dept_id, COUNT(*) FROM employee GROUP BY dept_id",
                        self.db)
        flags = clause_flags(a)
        self.assertEqual(set(flags), set(CLAUSE_FLAGS))
        self.assertTrue(flags["has_aggregate"])
        self.assertTrue(flags["has_group_by"])
        self.assertFalse(flags["has_join"])

        b = analyze_sql("SELECT COUNT(*) FROM employee", self.db)
        m = clause_match(b, a)
        # only clauses the GOLD uses are scored
        self.assertEqual(set(m), {"has_aggregate", "has_group_by"})
        self.assertTrue(m["has_aggregate"])   # both aggregate
        self.assertFalse(m["has_group_by"])   # gold groups, pred does not

    def test_clause_match_ignores_absent_gold_clauses(self) -> None:
        """A clause the gold never uses must not award free agreement."""
        gold = analyze_sql("SELECT COUNT(*) FROM employee", self.db)  # no join
        junk = analyze_sql("SELECT 1", self.db)
        m = clause_match(junk, gold)
        self.assertNotIn("has_join", m)
        self.assertEqual(set(m), {"has_aggregate"})
        self.assertFalse(m["has_aggregate"])

    def test_no_db_still_scores_text(self) -> None:
        """Without a schema we can still measure token/skeleton similarity."""
        pred = _pred("SELECT COUNT(*) FROM employee")
        score = score_prediction(pred, None)
        self.assertGreater(score.token_acc, 0.9)
        self.assertTrue(score.skeleton_match)


# --------------------------------------------------------------------------- #
# failure taxonomy
# --------------------------------------------------------------------------- #
class TestPartialTaxonomy(unittest.TestCase):
    """The taxonomy must name the *actionable* error, in a stable priority order."""

    def setUp(self) -> None:
        self.db = _db()

    def test_taxonomy_is_complete_and_ordered(self) -> None:
        self.assertIn("unparseable", PARTIAL_FAILURE_TAXONOMY)
        self.assertIn("wrong_table", PARTIAL_FAILURE_TAXONOMY)
        self.assertIn("missing_join", PARTIAL_FAILURE_TAXONOMY)

    def test_empty_prediction_is_unparseable(self) -> None:
        self.assertEqual(score_prediction(_pred(""), self.db).failure, "unparseable")

    def test_missing_join(self) -> None:
        gold = ("SELECT T1.emp_name FROM department AS T1 JOIN employee AS T2 "
                "ON T1.dept_id = T2.dept_id")
        pred = _pred("SELECT emp_name FROM employee", gold, executable=True)
        score = score_prediction(pred, self.db)
        self.assertEqual(score.failure, "missing_join")

    def test_extra_join(self) -> None:
        """Prediction joins two tables where gold needs only one.

        The JOIN condition must be schema-valid, otherwise the schema-error check
        correctly outranks the structural one and reports wrong_column instead.
        """
        gold = "SELECT emp_name FROM employee"
        pred = _pred("SELECT T2.emp_name FROM department AS T1 JOIN employee AS T2 "
                     "ON T1.dept_id = T2.dept_id", gold, executable=True)
        self.assertEqual(score_prediction(pred, self.db).failure, "extra_join")

    def test_missing_group_by(self) -> None:
        gold = "SELECT dept_id, COUNT(*) FROM employee GROUP BY dept_id"
        pred = _pred("SELECT dept_id, COUNT(*) FROM employee", gold, executable=True)
        self.assertEqual(score_prediction(pred, self.db).failure, "missing_group_by")

    def test_wrong_aggregate(self) -> None:
        """COUNT vs SUM: same shape, different aggregate function."""
        gold = "SELECT COUNT(*) FROM employee"
        pred = _pred("SELECT SUM(salary) FROM employee", gold, executable=True)
        self.assertEqual(score_prediction(pred, self.db).failure, "wrong_aggregate")

    def test_missing_aggregate(self) -> None:
        """Gold aggregates, prediction selects a bare column."""
        gold = "SELECT COUNT(*) FROM employee"
        pred = _pred("SELECT emp_name FROM employee", gold, executable=True)
        self.assertEqual(score_prediction(pred, self.db).failure, "wrong_aggregate")

    def test_success_has_empty_failure(self) -> None:
        pred = _pred(GOLD, GOLD, executable=True, exec_match=True)
        self.assertEqual(score_prediction(pred, self.db).failure, "")


# --------------------------------------------------------------------------- #
# aggregation + discrimination
# --------------------------------------------------------------------------- #
class TestAggregatePartial(unittest.TestCase):
    """Aggregation must be a real mean, and must discriminate quality levels."""

    def setUp(self) -> None:
        self.db = _db()

    def _scores(self, preds: list[str], gold: str = GOLD) -> list:
        return [score_prediction(_pred(p, gold), self.db) for p in preds]

    def test_perfect_set_scores_one(self) -> None:
        """A query naming a real column scores full schema-linking credit."""
        gold = "SELECT emp_name FROM employee"
        agg = aggregate_partial(self._scores([gold, gold], gold))
        self.assertEqual(agg["token_acc"], 1.0)
        self.assertEqual(agg["skeleton_match"], 1.0)
        self.assertEqual(agg["table_set_match"], 1.0)
        self.assertEqual(agg["column_f1"], 1.0)

    def test_column_free_query_scores_zero_not_one(self) -> None:
        """``SELECT COUNT(*)`` names no column, so it earns no column credit.

        Awarding 1.0 here would hand perfect schema-linking credit to a query that
        referenced no schema object at all.
        """
        agg = aggregate_partial(self._scores([GOLD, GOLD]))
        self.assertEqual(agg["column_f1"], 0.0)

    def test_empty_set_is_safe(self) -> None:
        agg = aggregate_partial([])
        self.assertEqual(agg["n"], 0)
        self.assertEqual(agg["token_acc"], 0.0)
        self.assertEqual(set(agg["failure_taxonomy"]), set(PARTIAL_FAILURE_TAXONOMY))

    def test_near_miss_scores_above_total_miss(self) -> None:
        """The whole point: partial credit must separate close from hopeless."""
        gold = ("SELECT dept_id, COUNT(*) FROM employee GROUP BY dept_id "
                "ORDER BY COUNT(*) DESC")
        near = aggregate_partial(self._scores([gold], gold))
        far = aggregate_partial(self._scores(["SELECT 1"], gold))
        self.assertGreater(near["token_acc"], far["token_acc"])
        self.assertGreater(near["table_set_match"], far["table_set_match"])
        self.assertGreater(near["clause_match_mean"], far["clause_match_mean"])
        # the near miss keeps the query skeleton, the total miss does not
        self.assertGreater(near["skeleton_match"], far["skeleton_match"])

    def test_all_zero_exec_acc_can_still_have_learning_signal(self) -> None:
        """The exact scenario the Lead flagged: exec 0%, partial > 0."""
        preds = [
            _pred("SELECT COUNT(emp_id) FROM employee", GOLD),
            _pred("SELECT COUNT(*) FROM employee", GOLD),
        ]
        m = summarize(preds, n_boot=100, databases={"tiny": self.db})
        self.assertEqual(m["exec_acc"], 0.0)
        self.assertGreater(m["partial"]["token_acc"], 0.5)
        self.assertGreater(m["partial"]["table_set_match"], 0.0)
        self.assertGreater(m["partial"]["clause_match_mean"], 0.0)

    def test_hallucination_counter(self) -> None:
        """Count predictions that reference tables absent from the schema."""
        gold = "SELECT COUNT(*) FROM employee"
        # a real-but-wrong table is not a hallucination (it exists in the schema)
        agg = aggregate_partial(
            [score_prediction(_pred(s, gold), self.db) for s in
             ["SELECT COUNT(*) FROM employee", "SELECT COUNT(*) FROM department"]])
        self.assertEqual(agg["examples_with_table_hallucination"], 0)
        # a table that does not exist at all IS a hallucination
        agg2 = aggregate_partial(
            [score_prediction(_pred("SELECT COUNT(*) FROM ghost_table", gold), self.db)])
        self.assertEqual(agg2["examples_with_table_hallucination"], 1)


# --------------------------------------------------------------------------- #
# integration: strict metrics unchanged, partial added, real sqlite
# --------------------------------------------------------------------------- #
class TestSummarizeIntegration(unittest.TestCase):
    """Strict exec_acc semantics must not change; partial block must appear."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("partial_integration")
        cls.db = _db()
        cls.db_path = build_sqlite(cls.db, cls.dir / "tiny.sqlite")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _scored_pred(self, pred_sql: str, gold: str) -> Prediction:
        """Build a Prediction scored by REAL sqlite execution."""
        result = evaluate_pair(self.db_path, pred_sql, gold)
        return Prediction(id=new_id("p"), example_id="e1", db_id="tiny",
                          question="q", gold_sql=gold, pred_sql=pred_sql,
                          executable=bool(result["executable"]),
                          exec_match=bool(result["exec_match"]),
                          exact_match=bool(result["exact_match"]),
                          reason=str(result["reason"]))

    def test_strict_semantics_unchanged(self) -> None:
        m = summarize([self._scored_pred(GOLD, GOLD)], n_boot=50,
                      databases={"tiny": self.db})
        self.assertEqual(m["exec_acc"], 1.0)
        self.assertEqual(m["exact_match"], 1.0)
        self.assertEqual(m["executable_rate"], 1.0)

    def test_partial_block_present_at_top_and_in_slices(self) -> None:
        preds = [
            self._scored_pred("SELECT COUNT(emp_id) FROM employee", GOLD),
            self._scored_pred("SELECT COUNT(*) FROM employee", GOLD),
        ]
        m = summarize(preds, n_boot=100, databases={"tiny": self.db})
        for key in ("token_acc", "skeleton_match", "table_set_match",
                    "column_f1", "clause_match", "failure_taxonomy", "ci"):
            self.assertIn(key, m["partial"], f"partial missing {key}")
        for bucket in m["by_difficulty"].values():
            self.assertIn("token_acc", bucket)
            self.assertIn("table_set_match", bucket)
        for group in m["by_db"].values():
            self.assertIn("token_acc", group)

    def test_partial_cis_are_reported(self) -> None:
        preds = [self._scored_pred("SELECT COUNT(*) FROM employee", GOLD)
                 for _ in range(4)]
        m = summarize(preds, n_boot=200, databases={"tiny": self.db})
        ci = m["partial"]["ci"]
        for metric in ("token_acc", "skeleton_match", "table_set_match",
                       "column_f1", "clause_match_mean"):
            lo, hi = ci[metric]
            self.assertLessEqual(lo, hi)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_real_execution_drives_strict_not_partial(self) -> None:
        """A valid-but-wrong result set is exec_match=False but high token_acc."""
        pred = self._scored_pred("SELECT SUM(salary) FROM employee", GOLD)
        self.assertFalse(pred.exec_match)
        self.assertTrue(pred.executable)
        m = summarize([pred], n_boot=50, databases={"tiny": self.db})
        self.assertEqual(m["exec_acc"], 0.0)
        self.assertGreater(m["partial"]["token_acc"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
