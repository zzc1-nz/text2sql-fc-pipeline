"""Tests for the Lead-owned core modules.

These cover the pieces every other stage depends on: token accounting, schema
modelling, SQL static analysis, real sqlite execution and result comparison, and
the fixture dataset. If these are wrong, every downstream metric is wrong, so
they are tested hardest.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from fcdata.config import (Config, apply_overrides, ensure_dirs, load_config,
                           scratch_dir)
from fcdata.dbexec import (build_sqlite, evaluate_pair, execute_sql,
                           introspect_sqlite, is_safe_to_execute, result_match)
from fcdata.records import (Example, Prediction, SynthRecord, ToolSchema,
                            read_jsonl, write_jsonl)
from fcdata.schema import Database
from fcdata.spider import load_spider
from fcdata.sqlutils import (analyze_sql, difficulty_score, extract_sql,
                             inject_error, normalize_sql, sql_skeleton,
                             tokenize)
from fcdata.tokens import (TokenLedger, UsageRecord, estimate_messages_tokens,
                           estimate_tokens, summarize_ledger)

REPO = Path(__file__).resolve().parent.parent


def tiny_db() -> Database:
    return Database.from_spider_json({
        "db_id": "tiny",
        "table_names_original": ["author", "book"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "name"],
                                  [1, "id"], [1, "author_id"], [1, "title"],
                                  [1, "year"]],
        "column_types": ["text", "number", "text", "number", "number", "text", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    })


class ScratchMixin:
    """Provides a writable scratch dir (tempfile.mkdtemp is broken in this sandbox)."""

    _dirs: list[Path] = []

    def scratch(self, name: str = "t") -> Path:
        d = scratch_dir(name)
        self._dirs.append(d)
        return d

    def tearDown(self) -> None:  # noqa: D401 - unittest hook
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._dirs.clear()


# --------------------------------------------------------------------------- #
class TestTokens(unittest.TestCase):
    def test_estimate_tokens_monotonic(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertLess(estimate_tokens("hi"), estimate_tokens("hi there " * 20))
        self.assertGreater(estimate_tokens("你好世界"), 0)

    def test_estimate_messages_tokens(self) -> None:
        msgs = [{"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "SELECT * FROM t"}]
        n = estimate_messages_tokens(msgs)
        self.assertGreater(n, estimate_tokens("you are helpful"))

    def test_ledger_aggregation_and_cost(self) -> None:
        d = scratch_dir("ledger")
        try:
            path = d / "l.jsonl"
            led = TokenLedger(path, prices={"cheap": (1.0, 2.0)}, reset_file=True)
            led.add("synth", "cheap", "m", 1_000_000, 0)
            led.add("synth", "cheap", "m", 0, 1_000_000)
            led.add("validate", "judge", "j", 500, 500, cached=True)
            agg = led.aggregate()
            self.assertEqual(agg["synth"]["total_tokens"], 2_000_000)
            self.assertAlmostEqual(agg["synth"]["cost_usd"], 3.0, places=6)
            self.assertEqual(agg["validate"]["cached_calls"], 1)
            self.assertEqual(len(led), 3)
            # persisted ledger reloads to the same totals
            summary = summarize_ledger(path)
            self.assertEqual(summary["totals"]["total_tokens"],
                             led.totals()["total_tokens"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_ledger_budget_tracking(self) -> None:
        led = TokenLedger(None)
        led.add("synth", "cheap", "m", 100, 50)
        self.assertEqual(led.spent("synth"), 150)
        self.assertEqual(led.spent(), 150)
        self.assertEqual(led.spent("validate"), 0)


# --------------------------------------------------------------------------- #
class TestSchema(unittest.TestCase):
    def test_from_spider_json_roundtrip(self) -> None:
        db = tiny_db()
        self.assertEqual([t.name for t in db.tables], ["author", "book"])
        self.assertEqual(len(db.foreign_keys), 1)
        fk = db.foreign_keys[0]
        self.assertEqual(db.tables[fk.src_table].name, "book")
        self.assertEqual(db.tables[fk.dst_table].name, "author")

    def test_render_ddl(self) -> None:
        ddl = tiny_db().render_ddl()
        self.assertIn("CREATE TABLE author", ddl)
        self.assertIn("PRIMARY KEY", ddl)
        self.assertIn("FOREIGN KEY", ddl)

    def test_to_spider_json_roundtrip(self) -> None:
        db = tiny_db()
        again = Database.from_spider_json(db.to_spider_json())
        self.assertEqual(len(again.tables), len(db.tables))
        self.assertEqual([c.name for t in again.tables for c in t.columns],
                         [c.name for t in db.tables for c in t.columns])


# --------------------------------------------------------------------------- #
class TestSqlUtils(unittest.TestCase):
    def test_extract_sql_from_fence(self) -> None:
        text = "Sure! Here you go:\n```sql\nSELECT a FROM t;\n```\nHope that helps."
        self.assertEqual(extract_sql(text), "SELECT a FROM t")

    def test_extract_sql_strips_prose(self) -> None:
        self.assertEqual(extract_sql("The query is: SELECT x FROM y"),
                         "SELECT x FROM y")

    def test_extract_sql_semicolon_in_literal(self) -> None:
        sql = extract_sql("SELECT * FROM t WHERE name = 'a;b'; SELECT 1")
        self.assertIn("'a;b'", sql)
        self.assertNotIn("SELECT 1", sql)

    def test_tokenize(self) -> None:
        toks = tokenize("SELECT a.b FROM t WHERE x >= 10 AND y != 'z'")
        kinds = [t.kind for t in toks]
        self.assertIn("keyword", kinds)
        self.assertIn("string", kinds)
        values = [t.value for t in toks]
        self.assertIn(">=", values)
        self.assertIn("!=", values)

    def test_analyze_valid_sql(self) -> None:
        db = tiny_db()
        an = analyze_sql("SELECT name FROM author WHERE id = 1", db)
        self.assertTrue(an.ok, msg=[str(i) for i in an.issues])
        self.assertEqual(an.tables, ["author"])

    def test_analyze_join(self) -> None:
        db = tiny_db()
        an = analyze_sql(
            "SELECT a.name, b.title FROM author AS a JOIN book AS b "
            "ON a.id = b.author_id WHERE b.year > 2000", db)
        self.assertTrue(an.ok, msg=[str(i) for i in an.issues])
        self.assertTrue(an.has_join)
        self.assertEqual(sorted(an.tables), ["author", "book"])

    def test_analyze_unknown_table(self) -> None:
        db = tiny_db()
        an = analyze_sql("SELECT name FROM nonexistent", db)
        self.assertFalse(an.ok)
        self.assertIn("unknown_table", [i.code for i in an.issues])

    def test_analyze_unknown_column(self) -> None:
        db = tiny_db()
        an = analyze_sql("SELECT bogus_column FROM author", db)
        self.assertFalse(an.ok)
        self.assertIn("unknown_column", [i.code for i in an.issues])

    def test_analyze_unbalanced_paren(self) -> None:
        an = analyze_sql("SELECT COUNT(* FROM author")
        self.assertFalse(an.ok)
        self.assertIn("unbalanced_paren", [i.code for i in an.issues])

    def test_analyze_rejects_destructive(self) -> None:
        an = analyze_sql("SELECT 1 FROM t; DROP TABLE t")
        codes = [i.code for i in an.issues]
        self.assertTrue("destructive_statement" in codes or "not_select" in codes)

    def test_normalize_sql(self) -> None:
        a = normalize_sql("SELECT  A.Name  AS n FROM author;")
        b = normalize_sql("select a.name from author")
        self.assertEqual(a, b)

    def test_sql_skeleton_abstracts_literals(self) -> None:
        a = sql_skeleton("SELECT name FROM author WHERE id = 1")
        b = sql_skeleton("SELECT name FROM author WHERE id = 42")
        self.assertEqual(a, b)

    def test_difficulty_orders_correctly(self) -> None:
        simple = difficulty_score("SELECT name FROM author")
        hard = difficulty_score(
            "SELECT a.name, COUNT(*) FROM author AS a JOIN book AS b "
            "ON a.id = b.author_id WHERE b.year > 2000 GROUP BY a.name "
            "HAVING COUNT(*) > 2 ORDER BY COUNT(*) DESC LIMIT 3")
        self.assertLess(simple, hard)
        self.assertLessEqual(hard, 1.0)
        self.assertGreaterEqual(simple, 0.0)

    def test_inject_error_changes_sql(self) -> None:
        db = tiny_db()
        gold = "SELECT COUNT(*) FROM book WHERE year > 2000"
        for kind in ("wrong_aggregate", "wrong_operator", "wrong_literal"):
            bad = inject_error(gold, db, kind)
            self.assertNotEqual(normalize_sql(bad), normalize_sql(gold),
                                msg=f"injection {kind} was a no-op")


# --------------------------------------------------------------------------- #
class TestDbExec(ScratchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.dir = self.scratch("dbexec")
        db = tiny_db()
        db.rows = {
            "author": [{"id": 1, "name": "Ann"}, {"id": 2, "name": "Bob"}],
            "book": [{"id": 10, "author_id": 1, "title": "A", "year": 2001},
                     {"id": 11, "author_id": 1, "title": "B", "year": 2005},
                     {"id": 12, "author_id": 2, "title": "C", "year": 1999}],
        }
        self.db = db
        self.path = build_sqlite(db, self.dir / "tiny.sqlite")

    def test_execute_ok(self) -> None:
        r = execute_sql(self.path, "SELECT name FROM author ORDER BY id")
        self.assertTrue(r.ok, r.error)
        self.assertEqual([row[0] for row in r.rows], ["Ann", "Bob"])

    def test_execute_error(self) -> None:
        r = execute_sql(self.path, "SELECT nope FROM author")
        self.assertFalse(r.ok)
        self.assertIn("no such column", r.error.lower())

    def test_execute_rejects_writes(self) -> None:
        for sql in ("DROP TABLE author", "DELETE FROM author", "UPDATE author SET name='x'"):
            self.assertFalse(is_safe_to_execute(sql), msg=sql)
            self.assertFalse(execute_sql(self.path, sql).ok, msg=sql)

    def test_result_match_ignores_row_order(self) -> None:
        gold = execute_sql(self.path, "SELECT name FROM author")
        pred = execute_sql(self.path, "SELECT name FROM author ORDER BY name DESC")
        self.assertTrue(result_match(pred, gold).exec_match)

    def test_result_match_detects_row_count(self) -> None:
        gold = execute_sql(self.path, "SELECT name FROM author")
        pred = execute_sql(self.path, "SELECT name FROM author WHERE id = 1")
        m = result_match(pred, gold)
        self.assertFalse(m.exec_match)
        self.assertIn("row count", m.reason)

    def test_result_match_numeric_equivalence(self) -> None:
        gold = execute_sql(self.path, "SELECT COUNT(*) FROM book")
        pred = execute_sql(self.path, "SELECT 3.0")
        self.assertTrue(result_match(pred, gold).exec_match)

    def test_evaluate_pair_exec_and_em(self) -> None:
        gold = "SELECT COUNT(*) FROM book"
        res = evaluate_pair(self.path, "SELECT COUNT(*) FROM book", gold)
        self.assertTrue(res["exec_match"])
        self.assertTrue(res["exact_match"])
        # different formulation, same result -> exec match but not exact match
        res2 = evaluate_pair(self.path, "SELECT COUNT(id) FROM book", gold)
        self.assertTrue(res2["exec_match"])
        self.assertFalse(res2["exact_match"])
        # broken sql
        res3 = evaluate_pair(self.path, "SELECT bad FROM book", gold)
        self.assertFalse(res3["exec_match"])
        self.assertFalse(res3["executable"])

    def test_introspect_roundtrip(self) -> None:
        again = introspect_sqlite(self.path, "tiny")
        self.assertEqual([t.name for t in again.tables], ["author", "book"])
        self.assertEqual(len(again.foreign_keys), 1)
        self.assertEqual(len(again.rows["book"]), 3)


# --------------------------------------------------------------------------- #
class TestFixtureDataset(unittest.TestCase):
    """The bundled fixture must load and every gold query must execute."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = REPO / "data" / "fixture"
        if not (cls.root / "tables.json").exists():
            raise unittest.SkipTest("fixture not built; run `python -m fcdata.build_fixture`")
        cls.data = load_spider(cls.root)

    def test_load(self) -> None:
        self.assertEqual(len(self.data.databases), 6)
        self.assertGreater(len(self.data.train), 50)
        self.assertGreater(len(self.data.dev), 10)

    def test_all_gold_sql_executes(self) -> None:
        bad = []
        for ex in self.data.train + self.data.dev:
            r = execute_sql(self.data.db_path(ex.db_id), ex.sql)
            if not r.ok:
                bad.append((ex.id, r.error))
        self.assertEqual(bad, [], msg=f"{len(bad)} gold queries failed: {bad[:5]}")

    def test_stats_shape(self) -> None:
        s = self.data.stats()
        for key in ("n_databases", "n_train", "n_dev", "avg_difficulty_train"):
            self.assertIn(key, s)


# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.train.backend, "reference")
        self.assertIn("cheap", cfg.providers)
        self.assertEqual(cfg.provider("judge").temperature, 0.0)

    def test_overrides(self) -> None:
        cfg = Config()
        apply_overrides(cfg, ["train.epochs=7", "synth.cot_per_example=3",
                              "providers.cheap.model=foo"])
        self.assertEqual(cfg.train.epochs, 7)
        self.assertEqual(cfg.synth.cot_per_example, 3)
        self.assertEqual(cfg.providers["cheap"].model, "foo")

    def test_override_bad_key(self) -> None:
        with self.assertRaises(KeyError):
            apply_overrides(Config(), ["train.not_a_field=1"])

    def test_load_yaml(self) -> None:
        cfg = load_config(REPO / "configs" / "demo.yaml", root=REPO)
        self.assertEqual(cfg.run_name, "demo")
        self.assertEqual(cfg.train.backend, "reference")
        self.assertGreater(cfg.synth.paraphrase_per_example, 0)


# --------------------------------------------------------------------------- #
class TestRecords(unittest.TestCase):
    def test_jsonl_roundtrip(self) -> None:
        d = scratch_dir("records")
        try:
            p = d / "x.jsonl"
            recs = [SynthRecord(id="a", source_id="s", db_id="db", kind="cot",
                                question="q", sql="SELECT 1", reasoning="why"),
                    SynthRecord(id="b", source_id="s", db_id="db", kind="paraphrase",
                                question="q2", sql="SELECT 2")]
            self.assertEqual(write_jsonl(p, recs), 2)
            back = [SynthRecord(**x) for x in read_jsonl(p)]
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0].reasoning, "why")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_tool_schema_openai_shape(self) -> None:
        tool = ToolSchema().as_openai_tool()
        self.assertEqual(tool["type"], "function")
        self.assertIn("parameters", tool["function"])
        self.assertEqual(tool["function"]["parameters"]["required"], ["sql"])

    def test_prediction_dict(self) -> None:
        p = Prediction(id="1", example_id="e", db_id="d", question="q",
                       gold_sql="SELECT 1", pred_sql="SELECT 1", exec_match=True)
        self.assertEqual(p.as_dict()["exec_match"], True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
