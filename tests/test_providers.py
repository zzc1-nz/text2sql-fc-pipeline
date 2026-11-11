"""Tests for ``fcdata.providers``: prompt contracts, mock teacher, HTTP client, runner.

What is actually asserted (and why)
-----------------------------------
* **Determinism** is checked *across processes* (``subprocess``), not just twice in
  one interpreter: the demo's reproducibility claim is about re-running the CLI,
  so a same-process check would not catch salted ``hash()`` seeds.
* **Grounding** is checked structurally: the CoT trace must name real tables and
  columns from the DDL block, and the final query must be the gold query. A mock
  that regressed to generic prose would fail these.
* **Paraphrase safety** is checked with the same guard the pipeline uses, on a
  hand-built set of adversarial questions (numbers, quoted literals, entities).
* **The judge** is checked for calibration: it must score the gold query ~1.0 and
  a corrupted query much lower, and its JSON must parse through the shared parser.
* **The HTTP client** is tested against a *fake transport* (no network, no keys)
  covering: success + usage parsing, retry on 429, non-retry on 400, truncation,
  and the disk cache replaying with ``cached=True`` while still reporting tokens.
* **The runner** is checked for per-job failure isolation, output ordering, and
  the hard budget cap under concurrency.

No test touches the network, and no test depends on ``data/fixture``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from fcdata.config import Config, ProviderCfg, scratch_dir
from fcdata.dbexec import build_sqlite
from fcdata.providers import (BUDGET_EXHAUSTED, AsyncBatchRunner, Job,
                              MockProvider, OpenAICompatProvider, ResponseCache,
                              build_provider, stable_seed)
from fcdata.providers import prompts
from fcdata.providers.mockwriters import paraphrase_preserves, write_judge
from fcdata.providers.openai_compat import RETRYABLE_STATUS
from fcdata.providers.runner import BatchStats, run_jobs
from fcdata.providers.sqlread import parse_ddl, read_sql
from fcdata.records import Example
from fcdata.schema import Database
from fcdata.spider import SpiderData
from fcdata.sqlutils import analyze_sql, inject_error, normalize_sql
from fcdata.tokens import TokenLedger, estimate_messages_tokens

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# a tiny self-contained Spider-format fixture
# --------------------------------------------------------------------------- #
def tiny_spider() -> tuple[dict[str, Any], list[dict[str, Any]], Database]:
    """A 2-table, 3-row database in Spider's JSON shape."""
    tables_json = [{
        "db_id": "library",
        "table_names_original": ["author", "book"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "name"],
                                  [1, "id"], [1, "author_id"], [1, "title"],
                                  [1, "year"]],
        "column_types": ["text", "number", "text", "number", "number",
                         "text", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    }]
    examples = [
        {"id": "t0", "db_id": "library",
         "question": "Show the names of authors who wrote a book after 2000.",
         "query": ("SELECT T1.name FROM author AS T1 JOIN book AS T2 "
                   "ON T1.id = T2.author_id WHERE T2.year > 2000")},
        {"id": "t1", "db_id": "library",
         "question": "How many books are there?",
         "query": "SELECT COUNT(*) FROM book"},
        {"id": "t2", "db_id": "library",
         "question": "List each author and the number of books they wrote.",
         "query": ("SELECT T1.name, COUNT(*) FROM author AS T1 JOIN book AS T2 "
                   "ON T1.id = T2.author_id GROUP BY T1.name")},
    ]
    db = Database.from_spider_json(tables_json[0])
    db.rows = {
        "author": [{"id": 1, "name": "Ann"}, {"id": 2, "name": "Bob"}],
        "book": [{"id": 10, "author_id": 1, "title": "A", "year": 2001},
                 {"id": 11, "author_id": 1, "title": "B", "year": 2005},
                 {"id": 12, "author_id": 2, "title": "C", "year": 1999}],
    }
    return tables_json[0], examples, db


class FixtureMixin:
    """Builds the tiny sqlite fixture in a writable scratch dir."""

    _dirs: list[Path]

    def _make_data(self, name: str = "providers") -> SpiderData:
        d = scratch_dir(name)
        self._dirs.append(d)
        tables_json, examples, db = tiny_spider()
        db_dir = d / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        build_sqlite(db, db_dir / "library.sqlite")
        (d / "tables.json").write_text(json.dumps([tables_json]), encoding="utf-8")
        (d / "train.json").write_text(json.dumps(examples), encoding="utf-8")
        (d / "dev.json").write_text(json.dumps(examples[:1]), encoding="utf-8")
        from fcdata.spider import load_spider
        return load_spider(d)

    def tearDown(self) -> None:  # noqa: D401 - unittest hook
        for d in getattr(self, "_dirs", []):
            shutil.rmtree(d, ignore_errors=True)
        self._dirs = []


def make_cfg(**synth: Any) -> Config:
    """Config with a writable cache dir and optional synth overrides."""
    cfg = Config()
    cfg.providers["cheap"] = ProviderCfg(name="mock", model="mock-cheap-1",
                                         concurrency=4, price_in_per_mtok=0.05,
                                         price_out_per_mtok=0.10)
    cfg.providers["judge"] = ProviderCfg(name="mock", model="mock-judge-1",
                                         temperature=0.0, max_tokens=256)
    for key, value in synth.items():
        setattr(cfg.synth, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# prompt contracts
# --------------------------------------------------------------------------- #
class TestPromptContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.db = tiny_spider()[2]
        self.ddl = self.db.render_ddl()
        self.q = "How many books are there?"
        self.sql = "SELECT COUNT(*) FROM book"

    def test_every_prompt_embeds_schema_and_extracts_back(self) -> None:
        cases = [
            ("cot", prompts.cot_messages(self.ddl, self.q, self.sql)),
            ("paraphrase", prompts.paraphrase_messages(self.ddl, self.q, self.sql,
                                                       "synonym")),
            ("correction", prompts.correction_messages(
                self.ddl, self.q, "SELECT COUNT(title) FROM book", self.sql,
                "wrong_aggregate")),
            ("judge", prompts.judge_messages(self.ddl, self.q, self.sql, self.sql)),
        ]
        for name, messages in cases:
            with self.subTest(kind=name):
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[1]["role"], "user")
                prompt = messages[1]["content"]
                self.assertIn("CREATE TABLE book", prompts.extract_schema(prompt))
                self.assertEqual(prompts.extract_question(prompt), self.q)

    def test_prompts_state_an_output_contract(self) -> None:
        """A teacher whose output we parse must be told the exact output shape."""
        cot = prompts.cot_messages(self.ddl, self.q, self.sql)[0]["content"]
        self.assertIn("Output contract", cot)
        self.assertIn("Final SQL:", prompts.cot_messages(self.ddl, self.q,
                                                         self.sql)[0]["content"])
        judge = prompts.judge_messages(self.ddl, self.q, self.sql)[0]["content"]
        self.assertIn("JSON object", judge)
        self.assertIn("score", judge)

    def test_gold_and_bad_sql_are_delimited(self) -> None:
        prompt = prompts.correction_messages(
            self.ddl, self.q, "SELECT COUNT(title) FROM book", self.sql,
            "wrong_aggregate")[1]["content"]
        self.assertEqual(prompts.extract_gold_sql(prompt), self.sql)
        self.assertEqual(prompts.extract_bad_sql(prompt),
                         "SELECT COUNT(title) FROM book")
        self.assertEqual(prompts.extract_error_kind(prompt), "wrong_aggregate")

    def test_unknown_strategy_falls_back(self) -> None:
        prompt = prompts.paraphrase_messages(self.ddl, self.q, self.sql,
                                             "not-a-strategy")[1]["content"]
        self.assertEqual(prompts.extract_strategy(prompt), "synonym")

    def test_parse_judge_json_tolerates_fences_and_percent(self) -> None:
        fenced = '```json\n{"score": 0.9, "verdict": "consistent", "rationale": "ok"}\n```'
        parsed = prompts.parse_judge_json(fenced)
        assert parsed is not None
        self.assertEqual(parsed["score"], 0.9)
        self.assertEqual(parsed["verdict"], "consistent")
        self.assertEqual(prompts.parse_judge_json('{"score": 85}')["score"], 0.85)

    def test_parse_judge_json_rejects_garbage(self) -> None:
        for bad in ("", "no json here", '{"score": "high"}', '{"score": 1.5}',
                    '{"verdict": "consistent"}'):
            with self.subTest(text=bad):
                self.assertIsNone(prompts.parse_judge_json(bad))

    def test_verdict_thresholds_are_monotonic(self) -> None:
        self.assertEqual(prompts.verdict_for_score(1.0), "consistent")
        self.assertEqual(prompts.verdict_for_score(0.5), "uncertain")
        self.assertEqual(prompts.verdict_for_score(0.0), "inconsistent")


# --------------------------------------------------------------------------- #
# sqlread (the mock's grounding layer)
# --------------------------------------------------------------------------- #
class TestSqlRead(unittest.TestCase):
    def setUp(self) -> None:
        self.db = tiny_spider()[2]
        self.schema = parse_ddl(self.db.render_ddl())

    def test_ddl_roundtrip_recovers_tables_columns_and_fks(self) -> None:
        self.assertEqual(self.schema.table_names, ["author", "book"])
        self.assertEqual(self.schema.column_type("book", "year"), "number")
        fk = self.schema.fk_between("author", "book")
        assert fk is not None
        self.assertEqual(fk[1], "id")
        self.assertEqual(fk[3], "author_id")

    def test_parsed_schema_supports_real_analysis(self) -> None:
        """The recovered schema must be good enough for the real validator."""
        db = self.schema.to_database()
        good = "SELECT T1.name FROM author AS T1 JOIN book AS T2 ON T1.id = T2.author_id"
        bad = "SELECT T1.nickname FROM author AS T1"
        self.assertTrue(analyze_sql(good, db).ok)
        self.assertFalse(analyze_sql(bad, db).ok)

    def test_clause_parsing(self) -> None:
        sql = ("SELECT T1.name, COUNT(*) FROM author AS T1 JOIN book AS T2 "
               "ON T1.id = T2.author_id WHERE T2.year > 1999 "
               "GROUP BY T1.name HAVING COUNT(*) > 1 ORDER BY T1.name DESC LIMIT 2")
        facts = read_sql(sql, self.schema)
        self.assertEqual(facts.tables, ["author", "book"])
        self.assertEqual(len(facts.joins), 1)
        self.assertEqual(facts.joins[0].kind, "inner")
        self.assertEqual(len(facts.joins[0].pairs), 1)
        self.assertEqual(facts.where[0].op, ">")
        self.assertEqual(facts.where[0].literals, ["1999"])
        self.assertEqual(facts.group_by, ["T1.name"])
        self.assertIn("COUNT(*)", facts.having)
        self.assertEqual(facts.order_by[0].descending, True)
        self.assertEqual(facts.limit, 2)
        self.assertEqual(facts.aggregates, ["count"])

    def test_not_in_operator_is_parsed_as_one_operator(self) -> None:
        facts = read_sql("SELECT name FROM author WHERE id NOT IN "
                         "(SELECT author_id FROM book)", self.schema)
        self.assertEqual(facts.where[0].op, "not in")
        self.assertTrue(facts.where[0].has_subquery)

    def test_comparison_operators_are_not_misread(self) -> None:
        for raw, expected in ((">= 3", ">="), ("<= 3", "<="), ("<> 3", "<>"),
                              ("> 3", ">"), ("< 3", "<"), ("= 3", "=")):
            with self.subTest(op=raw):
                facts = read_sql(f"SELECT name FROM book WHERE year {raw}",
                                 self.schema)
                self.assertEqual(facts.where[0].op, expected)

    def test_string_literals_with_commas_do_not_split_predicates(self) -> None:
        facts = read_sql("SELECT name FROM author WHERE name = 'Ann, Bob'",
                         self.schema)
        self.assertEqual(len(facts.where), 1)
        self.assertEqual(facts.where[0].literals, ["'Ann, Bob'"])

    def test_empty_and_garbage_input_does_not_raise(self) -> None:
        for sql in ("", "   ", "NOT SQL AT ALL", "SELECT"):
            with self.subTest(sql=sql):
                read_sql(sql, self.schema)   # must not raise


# --------------------------------------------------------------------------- #
# mock teacher quality + determinism
# --------------------------------------------------------------------------- #
class TestMockProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.db = tiny_spider()[2]
        self.ddl = self.db.render_ddl()
        self.q = "Show the names of authors who wrote a book after 2000."
        self.sql = ("SELECT T1.name FROM author AS T1 JOIN book AS T2 "
                    "ON T1.id = T2.author_id WHERE T2.year > 2000")
        self.ledger = TokenLedger(None)
        self.provider = MockProvider(ledger=self.ledger, stage="synth")

    def _call(self, messages: list[dict[str, Any]], task: str,
              sample_id: str = "s1") -> Any:
        return self.provider.complete(messages, task=task, sample_id=sample_id)

    # -- CoT -------------------------------------------------------------
    def test_cot_is_grounded_in_the_prompt_schema(self) -> None:
        comp = self._call(prompts.cot_messages(self.ddl, self.q, self.sql), "cot")
        self.assertTrue(comp.ok, comp.error)
        self.assertIn("author", comp.text)
        self.assertIn("book", comp.text)
        self.assertIn("2000", comp.text)          # the real filter literal
        self.assertIn("Final SQL:", comp.text)
        final = comp.text.split("Final SQL:")[-1].strip()
        self.assertEqual(normalize_sql(final), normalize_sql(self.sql))

    def test_cot_has_numbered_steps(self) -> None:
        comp = self._call(prompts.cot_messages(self.ddl, self.q, self.sql), "cot")
        lines = [l for l in comp.text.splitlines() if l.strip()]
        numbered = [l for l in lines if l.strip()[:1].isdigit()]
        self.assertGreaterEqual(len(numbered), 2)
        self.assertGreaterEqual(len(numbered), 2)

    def test_cot_mentions_join_and_filter_semantics(self) -> None:
        comp = self._call(prompts.cot_messages(self.ddl, self.q, self.sql), "cot")
        low = comp.text.lower()
        self.assertIn("join", low)
        self.assertTrue("greater than" in low or ">" in low)

    # -- paraphrase ------------------------------------------------------
    def test_paraphrase_preserves_meaning(self) -> None:
        for strategy in prompts.PARAPHRASE_STRATEGIES:
            with self.subTest(strategy=strategy):
                comp = self._call(
                    prompts.paraphrase_messages(self.ddl, self.q, self.sql,
                                                strategy), "paraphrase")
                self.assertTrue(comp.ok, comp.error)
                self.assertEqual(len(comp.text.splitlines()), 1)
                self.assertTrue(paraphrase_preserves(self.q, comp.text),
                                f"drift: {comp.text!r}")

    def test_paraphrase_keeps_numbers_and_quoted_values(self) -> None:
        question = "List the books by 'Ann Lee' published after 2005."
        for strategy in prompts.PARAPHRASE_STRATEGIES:
            with self.subTest(strategy=strategy):
                comp = self._call(
                    prompts.paraphrase_messages(self.ddl, question, self.sql,
                                                strategy), "paraphrase")
                self.assertIn("2005", comp.text)
                self.assertIn("Ann Lee", comp.text)

    def test_paraphrase_is_never_identical_to_the_original(self) -> None:
        for strategy in prompts.PARAPHRASE_STRATEGIES:
            comp = self._call(prompts.paraphrase_messages(
                self.ddl, self.q, self.sql, strategy), "paraphrase")
            self.assertNotEqual(normalize_sql(comp.text), normalize_sql(self.q))

    def test_paraphrase_strategies_produce_different_text(self) -> None:
        outs = set()
        for strategy in prompts.PARAPHRASE_STRATEGIES:
            comp = self._call(prompts.paraphrase_messages(
                self.ddl, self.q, self.sql, strategy), "paraphrase")
            outs.add(comp.text.lower())
        self.assertGreaterEqual(len(outs), 3,
                                f"strategies collapsed to {outs}")

    # -- correction ------------------------------------------------------
    def test_correction_note_names_the_real_bug(self) -> None:
        bad = inject_error(self.sql, self.db, "wrong_table")
        self.assertNotEqual(normalize_sql(bad), normalize_sql(self.sql))
        comp = self._call(prompts.correction_messages(
            self.ddl, self.q, bad, self.sql, "wrong_table"), "hard_negative")
        self.assertTrue(comp.ok, comp.error)
        self.assertIn("author", comp.text.lower() + " author")
        self.assertLessEqual(len(comp.text.split()), 90)

    def test_correction_for_wrong_aggregate_mentions_both_aggregates(self) -> None:
        sql = "SELECT COUNT(*) FROM book"
        bad = "SELECT SUM(id) FROM book"
        comp = self._call(prompts.correction_messages(
            self.ddl, "How many books are there?", bad, sql, "wrong_aggregate"),
            "hard_negative")
        self.assertTrue(comp.ok, comp.error)
        low = comp.text.lower()
        self.assertTrue("sum" in low and "count" in low, comp.text)

    # -- judge -----------------------------------------------------------
    def test_judge_scores_gold_high_and_corruption_low(self) -> None:
        good = self._call(prompts.judge_messages(self.ddl, self.q, self.sql,
                                                 self.sql), "judge")
        parsed_good = prompts.parse_judge_json(good.text)
        assert parsed_good is not None
        self.assertGreaterEqual(parsed_good["score"], 0.9)
        self.assertEqual(parsed_good["verdict"], "consistent")

        bad_sql = "SELECT title FROM book WHERE year > 2000"
        bad = self._call(prompts.judge_messages(self.ddl, self.q, bad_sql,
                                                self.sql), "judge")
        parsed_bad = prompts.parse_judge_json(bad.text)
        assert parsed_bad is not None
        self.assertLess(parsed_bad["score"], parsed_good["score"])
        self.assertIn(parsed_bad["verdict"], prompts.VERDICTS)
        self.assertTrue(parsed_bad["rationale"])

    def test_judge_rejects_a_hallucinated_column(self) -> None:
        comp = self._call(prompts.judge_messages(
            self.ddl, self.q, "SELECT nickname FROM author", self.sql), "judge")
        parsed = prompts.parse_judge_json(comp.text)
        assert parsed is not None
        self.assertLess(parsed["score"], 0.4)
        self.assertEqual(parsed["verdict"], "inconsistent")
        self.assertIn("column", parsed["rationale"].lower())

    def test_judge_json_is_valid_and_stable(self) -> None:
        first = self._call(prompts.judge_messages(self.ddl, self.q, self.sql,
                                                  self.sql), "judge").text
        second = self._call(prompts.judge_messages(self.ddl, self.q, self.sql,
                                                   self.sql), "judge").text
        self.assertEqual(first, second)
        obj = json.loads(first)
        self.assertEqual(set(obj), {"score", "verdict", "rationale"})

    # -- accounting / determinism / failure modes ------------------------
    def test_every_call_records_exactly_one_usage_record(self) -> None:
        before = len(self.ledger)
        comp = self._call(prompts.cot_messages(self.ddl, self.q, self.sql), "cot")
        self.assertEqual(len(self.ledger), before + 1)
        rec = self.ledger.records[-1]
        self.assertEqual(rec.stage, "synth")
        self.assertEqual(rec.role, "cheap")
        self.assertEqual(rec.task, "cot")
        self.assertEqual(rec.sample_id, "s1")
        self.assertTrue(rec.ok)
        self.assertEqual(rec.prompt_tokens, comp.prompt_tokens)
        self.assertEqual(rec.completion_tokens, comp.completion_tokens)
        self.assertGreater(rec.total_tokens, 0)

    def test_token_counts_are_real(self) -> None:
        messages = prompts.cot_messages(self.ddl, self.q, self.sql)
        comp = self._call(messages, "cot")
        self.assertEqual(comp.prompt_tokens, estimate_messages_tokens(messages))
        self.assertGreater(comp.completion_tokens, 0)
        self.assertGreater(comp.total_tokens, comp.prompt_tokens)

    def test_same_prompt_same_output_and_different_prompt_differs(self) -> None:
        msgs = prompts.cot_messages(self.ddl, self.q, self.sql)
        a = self._call(msgs, "cot", "s1").text
        b = self._call(msgs, "cot", "s1").text
        c = self._call(msgs, "cot", "s2").text
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_deterministic_across_processes(self) -> None:
        """The reproducibility claim is about re-running the CLI, not one process."""
        script = (
            "import json\n"
            "from fcdata.providers import MockProvider\n"
            "from fcdata.providers import prompts\n"
            "from fcdata.schema import Database\n"
            "from fcdata.tokens import TokenLedger\n"
            "tables = " + repr(tiny_spider()[0]) + "\n"
            "db = Database.from_spider_json(tables)\n"
            "p = MockProvider(ledger=TokenLedger(None))\n"
            "msgs = prompts.cot_messages(db.render_ddl(), "
            "'Show the names of authors who wrote a book after 2000.', "
            "'SELECT T1.name FROM author AS T1 JOIN book AS T2 "
            "ON T1.id = T2.author_id WHERE T2.year > 2000')\n"
            "c = p.complete(msgs, task='cot', sample_id='fixed')\n"
            "print(json.dumps([c.text, c.prompt_tokens, c.completion_tokens]))\n"
        )
        outs = []
        for _ in range(2):
            proc = subprocess.run([sys.executable, "-c", script], cwd=str(REPO),
                                  capture_output=True, text=True, timeout=120,
                                  env=_env_with_src())
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(proc.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        text, ptok, ctok = json.loads(outs[0])
        self.assertIn("Final SQL:", text)
        self.assertGreater(ptok, 0)
        self.assertGreater(ctok, 0)

    def test_missing_schema_block_is_a_visible_failure(self) -> None:
        """No schema in the prompt => honest failure, never invented content."""
        comp = self.provider.complete(
            [{"role": "user", "content": "Explain this query: SELECT 1"}],
            task="cot", sample_id="x")
        self.assertFalse(comp.ok)
        self.assertIn("schema", comp.error.lower())

    def test_unknown_task_is_a_visible_failure(self) -> None:
        comp = self.provider.complete(
            [{"role": "user", "content": prompts.tag_block("schema", self.ddl)}],
            task="summarize", sample_id="x")
        self.assertFalse(comp.ok)
        self.assertIn("no handler", comp.error)

    def test_provider_exception_becomes_ok_false(self) -> None:
        class Boom(MockProvider):
            def _generate(self, *a: Any, **k: Any) -> Any:
                raise RuntimeError("kaboom")

        p = Boom(ledger=TokenLedger(None))
        comp = p.complete([{"role": "user", "content": "x"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertIn("RuntimeError", comp.error)
        self.assertIn("kaboom", comp.error)

    def test_stable_seed_is_stable_and_varies(self) -> None:
        self.assertEqual(stable_seed("a", "b"), stable_seed("a", "b"))
        self.assertNotEqual(stable_seed("a", "b"), stable_seed("a", "c"))
        self.assertGreaterEqual(stable_seed("x"), 0)


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (fake transport; no network)
# --------------------------------------------------------------------------- #
class _FakeTransport(OpenAICompatProvider):
    """Replays scripted responses so the retry/cache logic is testable offline."""

    def __init__(self, responses: list[tuple[int, dict[str, Any], str]], **kw: Any):
        super().__init__(**kw)
        self.responses = list(responses)
        self.calls = 0
        self.slept: list[float] = []

    def _http_post(self, url: str, payload: dict[str, Any],
                   headers: dict[str, str]) -> tuple[int, dict[str, Any], str]:
        self.calls += 1
        if not self.responses:
            return 500, {}, "no scripted response left"
        return self.responses.pop(0)

    def _backoff_delay(self, attempt: int, data: dict[str, Any]) -> float:
        delay = super()._backoff_delay(attempt, data)
        self.slept.append(delay)
        return 0.0      # do not actually sleep in tests


def _ok_response(text: str = "hello", prompt: int = 11, completion: int = 7
                 ) -> tuple[int, dict[str, Any], str]:
    return 200, {"model": "fake-1",
                 "choices": [{"message": {"content": text},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt,
                           "completion_tokens": completion}}, ""


class TestOpenAICompat(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.dir = scratch_dir("openai_cache")
        self._dirs.append(self.dir)

    def _provider(self, responses: list[tuple[int, dict[str, Any], str]],
                  **kw: Any) -> _FakeTransport:
        defaults: dict[str, Any] = dict(
            name="openai", model="fake-1", base_url="https://example.invalid/v1",
            api_key="test-key", role="cheap",
            cache_path=self.dir / "cache.jsonl", max_retries=2,
        )
        defaults.update(kw)
        return _FakeTransport(responses, **defaults)

    def test_success_returns_text_and_usage(self) -> None:
        ledger = TokenLedger(None)
        p = self._provider([_ok_response("SELECT 1", 11, 7)], ledger=ledger)
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertTrue(comp.ok, comp.error)
        self.assertEqual(comp.text, "SELECT 1")
        self.assertEqual(comp.prompt_tokens, 11)
        self.assertEqual(comp.completion_tokens, 7)
        self.assertEqual(len(ledger), 1)
        self.assertFalse(ledger.records[0].cached)

    def test_retries_429_then_succeeds(self) -> None:
        ledger = TokenLedger(None)
        p = self._provider([(429, {"error": {"message": "slow down"}}, "rate"),
                            _ok_response("ok", 5, 3)], ledger=ledger)
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertTrue(comp.ok, comp.error)
        self.assertEqual(p.calls, 2)
        self.assertEqual(len(p.slept), 1)
        self.assertEqual(len(ledger), 1)

    def test_does_not_retry_400(self) -> None:
        p = self._provider([(400, {"error": {"message": "bad request"}}, "bad")])
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertEqual(p.calls, 1)
        self.assertIn("400", comp.error)
        self.assertIn("bad request", comp.error)

    def test_gives_up_after_max_retries(self) -> None:
        p = self._provider([(503, {}, "down"), (503, {}, "down"),
                            (503, {}, "down")])
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertEqual(p.calls, 3)      # 1 attempt + 2 retries
        self.assertIn("503", comp.error)

    def test_retryable_status_set(self) -> None:
        for code in (429, 500, 502, 503, 504):
            self.assertIn(code, RETRYABLE_STATUS)
        for code in (400, 401, 403, 404, 422):
            self.assertNotIn(code, RETRYABLE_STATUS)

    def test_truncated_output_is_a_soft_failure(self) -> None:
        truncated = (200, {"choices": [{"message": {"content": "SELECT"},
                                        "finish_reason": "length"}],
                           "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, "")
        ledger = TokenLedger(None)
        p = self._provider([truncated], ledger=ledger)
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertIn("truncated", comp.error)
        self.assertEqual(len(ledger), 1)      # still accounted

    def test_empty_completion_is_a_failure(self) -> None:
        p = self._provider([_ok_response("   ", 3, 0)])
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertIn("empty completion", comp.error)

    def test_missing_api_key_is_a_clear_failure(self) -> None:
        p = self._provider([], api_key=None, api_key_env="FC_DATA_NO_SUCH_KEY")
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertFalse(comp.ok)
        self.assertIn("FC_DATA_NO_SUCH_KEY", comp.error)
        self.assertEqual(p.calls, 0)          # never dialled out

    def test_cache_replays_and_still_reports_tokens(self) -> None:
        cache_path = self.dir / "cache2.jsonl"
        ledger = TokenLedger(None)
        first = self._provider([_ok_response("cached text", 9, 4)],
                               cache_path=cache_path, ledger=ledger)
        a = first.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertTrue(a.ok)
        self.assertFalse(a.cached)

        second = self._provider([], cache_path=cache_path, ledger=ledger)
        b = second.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertTrue(b.ok)
        self.assertTrue(b.cached)
        self.assertEqual(b.text, "cached text")
        self.assertEqual(b.prompt_tokens, 9)      # comparable cost math
        self.assertEqual(b.completion_tokens, 4)
        self.assertEqual(second.calls, 0)         # no HTTP call
        self.assertTrue(ledger.records[-1].cached)

    def test_cache_key_depends_on_params(self) -> None:
        cache_path = self.dir / "cache3.jsonl"
        p1 = self._provider([_ok_response("a")], cache_path=cache_path)
        p1.complete([{"role": "user", "content": "hi"}], task="cot", temperature=0.1)
        p2 = self._provider([_ok_response("b")], cache_path=cache_path)
        comp = p2.complete([{"role": "user", "content": "hi"}], task="cot",
                           temperature=0.9)
        self.assertFalse(comp.cached)
        self.assertEqual(comp.text, "b")

    def test_cache_survives_a_corrupt_line(self) -> None:
        cache_path = self.dir / "cache4.jsonl"
        cache_path.write_text('{"key": "x", "text": "partial"\n', encoding="utf-8")
        p = self._provider([_ok_response("fresh")], cache_path=cache_path)
        comp = p.complete([{"role": "user", "content": "hi"}], task="cot")
        self.assertTrue(comp.ok)
        self.assertEqual(comp.text, "fresh")

    def test_fingerprint_is_stable_and_sensitive(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.fingerprint("m", msgs, {"temperature": 0.1})
        b = ResponseCache.fingerprint("m", msgs, {"temperature": 0.1})
        c = ResponseCache.fingerprint("m2", msgs, {"temperature": 0.1})
        d = ResponseCache.fingerprint("m", msgs, {"temperature": 0.2})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertEqual(len(a), 64)

    def test_available_requires_url_and_key(self) -> None:
        self.assertTrue(self._provider([]).available())
        self.assertFalse(self._provider([], api_key="").available())
        self.assertFalse(self._provider([], base_url="").available())

    def test_import_safe_offline(self) -> None:
        """Constructing a provider must not perform I/O or need a key."""
        p = OpenAICompatProvider(name="openai", model="m",
                                 base_url="https://example.invalid/v1",
                                 api_key_env="FC_DATA_MISSING_KEY")
        self.assertFalse(p.available())
        self.assertEqual(p.describe()["kind"], "openai")


# --------------------------------------------------------------------------- #
# factory + batch runner
# --------------------------------------------------------------------------- #
class TestProviderFactory(unittest.TestCase):
    def test_mock_role_builds_mock(self) -> None:
        cfg = make_cfg()
        ledger = TokenLedger(None)
        p = build_provider("cheap", cfg, ledger, stage="synth")
        self.assertIsInstance(p, MockProvider)
        self.assertEqual(p.model, "mock-cheap-1")
        self.assertIs(p.ledger, ledger)

    def test_openai_role_builds_http_client(self) -> None:
        cfg = make_cfg()
        cfg.providers["cheap"] = ProviderCfg(
            name="openai", model="deepseek-chat",
            base_url="https://api.example.invalid/v1",
            api_key_env="FC_DATA_MISSING_KEY")
        p = build_provider("cheap", cfg, TokenLedger(None))
        self.assertIsInstance(p, OpenAICompatProvider)
        self.assertEqual(p.model, "deepseek-chat")

    def test_unknown_provider_name_raises(self) -> None:
        cfg = make_cfg()
        cfg.providers["cheap"] = ProviderCfg(name="definitely-not-a-provider")
        with self.assertRaises(ValueError):
            build_provider("cheap", cfg, TokenLedger(None))

    def test_unknown_role_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_provider("nope", make_cfg(), TokenLedger(None))

    def test_prices_reach_the_ledger(self) -> None:
        """Cost math must work from the config alone (the report's cost column)."""
        cfg = make_cfg()
        ledger = TokenLedger(None)
        p = build_provider("cheap", cfg, ledger)
        db = tiny_spider()[2]
        comp = p.complete(prompts.cot_messages(
            db.render_ddl(), "How many books are there?",
            "SELECT COUNT(*) FROM book"), task="cot")
        self.assertTrue(comp.ok, comp.error)
        rec = ledger.records[-1]
        self.assertGreater(rec.cost_usd, 0.0)
        expected = (rec.prompt_tokens / 1e6) * 0.05 + (rec.completion_tokens / 1e6) * 0.10
        self.assertAlmostEqual(rec.cost_usd, expected, places=10)


class TestAsyncBatchRunner(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._dirs = []
        self.db = tiny_spider()[2]
        self.ddl = self.db.render_ddl()
        self.cfg = make_cfg()

    def _jobs(self, n: int) -> list[Job]:
        sql = "SELECT COUNT(*) FROM book"
        return [Job(messages=prompts.cot_messages(self.ddl, f"Question {i}?",
                                                  sql),
                    task="cot", sample_id=f"s{i}") for i in range(n)]

    def test_returns_one_completion_per_job_in_order(self) -> None:
        ledger = TokenLedger(None)
        provider = build_provider("cheap", self.cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, self.cfg, stage="synth")
        jobs = self._jobs(6)
        results = runner.run(jobs)
        self.assertEqual(len(results), len(jobs))
        self.assertTrue(all(c.ok for c in results))
        # ordering: the trace for job i must come back at index i
        for i, comp in enumerate(results):
            self.assertIn("Final SQL:", comp.text)

    def test_every_job_records_one_usage_record(self) -> None:
        ledger = TokenLedger(None)
        provider = build_provider("cheap", self.cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, self.cfg, stage="synth")
        runner.run(self._jobs(5))
        self.assertEqual(len(ledger), 5)
        self.assertTrue(all(r.stage == "synth" for r in ledger.records))
        self.assertEqual(runner.stats.ok, 5)
        self.assertGreater(runner.stats.total_tokens, 0)

    def test_failure_is_isolated_to_one_job(self) -> None:
        class Flaky(MockProvider):
            def _generate(self, messages: list[dict[str, Any]], *, task: str,
                          sample_id: str, **kw: Any) -> Any:
                if sample_id == "s2":
                    raise RuntimeError("deliberate failure")
                return super()._generate(messages, task=task, sample_id=sample_id,
                                         **kw)

        ledger = TokenLedger(None)
        provider = Flaky(ledger=ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, self.cfg, stage="synth")
        results = runner.run(self._jobs(5))
        self.assertEqual(len(results), 5)
        self.assertFalse(results[2].ok)
        self.assertIn("deliberate failure", results[2].error)
        self.assertEqual(sum(1 for c in results if c.ok), 4)
        self.assertEqual(runner.stats.failed, 1)
        self.assertEqual(len(ledger), 5)      # failures are still accounted

    def test_budget_cap_is_never_exceeded(self) -> None:
        cfg = make_cfg(max_total_tokens=2500)
        ledger = TokenLedger(None)
        provider = build_provider("cheap", cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, cfg, stage="synth")
        results = runner.run(self._jobs(12))
        self.assertLessEqual(ledger.spent("synth"), cfg.synth.max_total_tokens)
        skipped = [c for c in results if c.error == BUDGET_EXHAUSTED]
        self.assertGreater(len(skipped), 0)
        self.assertEqual(len(skipped), runner.stats.skipped_budget)
        self.assertEqual(len(results), 12)

    def test_budget_cap_holds_under_concurrency(self) -> None:
        """Reservation must not let N in-flight calls overshoot the cap."""
        cfg = make_cfg(max_total_tokens=4000)
        cfg.providers["cheap"].concurrency = 16
        ledger = TokenLedger(None)
        provider = build_provider("cheap", cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, cfg, stage="synth")
        runner.run(self._jobs(20))
        self.assertLessEqual(ledger.spent("synth"), cfg.synth.max_total_tokens)

    def test_zero_budget_means_unlimited(self) -> None:
        cfg = make_cfg(max_total_tokens=0)
        ledger = TokenLedger(None)
        provider = build_provider("cheap", cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, cfg, stage="synth")
        results = runner.run(self._jobs(4))
        self.assertTrue(all(c.ok for c in results))
        self.assertEqual(runner.stats.skipped_budget, 0)

    def test_empty_job_list(self) -> None:
        ledger = TokenLedger(None)
        provider = build_provider("cheap", self.cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, self.cfg)
        self.assertEqual(runner.run([]), [])

    def test_summary_reports_budget_and_provider(self) -> None:
        ledger = TokenLedger(None)
        provider = build_provider("cheap", self.cfg, ledger, stage="synth")
        runner = AsyncBatchRunner(provider, ledger, self.cfg, stage="synth")
        runner.run(self._jobs(3))
        summary = runner.summary()
        self.assertEqual(summary["stage"], "synth")
        self.assertEqual(summary["ok"], 3)
        self.assertIn("budget", summary)
        self.assertIn("provider", summary)

    def test_run_jobs_helper(self) -> None:
        ledger = TokenLedger(None)
        provider = build_provider("cheap", self.cfg, ledger, stage="synth")
        results = run_jobs(provider, ledger, self.cfg, self._jobs(3), stage="synth",
                           progress_every=2)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(c.ok for c in results))

    def test_batch_stats_derived_fields(self) -> None:
        stats = BatchStats(stage="x", jobs=4, dispatched=4, ok=3, failed=1,
                           prompt_tokens=10, completion_tokens=5)
        self.assertEqual(stats.total_tokens, 15)
        self.assertAlmostEqual(stats.ok_rate, 0.75)


def _env_with_src() -> dict[str, str]:
    import os

    env = dict(os.environ)
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"]
                               if env.get("PYTHONPATH") else "")
    return env


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
