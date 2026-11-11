"""Unit tests for the function-calling format stage.

The whole point of this stage is the *target syntax*, so the tests are written as
an executable statement of the product contract:

* the assistant turn is a real OpenAI-style tool call,
* ``arguments`` is a JSON **string** (not an object),
* the SQL inside round-trips through ``sqlutils.extract_sql``,
* the rendered text parses back to the same SQL for all three chat templates,
* ``render_prompt(msgs[:-1]) + render_target_text(rec) == render_messages(msgs)``
  so training and inference prompts cannot drift apart.

Self-contained: a two-table toy database in a scratch dir; no network.
Run with::

    $env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_format.py' -v
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from typing import Any

from fcdata.config import Config, load_config, scratch_dir
from fcdata.dbexec import build_sqlite
from fcdata.format import (DEFAULT_TEMPLATE, TEMPLATES, assistant_sql, build_and_write,
                           build_assistant_message, build_eval_messages,
                           build_eval_prompt, build_sft_records, parse_tool_call,
                           render_messages, render_prompt, render_supervised_pair,
                           render_target_text, sanitize_reasoning, tool_definitions,
                           tool_schema, validate_target)
from fcdata.format.chat import derive_plan
from fcdata.records import Example, SynthRecord, ValidationReport, ValidatedRecord, read_sft
from fcdata.schema import Database
from fcdata.spider import SpiderData
from fcdata.sqlutils import extract_sql, normalize_sql

# --------------------------------------------------------------------------- #
# tiny fixture (same shape as tests/test_validate.py, kept independent)
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
JOIN_SQL = ("SELECT T1.stu_name, T2.dept_name FROM student AS T1 "
            "JOIN department AS T2 ON T1.dept_id = T2.dept_id")
AGG_SQL = "SELECT dept_id, COUNT(*) FROM student GROUP BY dept_id ORDER BY COUNT(*) DESC LIMIT 1"


def make_db() -> Database:
    db = Database.from_spider_json(TABLES_JSON)
    db.rows = {k: [dict(r) for r in v] for k, v in ROWS.items()}
    return db


class FormatCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = scratch_dir("format_test")
        cls.db = make_db()
        cls.sqlite_dir = cls.root / "databases"
        cls.sqlite_dir.mkdir(parents=True, exist_ok=True)
        cls.db_path = build_sqlite(cls.db, cls.sqlite_dir / "toy.sqlite")
        cls.spider = SpiderData(cls.root, {"toy": cls.db}, [], [], cls.sqlite_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def cfg(self, **overrides: Any) -> Config:
        cfg = load_config("configs/demo.yaml", root=".") if Path("configs/demo.yaml").exists() \
            else Config()
        for key, value in overrides.items():
            section, field = key.split(".", 1)
            setattr(getattr(cfg, section), field, value)
        cfg.paths.root = str(self.root)
        return cfg

    def synth(self, kind: str = "cot", sql: str = GOOD_SQL, **kw: Any) -> SynthRecord:
        base = dict(id=f"synth_{kind}", source_id="src_1", db_id="toy", kind=kind,
                    question="Which students are in CS?", sql=sql,
                    reasoning="", difficulty=0.3, prompt_tokens=120,
                    completion_tokens=40)
        base.update(kw)
        return SynthRecord(**base)

    def validated(self, rec: SynthRecord, *, passed: bool = True,
                  score: float = 0.9) -> ValidatedRecord:
        report = ValidationReport(sql_parses=True, schema_refs_ok=True, executes=True,
                                  semantic_checked=True, semantic_score=score,
                                  semantic_verdict="consistent", passed=passed)
        return ValidatedRecord(record=rec, report=report)


# --------------------------------------------------------------------------- #
# tool schema
# --------------------------------------------------------------------------- #
class TestToolSchema(FormatCase):
    def test_openai_tools_body(self) -> None:
        body = tool_schema(self.cfg())
        self.assertIsInstance(body, dict)
        self.assertIn("tools", body)
        tool = body["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "execute_sql")
        params = tool["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["required"], ["sql"])
        self.assertEqual(params["properties"]["sql"]["type"], "string")
        # must survive a JSON round-trip (it is sent to an API / written to disk)
        self.assertEqual(json.loads(json.dumps(body)), body)

    def test_tool_definitions_is_the_list_form(self) -> None:
        defs = tool_definitions(self.cfg())
        self.assertIsInstance(defs, list)
        self.assertEqual(defs, tool_schema(self.cfg())["tools"])

    def test_tool_name_is_configurable(self) -> None:
        cfg = self.cfg(**{"fmt.tool_name": "run_sql"})
        self.assertEqual(tool_schema(cfg)["tools"][0]["function"]["name"], "run_sql")
        msg = build_assistant_message(cfg, GOOD_SQL, kind="raw", source_id="s")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "run_sql")


# --------------------------------------------------------------------------- #
# assistant target: the product contract
# --------------------------------------------------------------------------- #
class TestAssistantTarget(FormatCase):
    def test_arguments_is_a_json_string_not_an_object(self) -> None:
        msg = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s1")
        self.assertEqual(msg["role"], "assistant")
        fn = msg["tool_calls"][0]["function"]
        self.assertIsInstance(fn["arguments"], str)
        self.assertEqual(json.loads(fn["arguments"]), {"sql": GOOD_SQL})

    def test_sql_round_trips_through_extract_sql(self) -> None:
        for sql in (GOOD_SQL, JOIN_SQL, AGG_SQL):
            msg = build_assistant_message(self.cfg(), sql, kind="raw", source_id="s")
            inner = json.loads(msg["tool_calls"][0]["function"]["arguments"])["sql"]
            self.assertEqual(extract_sql(inner), inner)

    def test_call_id_is_deterministic(self) -> None:
        a = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s1")
        b = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s1")
        c = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s2")
        self.assertEqual(a["tool_calls"][0]["id"], b["tool_calls"][0]["id"])
        self.assertNotEqual(a["tool_calls"][0]["id"], c["tool_calls"][0]["id"])

    def test_fenced_sql_is_cleaned_before_embedding(self) -> None:
        msg = build_assistant_message(self.cfg(), f"```sql\n{GOOD_SQL}\n```", kind="raw")
        inner = json.loads(msg["tool_calls"][0]["function"]["arguments"])["sql"]
        self.assertEqual(inner, GOOD_SQL)

    def test_validate_target_accepts_a_correct_call(self) -> None:
        msg = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s")
        self.assertEqual(validate_target(msg, GOOD_SQL, "qwen"), [])

    def test_validate_target_rejects_an_object_arguments_blob(self) -> None:
        broken = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "execute_sql", "arguments": {"sql": GOOD_SQL}}}]}
        problems = validate_target(broken, GOOD_SQL, "qwen")
        self.assertTrue(any("expected JSON string" in p for p in problems))


# --------------------------------------------------------------------------- #
# rendering: all three templates
# --------------------------------------------------------------------------- #
class TestTemplates(FormatCase):
    def _record(self) -> Any:
        recs, _ = build_sft_records(self.cfg(), [self.validated(self.synth("cot"))],
                                    self.spider, [])
        return recs[0]

    def test_all_templates_render(self) -> None:
        rec = self._record()
        for tpl in TEMPLATES:
            text = render_messages(rec.messages, tpl)
            self.assertIsInstance(text, str)
            self.assertIn(GOOD_SQL, text)
            self.assertIn("Which students are in CS?", text)

    def test_prompt_plus_target_equals_full_render(self) -> None:
        rec = self._record()
        for tpl in TEMPLATES:
            prompt = render_prompt(rec.messages[:-1], tpl)
            target = render_target_text(rec, tpl)
            self.assertEqual(prompt + target, render_messages(rec.messages, tpl),
                             f"template {tpl} drifts between prompt and target rendering")

    def test_supervised_pair_is_a_literal_suffix_split(self) -> None:
        rec = self._record()
        for tpl in TEMPLATES:
            prompt, target = render_supervised_pair(rec, tpl)
            whole = render_messages(rec.messages, tpl)
            self.assertTrue(whole.startswith(prompt))
            self.assertTrue(whole.endswith(target))
            self.assertEqual(len(prompt) + len(target), len(whole))

    def test_template_special_tokens(self) -> None:
        rec = self._record()
        qwen = render_messages(rec.messages, "qwen")
        self.assertIn("<|im_start|>system\n", qwen)
        self.assertIn("<|im_start|>assistant\n", qwen)
        self.assertTrue(qwen.endswith("<|im_end|>\n"))
        llama = render_messages(rec.messages, "llama3")
        self.assertTrue(llama.startswith("<|begin_of_text|>"))
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\n", llama)
        self.assertIn("<|eot_id|>", llama)
        plain = render_messages(rec.messages, "plain")
        self.assertNotIn("<|", plain)
        self.assertIn("### System:", plain)
        self.assertIn("### Assistant:", plain)

    def test_prompt_ends_with_generation_prefix(self) -> None:
        rec = self._record()
        self.assertTrue(render_prompt(rec.messages[:-1], "qwen").endswith("<|im_start|>assistant\n"))
        self.assertTrue(render_prompt(rec.messages[:-1], "llama3").endswith(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"))
        self.assertTrue(render_prompt(rec.messages[:-1], "plain").endswith("### Assistant:\n"))

    def test_target_parses_back_to_the_same_sql(self) -> None:
        rec = self._record()
        for tpl in TEMPLATES:
            body = render_target_text(rec, tpl)
            parsed = parse_tool_call(body)
            self.assertIsNotNone(parsed, f"template {tpl} target is not parseable")
            self.assertEqual(normalize_sql(parsed["sql"]), normalize_sql(GOOD_SQL))
            self.assertEqual(parsed["name"], "execute_sql")

    def test_qwen_uses_the_native_tool_call_wrapper(self) -> None:
        rec = self._record()
        text = render_target_text(rec, "qwen")
        self.assertIn("<tool_call>", text)
        self.assertIn("</tool_call>", text)
        self.assertIn('"arguments": {"sql":', text)

    def test_llama3_uses_the_python_tag_and_parameters_key(self) -> None:
        rec = self._record()
        text = render_target_text(rec, "llama3")
        self.assertIn("<|python_tag|>", text)
        self.assertIn('"parameters"', text)
        self.assertIn('"name": "execute_sql"', text)

    def test_plain_renders_the_openai_wire_message(self) -> None:
        rec = self._record()
        text = render_target_text(rec, "plain").strip()
        payload = json.loads(text)
        self.assertIn("tool_calls", payload)
        self.assertIsInstance(payload["tool_calls"][0]["function"]["arguments"], str)

    def test_unknown_template_is_rejected(self) -> None:
        rec = self._record()
        with self.assertRaises(ValueError):
            render_messages(rec.messages, "chatml-typo")

    def test_render_target_rejects_non_assistant_tail(self) -> None:
        with self.assertRaises(ValueError):
            render_target_text([{"role": "user", "content": "hi"}], "qwen")


# --------------------------------------------------------------------------- #
# parsing model output
# --------------------------------------------------------------------------- #
class TestParseToolCall(FormatCase):
    def test_parses_the_wire_message(self) -> None:
        msg = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s")
        parsed = parse_tool_call(msg)
        self.assertEqual(parsed["sql"], GOOD_SQL)
        self.assertEqual(parsed["arguments"], {"sql": GOOD_SQL})
        self.assertTrue(parsed["call_id"])

    def test_parses_qwen_text(self) -> None:
        text = ('I will query the student table.\n<tool_call>\n'
                '{"name": "execute_sql", "arguments": {"sql": "%s"}}\n</tool_call>'
                % GOOD_SQL)
        self.assertEqual(parse_tool_call(text)["sql"], GOOD_SQL)

    def test_parses_llama3_text(self) -> None:
        text = '<|python_tag|>{"name": "execute_sql", "parameters": {"sql": "%s"}}' % GOOD_SQL
        self.assertEqual(parse_tool_call(text)["sql"], GOOD_SQL)

    def test_parses_bare_json_object(self) -> None:
        self.assertEqual(parse_tool_call('{"sql": "%s"}' % GOOD_SQL)["sql"], GOOD_SQL)

    def test_parses_a_json_encoded_arguments_string(self) -> None:
        text = json.dumps({"name": "execute_sql",
                           "arguments": json.dumps({"sql": GOOD_SQL})})
        self.assertEqual(parse_tool_call(text)["sql"], GOOD_SQL)

    def test_returns_none_for_prose(self) -> None:
        self.assertIsNone(parse_tool_call("The answer is 2 students."))
        self.assertIsNone(parse_tool_call({"role": "assistant", "content": "no call"}))
        self.assertIsNone(parse_tool_call(""))

    def test_assistant_sql_falls_back_to_prose_extraction(self) -> None:
        self.assertEqual(assistant_sql({"role": "assistant",
                                        "content": f"```sql\n{GOOD_SQL}\n```"}), GOOD_SQL)
        msg = build_assistant_message(self.cfg(), GOOD_SQL, kind="raw", source_id="s")
        self.assertEqual(assistant_sql(msg), GOOD_SQL)


# --------------------------------------------------------------------------- #
# CoT reasoning hygiene
# --------------------------------------------------------------------------- #
class TestReasoning(FormatCase):
    def test_verbatim_sql_is_stripped_from_reasoning(self) -> None:
        reasoning = f"First I look at the student table. The query is {GOOD_SQL}. Done."
        cleaned = sanitize_reasoning(reasoning, GOOD_SQL)
        self.assertNotIn(GOOD_SQL.lower(), cleaned.lower())
        self.assertIn("student table", cleaned)

    def test_fenced_sql_block_is_stripped(self) -> None:
        cleaned = sanitize_reasoning(f"Plan:\n```sql\n{GOOD_SQL}\n```\nThen run it.", GOOD_SQL)
        self.assertNotIn("select", cleaned.lower())

    def test_prose_sql_sentence_is_dropped(self) -> None:
        cleaned = sanitize_reasoning(
            "I need the CS students. I will run SELECT stu_name FROM student WHERE dept_id = 1.",
            GOOD_SQL)
        self.assertNotIn("select stu_name", cleaned.lower())

    def test_natural_plan_survives(self) -> None:
        reasoning = "I should read the student rows and keep only the CS department."
        self.assertEqual(sanitize_reasoning(reasoning, GOOD_SQL), reasoning)

    def test_derive_plan_is_grounded_in_the_query_structure(self) -> None:
        plan = derive_plan(JOIN_SQL, self.db)
        self.assertIn("student", plan)
        self.assertIn("department", plan)
        self.assertIn("join", plan.lower())
        self.assertNotIn("select", plan.lower())

    def test_cot_target_never_contains_the_sql_in_the_reasoning_span(self) -> None:
        rec = self.synth("cot", reasoning=f"The answer is {GOOD_SQL} exactly.")
        recs, stats = build_sft_records(self.cfg(), [self.validated(rec)], self.spider, [])
        content = recs[0].messages[-1]["content"]
        self.assertTrue(content, "the CoT span must not be empty")
        self.assertNotIn(normalize_sql(GOOD_SQL), normalize_sql(content))
        self.assertTrue(stats["by_kind"]["cot"]["n"] == 1)

    def test_raw_and_paraphrase_have_no_reasoning(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Which students are in CS?",
                      sql=GOOD_SQL, split="train")
        recs, _ = build_sft_records(self.cfg(), [], self.spider, [raw])
        self.assertEqual(recs[0].messages[-1]["content"], "")


# --------------------------------------------------------------------------- #
# per-kind curricula
# --------------------------------------------------------------------------- #
class TestKinds(FormatCase):
    def _one(self, rec: SynthRecord, raw: list[Example] | None = None):
        recs, stats = build_sft_records(self.cfg(), [self.validated(rec)], self.spider,
                                       raw or [])
        return recs, stats

    def test_cot_assistant_has_reasoning_then_call(self) -> None:
        rec = self.synth("cot", reasoning="Read student rows and filter by department.")
        recs, _ = self._one(rec)
        assistant = recs[0].messages[-1]
        self.assertTrue(assistant["content"].strip())
        self.assertTrue(assistant["tool_calls"])
        self.assertEqual(recs[0].kind, "cot")
        self.assertEqual(recs[0].target_sql, GOOD_SQL)

    def test_paraphrase_uses_the_rewritten_question(self) -> None:
        rec = self.synth("paraphrase", question="Which pupils belong to the CS department?",
                         original_question="Which students are in CS?")
        recs, _ = self._one(rec)
        self.assertIn("pupils", recs[0].messages[1]["content"])
        self.assertEqual(recs[0].messages[-1]["content"], "")

    def test_hard_negative_user_turn_shows_the_bad_sql_and_the_real_symptom(self) -> None:
        rec = self.synth("hard_negative", bad_sql="SELECT nickname FROM student",
                         error_kind="wrong_column",
                         correction_note="nickname is not a column of student")
        recs, _ = self._one(rec)
        user = recs[0].messages[1]["content"]
        self.assertIn("SELECT nickname FROM student", user)
        self.assertIn("SQLite rejected it with", user)
        self.assertIn("nickname is not a column", user)
        self.assertIn("corrected", user.lower())
        # the target is the gold SQL, never the bad one
        target = json.loads(recs[0].messages[-1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(target["sql"], GOOD_SQL)
        self.assertEqual(recs[0].meta["error_kind"], "wrong_column")

    def test_hard_negative_reports_wrong_rows_when_the_bad_sql_executes(self) -> None:
        rec = self.synth("hard_negative", bad_sql="SELECT stu_name FROM student WHERE dept_id = 2",
                         error_kind="wrong_literal")
        recs, _ = self._one(rec)
        self.assertIn("wrong rows", recs[0].messages[1]["content"])

    def test_raw_example_becomes_a_plain_call(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Which students are in CS?",
                      sql=GOOD_SQL, split="train", difficulty=0.25)
        recs, _ = self._one(self.synth("cot"), [raw])
        raw_rec = [r for r in recs if r.kind == "raw"][0]
        self.assertEqual(raw_rec.synth_tokens, 0)
        self.assertEqual(raw_rec.difficulty, 0.25)
        self.assertEqual(raw_rec.weight, 1.0)
        self.assertEqual(raw_rec.messages[-1]["content"], "")
        self.assertIn("Which students are in CS?", raw_rec.messages[1]["content"])


# --------------------------------------------------------------------------- #
# corpus-level behaviour
# --------------------------------------------------------------------------- #
class TestBuildSftRecords(FormatCase):
    def _corpus(self, cfg: Config | None = None):
        cfg = cfg or self.cfg()
        records = [
            self.validated(self.synth("cot", id="s_cot", reasoning="Read rows, filter CS.")),
            self.validated(self.synth("paraphrase", id="s_para",
                                      question="Which pupils study in CS?")),
            self.validated(self.synth("hard_negative", id="s_hn",
                                      bad_sql="SELECT nickname FROM student",
                                      error_kind="wrong_column")),
        ]
        raw = [Example(id="e1", db_id="toy", question="Which students are in CS?",
                       sql=GOOD_SQL, split="train"),
               Example(id="e2", db_id="toy", question="List all departments.",
                       sql="SELECT dept_name FROM department", split="dev")]
        return build_sft_records(cfg, records, self.spider, raw)

    def test_counts_and_kinds(self) -> None:
        records, stats = self._corpus()
        self.assertEqual(len(records), 5)
        self.assertEqual(stats["n_records"], 5)
        self.assertEqual(stats["n_train"], 4)
        self.assertEqual(stats["n_dev"], 1)
        self.assertEqual(set(stats["kinds"]), {"cot", "paraphrase", "hard_negative", "raw"})
        self.assertEqual(stats["by_kind"]["raw"]["n"], 2)

    def test_hard_negative_weight_is_applied(self) -> None:
        records, stats = self._corpus()
        hn = [r for r in records if r.kind == "hard_negative"][0]
        self.assertGreater(hn.weight, 1.0)
        self.assertEqual(stats["by_kind"]["hard_negative"]["weight_sum"], hn.weight)
        self.assertEqual([r.weight for r in records if r.kind == "raw"], [1.0, 1.0])

    def test_weight_is_configurable(self) -> None:
        cfg = self.cfg()
        cfg.fmt.hard_negative_weight = 3.0
        records, _ = self._corpus(cfg)
        self.assertEqual([r.weight for r in records if r.kind == "hard_negative"], [3.0])

    def test_synth_tokens_come_from_the_synth_record(self) -> None:
        records, stats = self._corpus()
        cot = [r for r in records if r.kind == "cot"][0]
        self.assertEqual(cot.synth_tokens, 160)     # 120 prompt + 40 completion
        self.assertEqual(stats["synth_tokens_total"], 160 * 3)
        self.assertEqual([r.synth_tokens for r in records if r.kind == "raw"], [0, 0])

    def test_every_target_is_a_valid_tool_call(self) -> None:
        records, stats = self._corpus()
        self.assertEqual(stats["tool_call_valid_rate"], 1.0)
        self.assertEqual(stats["target_problems"], {})
        for rec in records:
            fn = rec.messages[-1]["tool_calls"][0]["function"]
            self.assertIsInstance(fn["arguments"], str)
            sql = json.loads(fn["arguments"])["sql"]
            self.assertEqual(extract_sql(sql), sql)
            self.assertEqual(normalize_sql(sql), normalize_sql(rec.target_sql))

    def test_tools_are_attached_to_every_record(self) -> None:
        records, _ = self._corpus()
        for rec in records:
            self.assertEqual(len(rec.tools), 1)
            self.assertEqual(rec.tools[0]["function"]["name"], "execute_sql")

    def test_system_prompt_carries_ddl_and_tool_definition(self) -> None:
        records, _ = self._corpus()
        system = records[0].messages[0]
        self.assertEqual(system["role"], "system")
        self.assertIn("CREATE TABLE student", system["content"])
        self.assertIn("execute_sql", system["content"])

    def test_schema_truncation_is_honoured(self) -> None:
        cfg = self.cfg(**{"fmt.max_schema_tables": 1})
        records, _ = self._corpus(cfg)
        system = records[0].messages[0]["content"]
        self.assertIn("CREATE TABLE student", system)
        self.assertNotIn("CREATE TABLE department", system)

    def test_build_is_deterministic(self) -> None:
        a, _ = self._corpus()
        b, _ = self._corpus()
        self.assertEqual([r.as_dict() for r in a], [r.as_dict() for r in b])

    def test_duplicate_source_ids_do_not_collide(self) -> None:
        rec = self.synth("cot", id="dup")
        records, _ = build_sft_records(self.cfg(), [self.validated(rec),
                                                    self.validated(rec)], self.spider, [])
        self.assertEqual(len({r.id for r in records}), 2)

    def test_unknown_template_raises(self) -> None:
        cfg = self.cfg(**{"fmt.chat_template": "gpt2"})
        with self.assertRaises(ValueError):
            build_sft_records(cfg, [], self.spider, [])


class TestBuildAndWrite(FormatCase):
    def test_writes_artifacts(self) -> None:
        out_dir = scratch_dir("format_write")
        try:
            cfg = self.cfg()
            records = [self.validated(self.synth("cot", reasoning="Read rows."))]
            raw = [Example(id="e1", db_id="toy", question="Which students are in CS?",
                           sql=GOOD_SQL, split="train"),
                   Example(id="e2", db_id="toy", question="List all departments.",
                           sql="SELECT dept_name FROM department", split="dev")]
            stats = build_and_write(cfg, records, self.spider, raw, out_dir)
            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "dev.jsonl").exists())
            self.assertTrue((out_dir / "tool_schema.json").exists())
            self.assertTrue((out_dir / "format_stats.json").exists())
            train = read_sft(out_dir / "train.jsonl")
            dev = read_sft(out_dir / "dev.jsonl")
            self.assertEqual(len(train), 2)
            self.assertEqual(len(dev), 1)
            self.assertEqual(stats["written"]["train"], 2)
            # the written tool schema is the OpenAI body, not a bare list
            schema = json.loads((out_dir / "tool_schema.json").read_text(encoding="utf-8"))
            self.assertIn("tools", schema)
            # round-tripped records keep the tool-call target intact
            fn = train[0].messages[-1]["tool_calls"][0]["function"]
            self.assertIsInstance(fn["arguments"], str)
            self.assertEqual(extract_sql(json.loads(fn["arguments"])["sql"]),
                             json.loads(fn["arguments"])["sql"])
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# inference prompt (shared with fcdata.eval -- must not drift)
# --------------------------------------------------------------------------- #
class TestEvalPrompt(FormatCase):
    def test_prompt_matches_the_training_prompt_prefix(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Which students are in CS?",
                      sql=GOOD_SQL, split="dev")
        cfg = self.cfg()
        records, _ = build_sft_records(cfg, [], self.spider, [raw])
        train_prompt = render_prompt(records[0].messages[:-1],
                                     cfg.fmt.chat_template)
        eval_prompt = build_eval_prompt(cfg, raw, self.db)
        self.assertEqual(train_prompt, eval_prompt,
                         "training and inference prompts must be byte-identical")

    def test_messages_have_no_assistant_turn(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Which students are in CS?",
                      sql=GOOD_SQL, split="dev")
        msgs = build_eval_messages(self.cfg(), raw, self.db)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("CREATE TABLE student", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "Which students are in CS?")

    def test_works_for_all_templates_and_resolves_db_from_spider_data(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Which students are in CS?",
                      sql=GOOD_SQL, split="dev")
        for tpl in TEMPLATES:
            prompt = build_eval_prompt(self.cfg(), raw, spider_data=self.spider,
                                       template=tpl)
            self.assertTrue(prompt.endswith(("assistant\n",
                                             "assistant<|end_header_id|>\n\n",
                                             "### Assistant:\n")), tpl)

    def test_works_without_a_database(self) -> None:
        raw = Example(id="e1", db_id="toy", question="Q?", sql=GOOD_SQL, split="dev")
        prompt = build_eval_prompt(self.cfg(), raw, None)
        self.assertIn("Q?", prompt)
        self.assertNotIn("CREATE TABLE", prompt)


class TestAblationArms(FormatCase):
    """The ablation runner filters SFT records by kind -- that must be exact.

    ``raw_only`` / ``plus_cot`` / ``plus_hard_negatives`` differ only in which
    synthetic kinds survive, so a filtering bug would silently invalidate the
    whole comparison table.
    """

    def _corpus(self):
        records = [
            self.validated(self.synth("cot", id="c1", reasoning="Read rows.")),
            self.validated(self.synth("paraphrase", id="p1",
                                      question="Which pupils study in CS?")),
            self.validated(self.synth("hard_negative", id="h1",
                                      bad_sql="SELECT nickname FROM student",
                                      error_kind="wrong_column")),
        ]
        raw = [Example(id="e1", db_id="toy", question="Which students are in CS?",
                       sql=GOOD_SQL, split="train"),
               Example(id="e2", db_id="toy", question="List all departments.",
                       sql="SELECT dept_name FROM department", split="dev")]
        return build_sft_records(self.cfg(), records, self.spider, raw)[0]

    def test_raw_only_arm(self) -> None:
        corpus = self._corpus()
        arm = [r for r in corpus if r.kind == "raw"]
        self.assertEqual(len(arm), 2)
        self.assertTrue(all(r.synth_tokens == 0 for r in arm))

    def test_plus_cot_arm_adds_only_cot(self) -> None:
        corpus = self._corpus()
        arm = [r for r in corpus if r.kind in {"raw", "cot"}]
        self.assertEqual(sorted(r.kind for r in arm), ["cot", "raw", "raw"])

    def test_plus_hard_negatives_arm_adds_everything(self) -> None:
        corpus = self._corpus()
        arm = [r for r in corpus if r.kind in {"raw", "cot", "paraphrase", "hard_negative"}]
        self.assertEqual(len(arm), len(corpus))
        kinds = sorted({r.kind for r in arm})
        self.assertEqual(kinds, ["cot", "hard_negative", "paraphrase", "raw"])

    def test_each_arm_has_a_valid_target(self) -> None:
        for rec in self._corpus():
            self.assertEqual(validate_target(rec.messages[-1], rec.target_sql,
                                             self.cfg().fmt.chat_template), [],
                             f"{rec.kind} target is malformed")


class TestRobustness(FormatCase):
    def test_record_with_unknown_database_still_builds(self) -> None:
        # the schema is missing, so there is no DDL -- but the sample must not crash
        rec = self.synth("cot", db_id="not_in_fixture", reasoning="Read rows.")
        records, stats = build_sft_records(self.cfg(), [self.validated(rec)], self.spider, [])
        self.assertEqual(len(records), 1)
        self.assertNotIn("CREATE TABLE", records[0].messages[0]["content"])
        self.assertEqual(stats["tool_call_valid_rate"], 1.0)

    def test_multiple_tool_calls_in_one_message_uses_the_first(self) -> None:
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "execute_sql", "arguments": json.dumps({"sql": GOOD_SQL})}},
            {"id": "b", "type": "function",
             "function": {"name": "execute_sql", "arguments": json.dumps({"sql": AGG_SQL})}},
        ]}
        self.assertEqual(parse_tool_call(msg)["sql"], GOOD_SQL)
        self.assertEqual(parse_tool_call(msg)["call_id"], "a")

    def test_malformed_arguments_string_falls_back_to_raw_sql(self) -> None:
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "execute_sql", "arguments": GOOD_SQL}}]}
        parsed = parse_tool_call(msg)
        self.assertEqual(parsed["sql"], GOOD_SQL)

    def test_validate_target_detects_sql_mismatch(self) -> None:
        msg = build_assistant_message(self.cfg(), AGG_SQL, kind="raw", source_id="s")
        problems = validate_target(msg, GOOD_SQL, "qwen")
        self.assertTrue(any("differs from target_sql" in p for p in problems))

    def test_validate_target_detects_missing_tool_calls(self) -> None:
        problems = validate_target({"role": "assistant", "content": "SELECT 1"},
                                   GOOD_SQL, "qwen")
        self.assertIn("no tool_calls", problems)

    def test_validate_target_detects_invalid_json_arguments(self) -> None:
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "execute_sql", "arguments": "{not json"}}]}
        problems = validate_target(msg, GOOD_SQL, "qwen")
        self.assertTrue(any("not valid JSON" in p for p in problems))

    def test_corpus_with_only_raw_examples(self) -> None:
        raw = [Example(id="e1", db_id="toy", question="Q?", sql=GOOD_SQL, split="train")]
        records, stats = build_sft_records(self.cfg(), [], self.spider, raw)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats["kinds"], ["raw"])
        self.assertEqual(stats["synth_tokens_total"], 0)

    def test_empty_corpus_is_safe(self) -> None:
        records, stats = build_sft_records(self.cfg(), [], self.spider, [])
        self.assertEqual(records, [])
        self.assertEqual(stats["n_records"], 0)
        self.assertEqual(stats["tool_call_valid_rate"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
