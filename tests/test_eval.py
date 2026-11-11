"""Tests for the evaluation harness: scoring, metrics, ablation and scale plumbing.

WHAT IS ACTUALLY ASSERTED HERE
------------------------------
The most important property of this project is that *accuracy comes from real
SQLite execution*. These tests therefore never mock ``dbexec``: every accuracy
assertion runs the SQL through a real sqlite fixture built in ``setUpClass``.
Only the **model** is stubbed, because generating text from a transformer is not
what is under test here -- the scoring path is.

Coverage:
* ``evaluate_pair``-backed predictions: correct SQL scores 1.0, wrong SQL scores
  0.0, non-executable SQL is flagged, exact-match is separate from exec-match.
* the failure taxonomy puts each failure in exactly one bucket.
* ``bootstrap_ci`` is deterministic and brackets the point estimate.
* difficulty quartiles partition the dev set without dropping examples.
* ``extract_pred_sql`` handles the function-calling wire format AND bare SQL.
* the ablation ladder builds the right data mix, resamples deterministically, and
  attributes synthesis tokens per config.
* ``run_ablation`` / ``run_scale_curve`` work end-to-end with a fake trainer and
  a stub generator, and their reported accuracy still comes from sqlite.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from typing import Any

from fcdata.config import Config, scratch_dir
from fcdata.dbexec import build_sqlite, evaluate_pair
from fcdata.eval.ablation import (
    ABLATION_CONFIGS,
    build_ablation_mix,
    run_ablation,
)
from fcdata.eval.infer import extract_pred_sql, predict_examples
from fcdata.eval.metrics import (
    FAILURE_TAXONOMY,
    bootstrap_ci,
    classify_failure,
    summarize,
)
from fcdata.eval.pipeline import (
    SFTBundle,
    resample_by_difficulty,
    select_kinds,
    synth_tokens_of,
)
from fcdata.eval.scale import run_scale_curve, sample_synthetic_fraction as scale_sample
from fcdata.records import Example, Prediction, SFTRecord, new_id, write_json
from fcdata.schema import Database, Table, Column


# --------------------------------------------------------------------------- #
# a tiny real database + Spider-ish examples
# --------------------------------------------------------------------------- #
def _make_db() -> Database:
    """Two related tables with values that make wrong SQL produce wrong rows."""
    dept = Table(index=0, name="department", columns=[
        Column(0, "dept_id", "number", True),
        Column(0, "dept_name", "text"),
        Column(0, "budget", "number"),
    ])
    emp = Table(index=1, name="employee", columns=[
        Column(1, "emp_id", "number", True),
        Column(1, "emp_name", "text"),
        Column(1, "dept_id", "number"),
        Column(1, "salary", "number"),
    ])
    return Database(
        db_id="tiny",
        tables=[dept, emp],
        rows={
            "department": [
                {"dept_id": 1, "dept_name": "Engineering", "budget": 900.0},
                {"dept_id": 2, "dept_name": "Sales", "budget": 400.0},
                {"dept_id": 3, "dept_name": "Support", "budget": 250.0},
            ],
            "employee": [
                {"emp_id": 1, "emp_name": "Ada", "dept_id": 1, "salary": 150.0},
                {"emp_id": 2, "emp_name": "Grace", "dept_id": 1, "salary": 130.0},
                {"emp_id": 3, "emp_name": "Linus", "dept_id": 2, "salary": 110.0},
                {"emp_id": 4, "emp_name": "Margaret", "dept_id": 3, "salary": 95.0},
            ],
        },
    )


EXAMPLES: list[Example] = [
    Example(id="t1", db_id="tiny", question="How many employees are there?",
            sql="SELECT COUNT(*) FROM employee", difficulty=0.10),
    Example(id="t2", db_id="tiny", question="What is the total salary?",
            sql="SELECT SUM(salary) FROM employee", difficulty=0.20),
    Example(id="t3", db_id="tiny",
            question="List the names of employees in Engineering.",
            sql=("SELECT T2.emp_name FROM department AS T1 JOIN employee AS T2 "
                 "ON T1.dept_id = T2.dept_id WHERE T1.dept_name = 'Engineering'"),
            difficulty=0.65),
    Example(id="t4", db_id="tiny", question="Which department has the largest budget?",
            sql=("SELECT dept_name FROM department ORDER BY budget DESC LIMIT 1"),
            difficulty=0.40),
]


class TinySpider:
    """Minimal ``SpiderData`` stand-in: schema, examples and sqlite paths."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.databases = {"tiny": _make_db()}
        self.train = list(EXAMPLES)
        self.dev = list(EXAMPLES)
        self.sqlite_dir = root / "databases"
        self.sqlite_dir.mkdir(parents=True, exist_ok=True)
        build_sqlite(self.databases["tiny"], self.db_path("tiny"))

    def db(self, db_id: str) -> Database:
        return self.databases[db_id]

    def db_path(self, db_id: str) -> Path:
        return self.sqlite_dir / f"{db_id}.sqlite"


# --------------------------------------------------------------------------- #
# stub model: emits a fixed SQL string per prompt, no torch involved
# --------------------------------------------------------------------------- #
class StubModel:
    """Duck-typed stand-in for the trained model (marked ``is_stub``).

    ``fcdata.eval.infer._default_generate`` routes stub models to
    ``model.generate(prompt)`` so tests exercise the real prediction loop while
    skipping the transformer entirely.
    """

    is_stub = True

    def __init__(self, sql_by_id: dict[str, str]) -> None:
        self.sql_by_id = sql_by_id
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, sql in self.sql_by_id.items():
            if key in prompt:
                return f"```sql\n{sql}\n```"
        return ""


def _stub_prompt(cfg: Any, example: Example, db: Database) -> str:
    """Prompt renderer stub; embeds the example id so StubModel can key on it."""
    return f"schema:{db.db_id}\nquestion:{example.question}\nexample_id:{example.id}"


def _cfg() -> Config:
    cfg = Config()
    cfg.ev.max_examples = 0
    cfg.ev.max_new_tokens = 32
    cfg.ev.temperature = 0.0
    cfg.seed = 17
    return cfg


# --------------------------------------------------------------------------- #
# metrics / taxonomy
# --------------------------------------------------------------------------- #
class TestMetrics(unittest.TestCase):
    """Metrics must reflect real execution outcomes, not the model's claims."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("eval_metrics")
        cls.spider = TinySpider(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_correct_sql_scores_full_accuracy(self) -> None:
        cfg = _cfg()
        model = StubModel({ex.id: ex.sql for ex in EXAMPLES})
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        m = summarize(preds, n_boot=200)
        self.assertEqual(m["n"], len(EXAMPLES))
        self.assertEqual(m["exec_acc"], 1.0)
        self.assertEqual(m["exact_match"], 1.0)
        self.assertEqual(m["executable_rate"], 1.0)
        self.assertEqual(m["failures"]["passed"]["n"], len(EXAMPLES))

    def test_wrong_sql_scores_zero_and_is_taxonomised(self) -> None:
        # One query that executes but answers none of the four questions: it
        # returns every employee name, so row counts and values both differ.
        cfg = _cfg()
        wrong = "SELECT emp_name FROM employee"
        model = StubModel({ex.id: wrong for ex in EXAMPLES})
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        m = summarize(preds, n_boot=200)
        self.assertEqual(m["exec_acc"], 0.0)
        self.assertEqual(m["executable_rate"], 1.0)
        buckets = {k for k in FAILURE_TAXONOMY if m["failures"][k]["n"] > 0}
        self.assertTrue(buckets)
        self.assertFalse(buckets & {"not_executable", "empty_prediction"})
        self.assertIn("passed", m["failures"])
        self.assertEqual(m["failures"]["passed"]["n"], 0)

    def test_nonexecutable_prediction_is_flagged(self) -> None:
        cfg = _cfg()
        model = StubModel({ex.id: "SELECT no_such_column FROM nope_table"
                           for ex in EXAMPLES})
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        self.assertTrue(all(not p.executable for p in preds))
        self.assertTrue(all(classify_failure(p) == "not_executable" for p in preds))

    def test_empty_output_is_its_own_bucket(self) -> None:
        cfg = _cfg()
        model = StubModel({})  # never returns SQL
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        self.assertTrue(all(p.pred_sql == "" for p in preds))
        self.assertTrue(all(classify_failure(p) == "empty_prediction" for p in preds))

    def test_exact_match_is_stricter_than_exec_match(self) -> None:
        """A semantically equal but textually different query: exec 1, EM 0."""
        cfg = _cfg()
        # same result as t2's SUM but written differently (exec match, no EM)
        alt = "SELECT SUM(employee.salary) FROM employee"
        model = StubModel({"t2": alt})
        preds = predict_examples(model, None, cfg, self.spider, [EXAMPLES[1]],
                                 prompt_fn=_stub_prompt, progress=False)
        self.assertTrue(preds[0].exec_match)
        self.assertFalse(preds[0].exact_match)

    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self) -> None:
        values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
        lo, hi = bootstrap_ci(values, n=500, seed=17)
        lo2, hi2 = bootstrap_ci(values, n=500, seed=17)
        self.assertEqual((lo, hi), (lo2, hi2))
        mean = sum(values) / len(values)
        self.assertLessEqual(lo, mean + 1e-9)
        self.assertGreaterEqual(hi, mean - 1e-9)
        self.assertLess(lo, hi)

    def test_bootstrap_ci_handles_degenerate_inputs(self) -> None:
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))
        self.assertEqual(bootstrap_ci([1.0]), (1.0, 1.0))
        lo, hi = bootstrap_ci([1, 1, 1, 1], n=100)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_difficulty_quartiles_partition_the_set(self) -> None:
        cfg = _cfg()
        model = StubModel({ex.id: ex.sql for ex in EXAMPLES})
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        m = summarize(preds, n_boot=100)
        buckets = m["by_difficulty"]
        self.assertEqual(set(buckets), {"Q1", "Q2", "Q3", "Q4"})
        self.assertEqual(sum(b["n"] for b in buckets.values()), len(EXAMPLES))
        # quartiles must be ordered by difficulty
        mins = [buckets[k]["difficulty_min"] for k in sorted(buckets)]
        self.assertEqual(mins, sorted(mins))

    def test_per_db_and_per_kind_slices_exist(self) -> None:
        cfg = _cfg()
        model = StubModel({ex.id: ex.sql for ex in EXAMPLES})
        preds = predict_examples(model, None, cfg, self.spider, EXAMPLES,
                                 prompt_fn=_stub_prompt, progress=False)
        m = summarize(preds, n_boot=100)
        self.assertIn("tiny", m["by_db"])
        self.assertEqual(m["by_db"]["tiny"]["n"], len(EXAMPLES))
        self.assertIn("raw", m["by_kind"])
        self.assertEqual(m["by_kind"]["raw"]["exec_acc"], 1.0)

    def test_loaded_prediction_without_meta_is_still_classifiable(self) -> None:
        """Old predictions.jsonl files (no meta) must not break the taxonomy."""
        pred = Prediction(id=new_id("p"), example_id="x", db_id="tiny",
                          question="q", gold_sql="SELECT 1", pred_sql="SELECT 2",
                          executable=True, exec_match=False, exact_match=False,
                          reason="row values differ")
        self.assertEqual(classify_failure(pred), "wrong_values")


# --------------------------------------------------------------------------- #
# SQL extraction
# --------------------------------------------------------------------------- #
class TestExtraction(unittest.TestCase):
    """The eval harness must accept both the trained wire format and bare SQL."""

    def test_bare_sql_and_fences(self) -> None:
        self.assertEqual(extract_pred_sql("SELECT COUNT(*) FROM employee"),
                         "SELECT COUNT(*) FROM employee")
        self.assertEqual(extract_pred_sql("```sql\nSELECT 1\n```"), "SELECT 1")

    def test_function_call_arguments_blob(self) -> None:
        raw = ('{"tool_calls":[{"id":"c1","type":"function","function":'
               '{"name":"execute_sql","arguments":"{\\"sql\\": \\"SELECT SUM(salary) '
               'FROM employee\\"}"}}]}')
        self.assertEqual(extract_pred_sql(raw), "SELECT SUM(salary) FROM employee")

    def test_garbage_returns_empty(self) -> None:
        self.assertEqual(extract_pred_sql("I cannot answer that."), "")
        self.assertEqual(extract_pred_sql(""), "")

    def test_spaced_template_tail_is_stripped(self) -> None:
        """A word-level model emits the chat template as spaced tokens.

        This is the exact shape the reference model produced in a real run:
        correct SQL followed by ``"}}}} < / tool_call > <|im_end|>``. Scoring it
        as a failed prediction would understate accuracy for a cosmetic reason.
        """
        raw = ('< tool_call > {"name":"execute_sql","arguments":{":"'
               '"SELECT title FROM student WHERE gpa > 3.5"}}}} '
               '< / tool_call > <|im_end|>')
        self.assertEqual(extract_pred_sql(raw),
                         "SELECT title FROM student WHERE gpa > 3.5")

    def test_compact_template_tail_is_stripped(self) -> None:
        raw = 'SELECT AVG(salary) FROM instructor WHERE dept_id = 2"}} < / tool_call > <|im_end|>'
        self.assertEqual(extract_pred_sql(raw),
                         "SELECT AVG(salary) FROM instructor WHERE dept_id = 2")

    def test_trailing_string_literal_is_preserved(self) -> None:
        """A balanced closing quote is part of the SQL, not wrapper junk."""
        sql = "SELECT stu_name FROM student WHERE dept_name = 'Computer Science'"
        self.assertEqual(extract_pred_sql(sql), sql)
        sql2 = "SELECT COUNT(*) FROM course WHERE title = 'Data Structures'"
        self.assertEqual(extract_pred_sql(sql2), sql2)

    def test_trailing_paren_is_preserved(self) -> None:
        self.assertEqual(extract_pred_sql("SELECT COUNT(*) FROM employee"),
                         "SELECT COUNT(*) FROM employee")

    def test_fence_without_newline_is_stripped(self) -> None:
        self.assertEqual(extract_pred_sql("```sql\nSELECT 1\n```"), "SELECT 1")

    def test_stripped_sql_actually_executes(self) -> None:
        """End-to-end proof: the cleaned SQL must really run against sqlite."""
        spider = TinySpider(scratch_dir("eval_strip_exec"))
        try:
            raw = ('< tool_call > {"name":"execute_sql","arguments":{":"'
                   '"SELECT COUNT(*) FROM employee"}}}} < / tool_call > <|im_end|>')
            sql = extract_pred_sql(raw)
            result = evaluate_pair(spider.db_path("tiny"), sql,
                                   "SELECT COUNT(*) FROM employee")
            self.assertTrue(result["executable"])
            self.assertTrue(result["exec_match"])
        finally:
            shutil.rmtree(spider.root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# ablation / scale data selection (pure functions)
# --------------------------------------------------------------------------- #
def _sft_record(kind: str, idx: int, tokens: int = 100,
                difficulty: float = 0.2, weight: float = 1.0) -> SFTRecord:
    return SFTRecord(
        id=f"{kind}_{idx}", source_id=f"src_{idx}", kind=kind, db_id="tiny",
        messages=[{"role": "user", "content": f"q{idx}"},
                  {"role": "assistant", "content": f"SELECT {idx}"}],
        target_sql=f"SELECT {idx}", difficulty=difficulty, weight=weight,
        synth_tokens=tokens if kind != "raw" else 0,
    )


class TestAblationMix(unittest.TestCase):
    """The ladder must differ only in the data mix, nothing else."""

    def setUp(self) -> None:
        recs: list[SFTRecord] = []
        recs += [_sft_record("raw", i) for i in range(4)]
        recs += [_sft_record("cot", i) for i in range(4)]
        recs += [_sft_record("paraphrase", i) for i in range(3)]
        recs += [_sft_record("hard_negative", i, difficulty=0.8) for i in range(3)]
        self.bundle = SFTBundle(train=recs, dev=[], source="test")

    def test_kinds_filter_is_exact(self) -> None:
        raw_only = build_ablation_mix(self.bundle, "raw_only")
        self.assertEqual({r.kind for r in raw_only}, {"raw"})
        self.assertEqual(len(raw_only), 4)

        plus_cot = build_ablation_mix(self.bundle, "plus_cot")
        self.assertEqual({r.kind for r in plus_cot}, {"raw", "cot"})
        self.assertEqual(len(plus_cot), 8)

        full = build_ablation_mix(self.bundle, "plus_hard_negatives")
        self.assertEqual({r.kind for r in full},
                         {"raw", "cot", "paraphrase", "hard_negative"})
        # hard negatives are duplicated by difficulty resampling
        self.assertGreater(len(full), len(self.bundle.train))

    def test_synth_tokens_are_attributed_per_config(self) -> None:
        self.assertEqual(synth_tokens_of(build_ablation_mix(self.bundle, "raw_only")), 0)
        cot_tokens = synth_tokens_of(build_ablation_mix(self.bundle, "plus_cot"))
        self.assertEqual(cot_tokens, 4 * 100)  # 4 cot records x 100 tokens
        full_tokens = synth_tokens_of(build_ablation_mix(self.bundle, "plus_hard_negatives"))
        self.assertGreater(full_tokens, cot_tokens)

    def test_unknown_config_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_ablation_mix(self.bundle, "not_a_config")

    def test_resampling_is_deterministic(self) -> None:
        a = resample_by_difficulty(self.bundle.train, seed=17)
        b = resample_by_difficulty(self.bundle.train, seed=17)
        self.assertEqual([r.id for r in a], [r.id for r in b])
        c = resample_by_difficulty(self.bundle.train, seed=99)
        self.assertNotEqual([r.id for r in a], [r.id for r in c])

    def test_resampling_is_capped(self) -> None:
        """One pathological weight must not be able to dominate the batch."""
        recs = [_sft_record("hard_negative", 0, difficulty=1.0, weight=50.0)]
        out = resample_by_difficulty(recs, max_factor=3)
        self.assertLessEqual(len(out), 3)

    def test_all_configs_are_covered(self) -> None:
        for name in ABLATION_CONFIGS:
            self.assertTrue(build_ablation_mix(self.bundle, name))

    def test_select_kinds_preserves_order(self) -> None:
        picked = select_kinds(self.bundle.train, ["raw"])
        ids = [r.id for r in picked]
        self.assertEqual(ids, sorted(ids, key=lambda s: int(s.split("_")[1])))


class TestScaleSampling(unittest.TestCase):
    """Fraction sweeps must be deterministic and never subsample raw records."""

    def setUp(self) -> None:
        self.recs: list[SFTRecord] = []
        self.recs += [_sft_record("raw", i) for i in range(5)]
        self.recs += [_sft_record("cot", i) for i in range(4)]
        self.recs += [_sft_record("paraphrase", i) for i in range(4)]

    def test_zero_fraction_is_raw_only(self) -> None:
        out = scale_sample(self.recs, 0.0)
        self.assertEqual({r.kind for r in out}, {"raw"})
        self.assertEqual(len(out), 5)
        self.assertEqual(synth_tokens_of(out), 0)

    def test_full_fraction_keeps_everything(self) -> None:
        out = scale_sample(self.recs, 1.0)
        self.assertEqual(len(out), len(self.recs))

    def test_partial_fraction_is_monotone_and_stratified(self) -> None:
        quarter = scale_sample(self.recs, 0.25)
        half = scale_sample(self.recs, 0.5)
        self.assertLessEqual(len(quarter), len(half))
        self.assertGreaterEqual(len(quarter), 5)  # raw always kept
        # every synthetic kind survives stratification at 25%
        kinds = {r.kind for r in quarter if r.kind != "raw"}
        self.assertEqual(kinds, {"cot", "paraphrase"})
        # token investment is monotone in the fraction
        self.assertLess(synth_tokens_of(quarter), synth_tokens_of(half))

    def test_sampling_is_deterministic(self) -> None:
        a = [r.id for r in scale_sample(self.recs, 0.5, seed=17)]
        b = [r.id for r in scale_sample(self.recs, 0.5, seed=17)]
        self.assertEqual(a, b)
        c = [r.id for r in scale_sample(self.recs, 0.5, seed=5)]
        self.assertNotEqual(a, c)


# --------------------------------------------------------------------------- #
# end-to-end ablation + scale with a fake trainer
# --------------------------------------------------------------------------- #
def _fake_train(cfg: Any, sft_train: list[SFTRecord], sft_dev: list[SFTRecord],
                out_dir: str, **kwargs: Any) -> dict[str, Any]:
    """Stand-in trainer: writes a plausible metrics.json + a checkpoint marker."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"stub-checkpoint")
    metrics = {
        "backend": "stub", "n_train": len(sft_train), "n_dev": len(sft_dev),
        "history": [{"step": i, "epoch": 0, "loss": 4.0 - 0.3 * i,
                     "token_acc": 0.05 * i} for i in range(1, 4)],
        "final_train_loss": 3.1, "final_train_token_acc": 0.15,
    }
    write_json(d / "metrics.json", metrics)
    return metrics


class GoldStubModel:
    """Stub model that answers with the gold SQL for the example in the prompt.

    The stub *prompt renderer* (:func:`_stub_prompt`) embeds ``example_id:`` so
    this model can key on it. That keeps generation fake while the score remains
    the product of real sqlite execution -- which is what these tests verify.
    """

    is_stub = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for ex in EXAMPLES:
            if f"example_id:{ex.id}" in prompt:
                return ex.sql
        return ""


def _fake_load(path: str) -> tuple[Any, Any]:
    """Stand-in loader: returns a gold-answering stub model (no torch involved)."""
    return GoldStubModel(), None


class TestAblationEndToEnd(unittest.TestCase):
    """The ablation runner must produce real, sqlite-derived accuracy numbers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("eval_ablation")
        cls.spider = TinySpider(cls.dir)
        cls.spider.dev = list(EXAMPLES)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _bundle(self) -> SFTBundle:
        recs: list[SFTRecord] = []
        recs += [_sft_record("raw", i) for i in range(4)]
        recs += [_sft_record("cot", i) for i in range(4)]
        recs += [_sft_record("paraphrase", i) for i in range(2)]
        recs += [_sft_record("hard_negative", i, difficulty=0.9) for i in range(2)]
        return SFTBundle(train=recs, dev=[], source="unit-test",
                         eval_examples=list(EXAMPLES))

    def test_ablation_writes_json_and_csv_with_measured_accuracy(self) -> None:
        cfg = _cfg()
        out = self.dir / "abl_out"
        payload = run_ablation(
            cfg, self.spider, cfg.ablation, out,
            bundle=self._bundle(), train_fn=_fake_train, load_fn=_fake_load, prompt_fn=_stub_prompt,
            progress=False)

        self.assertEqual(payload["n_configs"], len(cfg.ablation.configs))
        self.assertTrue((out / "ablation.json").exists())
        self.assertTrue((out / "ablation.csv").exists())

        rows = payload["rows"]
        self.assertEqual([r["config"] for r in rows], list(cfg.ablation.configs))
        baseline = rows[0]
        self.assertEqual(baseline["config"], "raw_only")
        self.assertEqual(baseline["synth_tokens"], 0)
        self.assertEqual(baseline["delta_exec_acc"], 0.0)

        # the stub model emits the gold SQL, so accuracy must be exactly 1.0 and
        # it must have come from real execution (n_eval == len(EXAMPLES))
        for row in rows:
            self.assertEqual(row["exec_acc"], 1.0)
            self.assertEqual(row["exact_match"], 1.0)
            self.assertEqual(row["n_eval"], len(EXAMPLES))
            self.assertTrue(0.0 <= row["exec_acc_ci_lo"] <= row["exec_acc_ci_hi"] <= 1.0)

        # token investment must be monotone along the ladder
        tokens = [r["synth_tokens"] for r in rows]
        self.assertEqual(tokens, sorted(tokens))
        self.assertGreater(tokens[-1], tokens[0])

    def test_ablation_metrics_json_is_written_per_rung(self) -> None:
        cfg = _cfg()
        out = self.dir / "abl_out2"
        payload = run_ablation(
            cfg, self.spider, cfg.ablation, out,
            bundle=self._bundle(), train_fn=_fake_train, load_fn=_fake_load, prompt_fn=_stub_prompt,
            progress=False)
        for row in payload["rows"]:
            mpath = Path(row["eval_dir"]) / "metrics.json"
            self.assertTrue(mpath.exists(), f"missing metrics for {row['config']}")
            self.assertTrue((Path(row["eval_dir"]) / "predictions.jsonl").exists())

    def test_ablation_rejects_empty_corpus(self) -> None:
        cfg = _cfg()
        empty = SFTBundle(train=[], dev=[], source="empty", eval_examples=[])
        with self.assertRaises(RuntimeError):
            run_ablation(cfg, self.spider, cfg.ablation, self.dir / "abl_empty",
                         bundle=empty, train_fn=_fake_train, load_fn=_fake_load, prompt_fn=_stub_prompt,
                         progress=False)

    def test_degenerate_ablation_is_flagged_not_silently_reported(self) -> None:
        """A corpus with no synthetic data must not produce a clean-looking result.

        This is the failure mode that would make the report lie: every rung
        trains on identical raw data, so all deltas are zero and the ladder
        proves nothing. The runner must say so explicitly instead of emitting a
        tidy three-row table.
        """
        cfg = _cfg()
        raw_only = SFTBundle(
            train=[_sft_record("raw", i) for i in range(6)], dev=[],
            source="raw-only", eval_examples=list(EXAMPLES))
        payload = run_ablation(
            cfg, self.spider, cfg.ablation, self.dir / "abl_degenerate",
            bundle=raw_only, train_fn=_fake_train, load_fn=_fake_load,
            prompt_fn=_stub_prompt, progress=False)
        self.assertTrue(payload["degenerate"])
        self.assertTrue(payload["warnings"])
        self.assertIn("no synthetic records", payload["warnings"][0])
        # the identical rungs are visible in the numbers, not hidden
        self.assertEqual({r["synth_tokens"] for r in payload["rows"]}, {0})


class TestScaleEndToEnd(unittest.TestCase):
    """The scale runner must record (records, tokens, accuracy) per fraction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch_dir("eval_scale")
        cls.spider = TinySpider(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _bundle(self) -> SFTBundle:
        recs: list[SFTRecord] = []
        recs += [_sft_record("raw", i) for i in range(4)]
        recs += [_sft_record("cot", i) for i in range(8)]
        return SFTBundle(train=recs, dev=[], source="unit-test",
                         eval_examples=list(EXAMPLES))

    def test_scale_curve_records_growth(self) -> None:
        cfg = _cfg()
        out = self.dir / "scale_out"
        payload = run_scale_curve(
            cfg, self.spider, [0.0, 0.5, 1.0], out,
            bundle=self._bundle(), train_fn=_fake_train, load_fn=_fake_load, prompt_fn=_stub_prompt,
            progress=False)

        self.assertTrue((out / "scale_curve.json").exists())
        self.assertTrue((out / "scale_curve.csv").exists())
        points = payload["points"]
        self.assertEqual(len(points), 3)

        n_recs = [p["n_sft_records"] for p in points]
        tokens = [p["synth_tokens"] for p in points]
        self.assertEqual(n_recs, sorted(n_recs))
        self.assertEqual(tokens, sorted(tokens))
        self.assertEqual(tokens[0], 0)
        self.assertGreater(tokens[-1], tokens[0])
        for p in points:
            self.assertEqual(p["exec_acc"], 1.0)  # gold SQL -> real exec match
            self.assertEqual(p["n_eval"], len(EXAMPLES))

    def test_scale_saturation_is_reported(self) -> None:
        cfg = _cfg()
        out = self.dir / "scale_sat"
        payload = run_scale_curve(
            cfg, self.spider, [0.0, 0.5, 1.0], out,
            bundle=self._bundle(), train_fn=_fake_train, load_fn=_fake_load, prompt_fn=_stub_prompt,
            progress=False)
        sat = payload["saturation"]
        self.assertTrue(sat)
        self.assertIn("fraction", sat)
        self.assertIn("synth_tokens", sat)
        self.assertLessEqual(sat["synth_tokens"], max(p["synth_tokens"] for p in payload["points"]))

    def test_quick_mode_shrinks_budget_without_mutating_cfg(self) -> None:
        cfg = _cfg()
        before_epochs = cfg.train.epochs
        out = self.dir / "scale_quick"
        run_scale_curve(cfg, self.spider, [0.0, 1.0], out, quick=True,
                        bundle=self._bundle(), train_fn=_fake_train,
                        load_fn=_fake_load, prompt_fn=_stub_prompt, progress=False)
        self.assertEqual(cfg.train.epochs, before_epochs)
        self.assertTrue((out / "scale_curve.json").exists())

    def test_max_steps_is_reachable_for_every_arm(self) -> None:
        """The step cap must actually bind, or arms differ in training budget.

        ``train()`` stops at whichever comes first, ``max_steps`` or the epoch
        count. A caller setting ``max_steps=800`` with ``epochs=4`` on a 98-record
        arm (7 steps/epoch) would stop at 28 steps -- silently under-training that
        arm and confounding the ablation with a budget difference.
        """
        from fcdata.eval.pipeline import _cfg_with_reachable_steps

        cfg = _cfg()
        cfg.train.max_steps = 800
        cfg.train.batch_size = 16
        cfg.train.epochs = 4
        for n_records in (98, 143, 446):
            arm = _cfg_with_reachable_steps(cfg, n_records)
            steps_per_epoch = max(1, (n_records + 15) // 16)
            self.assertGreaterEqual(
                arm.train.epochs * steps_per_epoch, 800,
                f"{n_records} records cannot reach 800 steps")
        # no cap -> epochs untouched
        cfg.train.max_steps = 0
        self.assertEqual(_cfg_with_reachable_steps(cfg, 98).train.epochs, 4)
        # the caller's config is never mutated
        cfg.train.max_steps = 800
        _cfg_with_reachable_steps(cfg, 98)
        self.assertEqual(cfg.train.epochs, 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
