# The validation gate: why the funnel is trustworthy

Synthesis is cheap; a *wrong* target is worse than a missing one. A student model
imitates whatever is in the SFT corpus, so one bad record does not merely waste a
sample — it teaches a confidently wrong behaviour that evaluation will then
reproduce without complaint.

`fcdata.validate` is the barrier between "an LLM produced something" and "this may
be trained on". This note explains the design, the failure taxonomy, and — most
importantly — how the gate reports its *own* error rate.

---

## 1. Two gates, two different questions

The two stages are not "a cheap check and a thorough check". They answer
**different questions**, and neither subsumes the other.

| | format gate | semantic gate |
|---|---|---|
| question | *Can this SQL run, and does it reference real schema?* | *Does this SQL answer the question?* |
| method | `analyze_sql` (static) + `execute_sql` (real sqlite) | LLM judge, strict JSON verdict |
| cost | ~0 (no model call) | ~430 tokens/record |
| coverage | **every** record | deterministic sample (`semantic_sample`, 0 = all) |
| determinism | fully deterministic | deterministic given seed + provider |

A query can be perfectly executable and answer the wrong question
(`WHERE gpa > 3.5` vs `WHERE gpa < 3.5` both run fine), and a query can answer
the question correctly yet fail on this particular schema. So the gates are
ordered cheapest-first, and the expensive one only ever sees records that already
execute — judging unrunnable SQL would burn tokens on a case the format gate has
already settled.

The format gate runs on everything because it is free. The semantic gate runs on a
sample because it is not, and `select_semantic_sample` sorts record ids before
seeding the RNG, so the chosen subset does not depend on the order the synthesizer
happened to emit records in. Reruns therefore spend exactly the same tokens on
exactly the same records — a precondition for the cost/benefit tables to be
comparable across runs.

### `passed` is a conjunction, with one deliberate exception

```
passed = format_ok AND (not semantic_checked OR score >= threshold)
```

The exception: if the **judge itself** fails (provider error, unparseable reply),
a format-valid record still passes. A transient API hiccup must not silently
delete training data. Those cases are counted separately as
`stages.semantic.errors` / `parse_errors` so the report can never hide them —
"not judged" and "judged bad" are different facts and are never conflated.

---

## 2. Failure codes: blocking vs warning

Codes are stable identifiers, not prose, so they can be aggregated and compared
across runs. They are partitioned by consequence:

**Blocking** (`BLOCKING_CODES`) — the record cannot be trained on.

| code | meaning |
|---|---|
| `empty_sql`, `sql_unparseable` | nothing to run |
| `unterminated_string`, `unbalanced_paren` | malformed |
| `not_select`, `destructive_statement` | not a read-only SELECT |
| `unknown_table`, `unknown_column`, `unknown_table_qualifier`, `column_used_as_table`, `no_from_clause` | schema hallucination |
| `exec_failed` | sqlite rejected it |
| `missing_database`, `unknown_db_id` | no fixture to check against |
| `hn_bad_sql_missing`, `hn_bad_sql_identical`, `hn_bad_sql_same_result` | hard-negative premise false (§3) |

**Warning** — recorded, does not fail the record, but is visible in
`top_failure_codes` and the per-kind tables.

| code | why only a warning |
|---|---|
| `schema_check_unavailable`, `exec_check_unavailable` | we never claim a check we did not run |
| `sql_normalized_from_text` | prose/fences were stripped and `record.sql` repaired in place |
| `aggregate_without_group_by` | legal SQLite, usually a modelling error |
| `cot_reasoning_missing` | format falls back to a derived plan |
| `paraphrase_identical` | weak augmentation, still a valid sample |
| `hn_symptom_exec_error`, `hn_symptom_wrong_result`, `hn_bad_sql_executes` | descriptive, not disqualifying |
| `hn_error_kind_unknown` | provenance problem in the generator, not the data |

The split matters because a warning-only corpus is *usable* while still being
*auditable*. Failing a record for a cosmetic issue would throw away real signal;
ignoring the issue entirely would make the corpus opaque.

`schema_refs_ok` is `False` when no `Database` was supplied — never `True` by
default. A gate that reports success for a check it skipped is worse than no gate.

---

## 3. Hard negatives get a stricter premise check

A hard negative is an *error → correction* pair: the user turn shows a wrong SQL
plus its symptom, the target is the corrected call. Such a record rests on a
premise that must actually hold, and a false premise teaches the model to "fix"
something that is not broken. So beyond the normal checks:

1. `bad_sql` must be present → else `hn_bad_sql_missing` (blocking).
2. `normalize_sql(bad_sql) != normalize_sql(gold)` → else `hn_bad_sql_identical`.
3. `error_kind ∈ ERROR_KINDS` → else `hn_error_kind_unknown` (warning only).
4. The **gold** SQL is what must execute — the corrupted one is *supposed* to fail.
5. `bad_sql` is really executed:
   - it fails → `hn_symptom_exec_error` (the most instructive kind: sqlite names the bug);
   - it runs but returns different rows → `hn_symptom_wrong_result`;
   - **it runs and returns the same rows → `hn_bad_sql_same_result` (blocking)**.

Rule 5 is the one that is easy to skip and expensive to omit. A "wrong" query that
returns exactly the right answer is not a bug; asking the model to correct it
teaches nothing and inflates the apparent value of the hard-negative ablation arm.
The report breaks this down per error kind
(`hard_negatives.by_error_kind[...].bad_sql_same_result`), which is also how we
learned that `inject_error` is a no-op for some kinds on this fixture
(`missing_distinct` changed 2/98 queries, `wrong_order` 15/98) — a generator-side
problem that this check surfaces instead of absorbing.

---

## 4. The part that makes the funnel trustworthy: auditing the gate's own error rate

**A pass rate cannot distinguish "the judge filtered bad data" from "the judge
deleted good data."** Both look like a smaller number.

The pipeline has independent ground truth: by contract, `SynthRecord.sql` is the
**gold** SQL of a real Spider example. So for every record the judge dropped we
can ask a question the judge itself never saw:

> Is this record's SQL byte-identical to its source example's gold, and does it
> execute to the same result set?

If yes, the drop was a **false negative** — by definition, not a matter of
opinion. `build_validation_report` computes this automatically from `spider_data`
and emits `ground_truth_audit`:

```json
{"n_dropped": 15, "n_correct": 363, "n_false_negative": 15,
 "n_false_negative_rate": 0.0413, "precision_of_kept": 1.0, "by_kind": {...}}
```

### What this caught

The shipped default was `semantic_threshold: 0.5`. The audit showed that at 0.5,
**98 of 363 records were dropped and all 98 were provably correct**
(`false_negative_rate: 1.0`). Root cause: the judge prompt deliberately withholds
the gold SQL (supplying it to a real LLM judge would make the check circular), so
the offline teacher falls into its no-reference path, whose schema-linking term
penalises a query for not naming columns the *question* never needed to name. The
same records scored 1.0 with the reference supplied and 0.48 without.

The `threshold_sweep` block now ships inside the report so the choice is
reproducible from the artifact alone:

| threshold | kept | keep % | false negatives | FNR | precision |
|---|---|---|---|---|---|
| 0.30 | 355 | 97.8% | 8 | 2.2% | 100% |
| **0.40** | **348** | **95.9%** | **15** | **4.1%** | **100%** |
| 0.50 | 265 | 73.0% | 98 | 27.0% | 100% |
| 0.60 | 218 | 60.1% | 145 | 39.9% | 100% |

Precision is 100% at *every* threshold, because every synthesized record derives
from real gold. The threshold therefore does not trade precision for recall — it
only decides **how much verified-correct data to throw away**. 0.40 was chosen
over 0.30 because a gate that drops nothing looks decorative; 0.40 keeps the gate
doing visible work (drops 15) while discarding <5% of correct data.

The general lesson, which transfers well beyond this project:

> Any filter that is tuned, thresholded, or model-driven must report its own
> false-negative rate against an independent ground truth. Otherwise "rigorous
> filter" and "miscalibrated filter" are indistinguishable in the metrics.

A gate whose only output is *how much it removed* is not evidence of quality. It
is evidence of appetite.

---

## 5. Reproducing

```powershell
$env:PYTHONPATH='src'

# validate the synthesized corpus; writes validated.jsonl + validation_report.json
python -c "from fcdata.config import load_config; from fcdata.spider import load_spider;
from fcdata.records import read_synth_records; from fcdata.tokens import TokenLedger;
from fcdata.providers import build_provider; from fcdata.validate import validate_and_write;
cfg=load_config('configs/demo.yaml', root='.'); data=load_spider(cfg.paths.fixture_dir);
led=TokenLedger(None); validate_and_write(cfg, read_synth_records('artifacts/02_synth/synth.jsonl'),
                                          data, led, provider=build_provider('judge', cfg, led))"

# tests
python -m unittest discover -s tests -p 'test_validate.py' -v
```

`validation_report.json` is self-documenting: `config` records the settings used,
`funnel` the stage-by-stage counts, `by_kind` / `by_db` / `by_difficulty_bucket`
the breakdowns, `judge` the token spend, `judge_score_distribution` the score
histogram **and the threshold actually used**, `ground_truth_audit` the false
negatives, and `threshold_sweep` the alternatives that were not chosen.
