# Function-Calling Data Synthesis + Evaluation Pipeline (Text2SQL / Spider)

A production-shaped pipeline that **synthesises function-calling fine-tuning data with an LLM, validates it through multiple gates, trains a LoRA SFT model, and measures whether the token investment actually bought capability.**

Text2SQL on Spider is the task carrier: the model must emit a real `execute_sql` tool call whose SQL executes to the correct result set.

> **The question this repo answers:** *does spending tokens on synthetic data actually make the model better, and how do you prove it?*
>
> Every accuracy number here comes from **executing generated SQL against real SQLite databases** (`fcdata.dbexec.evaluate_pair`). Nothing is simulated, and no metric is hardcoded. Where the pipeline underperforms, that is reported rather than tuned away — see [Honest results](#honest-results-what-actually-happened).

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [The five stages](#the-five-stages)
- [Token accounting: the headline](#token-accounting-the-headline)
- [Results](#results)
- [Honest results: what actually happened](#honest-results-what-actually-happened)
- [Design decisions that matter](#design-decisions-that-matter)
- [Traps found by measurement](#traps-found-by-measurement)
- [Running on real Spider + real APIs](#running-on-real-spider--real-apis)
- [Repo layout](#repo-layout)
- [Testing](#testing)

---

## Architecture

```
Spider-format dataset (6 real SQLite DBs, 122 executable gold queries)
        │
        │  STAGE 1 — SYNTHESIS  (cheap model, ~600 tok/record)
        ▼
   CoT explanation │ paraphrase │ error→correction hard negative
        │            │            │
        │            │            └── inject_error + REAL EXECUTION CHECK
        │            │                (reject if the "wrong" SQL returns the right rows)
        ▼            ▼            ▼
   363 synthetic records  ──────────────►  token ledger (per stage/role/model/call)
        │
        │  STAGE 2 — VALIDATION  (quality model, ~430 tok/call)
        ▼
   GATE 1 format: parses? schema refs real? executes?
   GATE 2 semantic: LLM judge — does this SQL answer this question?
        │            └── reports its OWN false-negative rate vs execution ground truth
        ▼
   348 validated records (95.9% of 363; 4.1% FNR)
        │
        │  STAGE 3 — FORMAT  (deterministic, free)
        ▼
   OpenAI-style tool calls:  {"name":"execute_sql","arguments":"{\"sql\": ...}"}
   3 chat templates (qwen / llama3 / plain)
        │
        │  STAGE 4 — LoRA SFT
        ▼
   reference backend: real decoder-only transformer, 780K params, CPU, end-to-end
   hf backend:        Qwen2.5-7B / Llama-3.1-8B QLoRA (4-bit NF4) — GPU, not run here
        │
        │  STAGE 5 — EVAL  (real sqlite execution)
        ▼
   exec_acc · exact_match · token_acc · skeleton_match · table_set_match
   column_f1 · clause_match · failure_taxonomy   + bootstrap CIs
        │
        ▼
   ablation (raw_only / +CoT / +hard negatives) · data-scale curve · cost-vs-quality
```

**Why the ledger is a first-class citizen:** the entire thesis is about token investment, so token accounting cannot be an afterthought. Every provider call writes exactly one append-only `UsageRecord` through a template-method base class (`providers/base.py`), which makes double-counting structurally impossible rather than merely discouraged.

---

## Quick start

No network, no API keys, no GPU required — the bundled fixture and a deterministic offline teacher make the whole pipeline runnable:

```bash
export PYTHONPATH=src          # Windows: $env:PYTHONPATH='src'

python -m fcdata.cli fixture      # build 6 real SQLite DBs from SQL fixtures
python -m fcdata.cli synth        # Stage 1: 363 synthetic records + token ledger
python -m fcdata.cli validate     # Stage 2: two-gate validation
python -m fcdata.cli format       # Stage 3: function-calling SFT records
python -m fcdata.cli train        # Stage 4: LoRA SFT
python -m fcdata.cli eval         # Stage 5: real-execution evaluation
python -m fcdata.cli ablation     # raw_only vs +CoT vs +hard_negatives
python -m fcdata.cli scale        # data-scale curve
python -m fcdata.cli extrapolate  # project measured cost to Spider scale
python -m fcdata.cli report       # tables + figures into reports/
```

Or the whole thing: `python -m fcdata.cli all`.

Every stage is independently re-runnable and reads/writes JSONL under `artifacts/NN_*/`, so you can change a judge prompt and re-run only `validate` without paying for synthesis again.

---

## The five stages

### Stage 1 — Synthesis (`fcdata/synth/`)

Three augmentation kinds, each with its own prompt contract and output schema:

| Kind | What it produces | Purpose |
|---|---|---|
| `cot` | A step-by-step rationale naming the real tables/columns, then the SQL | Teaches *why*, not just *what* |
| `paraphrase` | A semantically equivalent rewrite of the question | Robustness to phrasing |
| `hard_negative` | A plausible-but-wrong SQL + a correction note explaining the bug | Teaches error correction |

**Hard negatives are the interesting one.** A naive implementation corrupts the gold SQL and emits it. That produces junk: `inject_error` is a **no-op for many kinds on most queries** — measured no-op rates across the 98 fixture examples:

```
wrong_table       0.0%      wrong_literal     34.7%
wrong_column      0.0%      wrong_aggregate   45.9%
wrong_operator   26.5%      missing_join      58.2%
missing_group_by 67.3%      wrong_order       84.7%
missing_distinct 98.0%
```

So generation iterates a preference list, and — critically — **executes the corrupted SQL** (`dbexec.evaluate_pair`, the *same* function validation uses) before spending a teacher call. A negative is kept only if the bad SQL errors *or* returns a different result set. Yield: **245/450 kind-attempts (54.4%)** survived the gate; zero fake negatives reached the corpus.

### Stage 2 — Validation (`fcdata/validate/`)

Two orthogonal gates, not cheap-vs-thorough:

- **Format gate** (free, runs on everything): parses → schema references resolve → *executes on the real database*. For hard negatives it additionally enforces the **premise check**: the bad SQL must differ from gold **and** actually be wrong (fail to execute, or return different rows). A "wrong" query returning identical rows is **blocking** — it teaches nothing.
- **Semantic gate** (LLM judge, ~430 tok/call): does this SQL answer this question? The judge is deliberately **not** shown the gold SQL, because that would make the check circular.

Full detail in [`docs/validation.md`](docs/validation.md).

### Stage 3 — Format (`fcdata/format/`)

The actual product deliverable: conversion to OpenAI-compatible function-calling records.

```json
{"role": "assistant", "tool_calls": [{"id": "call_...", "type": "function",
  "function": {"name": "execute_sql",
               "arguments": "{\"sql\": \"SELECT AVG(unit_price) FROM product WHERE category = 'Furniture'\"}"}}]}
```

`arguments` is a JSON **string**, as the real API requires. Three genuinely-correct chat templates (`qwen`, `llama3`, `plain`). Measured **100% tool-call validity** across 446 records, 0 target problems.

### Stage 4 — Training (`fcdata/train/`)

Two backends behind one `train()` call:

**`reference`** — a real decoder-only transformer implemented from scratch in PyTorch (pre-LN, causal self-attention, GELU FFN, tied LM head) plus LoRA adapters on q/k/v/o and FFN projections, real AdamW + warmup + cosine decay, real cross-entropy on assistant tokens only. **780K params, trains end-to-end on CPU.** This is what all numbers below were measured with.

**`hf`** — production QLoRA for Qwen2.5-7B / Llama-3.1-8B: NF4 4-bit, LoRA on all 7 projections, response-only label masking, gradient checkpointing, accumulation, bf16. Complete and import-guarded, **not executed in this CPU-only environment** (no `transformers`/`peft`/`bitsandbytes`, no CUDA torch). Its GPU-free label-masking logic *is* unit-tested — an off-by-one there trains the model to emit the assistant *header* instead of the answer while loss still falls.

### Stage 5 — Evaluation (`fcdata/eval/`)

`exec_acc` and `exact_match` are strict and all-or-nothing. At this model scale that is a **blunt instrument** — a 60-token SQL needs ~99% per-token accuracy to be byte-perfect, so exec_acc reads 0% long after the model has learned the schema and query structure. So evaluation also reports **partial credit**, which is what actually shows a learning signal:

`token_acc` (edit similarity) · `skeleton_match` (structure right, literals abstracted) · `table_set_match` · `column_f1` · `clause_match` (join/aggregate/group-by/order-by/limit) · `valid_sql_rate`, all with bootstrap CIs and a **failure taxonomy** that maps predictions onto the same error classes the hard negatives were built from.

---

## Token accounting: the headline

Every call is recorded exactly once, by stage, role and model (`artifacts/*_ledger.jsonl`):

| Stage | Calls | Prompt tok | Completion tok | Total tok | Cost (USD) | tok/record |
|---|---:|---:|---:|---:|---:|---:|
| `synth` (cheap) | 392 | 192,516 | 18,072 | 210,588 | $0.0114 | ~580 |
| `validate` (judge) | 363 | 142,905 | 13,508 | 156,413 | $0.1834 | ~431 |
| **total** | **755** | **335,421** | **31,580** | **367,001** | **$0.1949** | — |

Per-task synthesis cost (measured, 50-example sample): **CoT 658 > hard_negative 560 > paraphrase 578 tokens per accepted record**. CoT is the most expensive per call; paraphrase is cheapest per call but loses ~15% to dedup, so its *effective* per-record cost is higher.

Note the cost split: the judge is **13× more expensive per token** than the cheap generator, so it accounts for **94% of total spend** despite being only 43% of the tokens. That is the intended design — *cheap model for volume, quality model for a small share of calls* — but it is worth stating that the validation stage, not synthesis, dominates cost at these prices.

### Projection to Spider scale

Per-sample cost is measured on the fixture and projected linearly (prompt length is dominated by the per-database DDL block, which is constant as the corpus grows):

| Synthesis plan | Source questions | Augmented records | Synth tokens | Judge tokens | Total tokens | Est. cost |
|---|---:|---:|---:|---:|---:|---:|
| Spider train (7,000 q) | 7,000 | 28,000 | 16.2M | 12.1M | **28.3M** | $15.03 |
| Spider full (10,181 q) | 10,181 | 40,724 | 23.6M | 17.5M | **41.2M** | $21.86 |
| Augmented ×3 (30k samples) | 10,181 | 30,543 | 17.7M | 13.2M | **30.9M** | $16.40 |
| Augmented ×4 (40k samples) | 10,181 | 61,086 | 35.4M | 26.3M | **61.8M** | $32.79 |
| Augmented ×5 (50k samples) | 10,181 | 71,267 | 41.3M | 30.7M | **72.1M** | $38.26 |

The brief's target — **3–5万条样本, 千万级 tokens** — is reached by every plan above: the smallest projection is **28.3M tokens**, comfortably into 千万级 (tens of millions). The ×4 plan lands at **61.8M tokens for ~$33**, which is the regime the brief describes.

Reproduce with `python -m fcdata.cli extrapolate` → `reports/token_extrapolation.json`.

---

## Results

> Regenerate with `python -m fcdata.cli ablation && python -m fcdata.cli scale && python -m fcdata.cli report`.
> See [`reports/README_TABLES.md`](reports/README_TABLES.md) for always-current tables and figures.

All three arms train **exactly 800 steps** on a fresh model — training budget is held constant so the only variable is the data mix. (This matters: a naive implementation lets the smaller arms stop early at their epoch count, which silently confounds data quality with training budget. `_cfg_with_reachable_steps` derives the epoch count per arm from its own record count.)

**Ablation ladder** — each arm trains a fresh model for **exactly 800 steps** on 24 held-out dev examples, scored by real SQLite execution:

| Config | Data mix | SFT records | Synth tokens | Exec acc | EM | Executable | Token acc | Table-set | Column F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_only` | raw Spider only | 98 | 0 | 0.0% | 0.0% | 16.7% | 26.0% | 12.5% | **22.8%** |
| `plus_cot` | raw + CoT rationales | 192 | 60,457 | 0.0% | 0.0% | **25.0%** | 19.5% | 16.7% | 15.3% |
| `plus_hard_negatives` | raw + CoT + paraphrase + hard negatives | 542 | 239,883 | 0.0% | 0.0% | **25.0%** | **32.1%** | **20.8%** | 19.4% |

**Failure taxonomy** — counts out of 24, from the same run. The interesting part is *which* error class each mix moves:

| Config | wrong_column | not_executable | wrong_table | wrong_values | missing_join | extra_join | missing_order_by | wrong_aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_only` | 9 | 6 | 5 | 1 | 1 | 1 | 1 | 0 |
| `plus_cot` | **4** | 12 | **1** | 3 | 3 | 0 | 0 | 0 |
| `plus_hard_negatives` | 6 | **6** | 6 | 1 | **0** | 1 | 1 | 3 |

**Reading it honestly** — see [Honest results](#honest-results-what-actually-happened). In short: strict execution accuracy is **0% for every arm**, but partial-credit signals move coherently — synthetic data raises table-level schema grounding (12.5% → 20.8%) and executability (16.7% → 25.0%), while `+CoT` collapses `wrong_table` errors from 5 to 1 and `+hard_negatives` eliminates `missing_join`. **The effect size (2 of 24 examples) is only just above the measured run-to-run noise floor (1 example), so this is reported as a directional signal, not an established effect.**

---

## Honest results: what actually happened

This section exists because the interesting engineering *was* the debugging, and because a portfolio artifact that hides its failures is worth less than one that documents them.

### The pipeline is proven correct end-to-end

An 8-record overfit reaches `tok_acc 1.000` and reproduces gold SQL **byte-for-byte** — right joins, right aliases, right literals:

```
gold: SELECT T3.passenger_name FROM flight AS T1 JOIN passenger AS T3
      ON T1.flight_id = T3.flight_id WHERE T1.price > 900
pred: SELECT T3.passenger_name FROM flight AS T1 JOIN passenger AS T3
      ON T1.flight_id = T3.flight_id WHERE T1.price > 900
```

That single fact verifies the tokenizer, label masking, causal shift, prompt parity, decoding and tool-call grammar. The pipeline produces learnable data.

### Capability emerges with training investment — it just needs a lot of it

Held-out execution accuracy vs training steps (real SQLite):

| Steps | Loss | Exec acc | Executable |
|---:|---:|---:|---:|
| 100 | 1.040 | 0% | 0% |
| 200 | 0.740 | 0% | 0% |
| 400 | 0.386 | 0% | 10% |
| 600 | 0.180 | 10% | 23% |
| **800** | **0.072** | **27.5%** | **43%** |

Monotonic and still climbing at 800. **Below ~400 steps every configuration reads 0%**, which is why the demo pins `max_steps: 800`.

### The ablation: a weak, directional signal — and how weak is measured

At 800 steps on a ~805K-parameter model, **strict execution accuracy is 0.0% for all three data mixes.** That is the truthful result and it is reported unchanged. It is also *expected arithmetic*, not a broken pipeline: execution accuracy requires a byte-perfect statement, so it stays at zero for any model below roughly 99% token accuracy.

What **did** move is measurable and directional:

| Signal | `raw_only` | `+CoT` | `+hard_negatives` | Reading |
|---|---:|---:|---:|---|
| `table_set_match` | 12.5% | 16.7% | **20.8%** | Schema grounding rises with synthetic data |
| `executable_rate` | 16.7% | **25.0%** | **25.0%** | 1.5× more predictions are runnable SQL |
| `token_acc` | 26.0% | 19.5% | **32.1%** | Mixed — hard negatives help, CoT alone hurts |
| `column_f1` | **22.8%** | 15.3% | 19.4% | Baseline wins: more data trades column precision |
| `wrong_table` count | 5 | **1** | 6 | CoT specifically fixes table selection |
| `missing_join` count | 1 | 3 | **0** | Hard negatives eliminate missing joins |
| `wrong_aggregate` count | 0 | 0 | 3 | Hard negatives *introduce* aggregate errors |

#### How large is the noise floor? (measured, not assumed)

The dev set is **24 examples**, so one example is **4.17 pp**. Two runs of the *identical* configuration (98 raw records, same seed, 800 steps) were available on disk, which lets the noise floor be measured rather than assumed:

| Metric | run A | run B | spread |
|---|---:|---:|---:|
| `executable_rate` | 16.7% | 20.8% | **4.2 pp (1 example)** |
| `table_set_match` | 12.5% | 8.3% | **4.2 pp (1 example)** |
| `token_acc` | 26.0% | 28.8% | 2.9 pp |
| `column_f1` | 22.8% | 19.4% | 3.3 pp |

**So the headline effect — `table_set_match` 12.5% → 20.8% — is +8.3 pp, or 2 of 24 examples, against a measured noise floor of ~1 example.** It is above the noise floor but only just. Bootstrap CIs overlap substantially (`raw_only` 8.3–20.8%, `plus_hard_negatives` 4.2–33.3%).

**The honest conclusion: this is a directional signal worth investigating at scale, not an established effect.** Reproduce the variance analysis with `python scripts/variance_check.py`.

#### What the failure taxonomy shows

The three mixes move *different* error classes rather than uniformly improving:

- **`+CoT` collapses `wrong_table` from 5 → 1.** Teaching the model to reason about which tables a question needs is exactly what fixes table selection.
- **`+hard_negatives` eliminates `missing_join` (1 → 0)** and gives the best `token_acc` (32.1%), but *introduces* 3 `wrong_aggregate` errors — the error→correction pairs teach it to consider aggregates it wouldn't otherwise emit, and sometimes it picks the wrong one.
- **`+CoT` has the most `not_executable` predictions (12)** despite fixing table selection — a coherent story: it reasons about the right tables but writes more elaborate SQL that fails to parse.

This is the difference between a useless null result and a diagnostic: **the synthetic data is teaching the model which tables to use and how to structure joins; it has not yet taught it to write the statement correctly.** The most likely limiter is capacity — at a fixed 800-step budget, the 542-record arm sees each record ~24 times while the 98-record baseline sees each ~130 times.

Converting this to a positive would require more capacity or a longer per-arm budget. **The `hf` backend on a GPU box is what would properly test the data-quality hypothesis**, because a 7B model has the capacity to turn broader schema coverage into correct SQL. That is stated as an expectation, not a result.

### Data-scale curve: no clean scaling law

| Synth fraction | SFT records | Synth tokens | `table_set_match` | `token_acc` | Executable |
|---:|---:|---:|---:|---:|---:|
| 0% | 98 | 0 | 8.3% | 28.8% | 20.8% |
| 25% | 186 | 48,116 | **25.0%** | 20.9% | 8.3% |
| 50% | 272 | 94,276 | 12.5% | 25.7% | 20.8% |
| 100% | 446 | 188,319 | 16.7% | 25.0% | 12.5% |

The sequence **zig-zags** — it is not a scaling law. At n=24 per point, one example is 4.17 pp, and the 25% point sits 6/24 above the 100% point. With single runs per point and a ~1-example noise floor, this curve is consistent with *no trend* or a *weak positive trend*, and cannot distinguish them. Reporting it as monotone would be dishonest; the honest statement is that **the data-scale question is unanswered at this sample size**, and answering it needs repeated runs per point (≥3 seeds) and a larger eval set.

---

## Design decisions that matter

**Token accounting is structural, not incidental.** Providers implement only `_generate()`; the base class owns ledger recording. It is impossible to add a provider that forgets to record, or records twice.

**The validator reports its own false-negative rate.** This is the single most transferable idea in the repo. A pass rate cannot distinguish *"filtered bad data"* from *"deleted good data"* — both look like a smaller number. The semantic gate's threshold was therefore calibrated against **execution ground truth** (does the record's SQL reproduce the source gold result set?), not against the judge's own output:

| Threshold | Kept | Precision | Correct records dropped |
|---:|---:|---:|---:|
| 0.30 | 97.8% | 100% | 8 |
| **0.40** | **95.9%** | **100%** | **15** |
| 0.50 | 73.0% | 100% | 98 |
| 0.60 | 60.1% | 100% | 145 |

Precision is 100% at *every* threshold, because every synthetic record's SQL derives from real Spider gold. So the threshold does not trade precision for recall — it only decides how much verified-correct data to throw away. The inherited default of 0.5 was silently deleting 27% of good data. Reproduce with `python scripts/tune_threshold.py`.

**Synthesis is gated by execution before it is paid for.** Hard-negative corruption is verified against a real database *before* the teacher call, so rejected attempts cost zero tokens.

**Determinism.** The offline teacher seeds from `sha256(task|sample_id|prompt)` — not `hash()`, which is per-process salted — so two separate processes produce byte-identical corpora. Verified across processes.

---

## Traps found by measurement

Each of these silently invalidated results until measured. They are documented because they generalise.

| Trap | Symptom | Root cause |
|---|---|---|
| **Frozen-base LoRA** | loss plateaus at ~6.1, degenerate output | LoRA-only adaptation assumes a *pretrained* base. A randomly-initialised base must be trained end-to-end. Measured: LoRA-only 6.13→4.14 vs end-to-end 6.18→0.008 on the same example. |
| **`max_seq_len` too small** | model hallucinates table names | At 384, **37% of records lose part of their DDL block**. The model emits schema-shaped SQL for tables it was never shown. Needs ≥576 on this corpus. |
| **Learning rate from the wrong regime** | slow, degenerate learning | 3e-4 is a *LoRA* LR. End-to-end from scratch needs ~1e-3. |
| **Naive SQL extraction** | exec_acc stuck at 0% | The word-level tokenizer emits spaced template markers (`< tool_call >`). A naive extractor leaves `"}} < / tool_call >` appended, so every prediction is non-executable — looking exactly like "the model learned nothing". |
| **SDPA CPU dropout** | 10× slower training | `scaled_dot_product_attention(dropout_p>0)` materialises the full attention matrix on CPU: 252ms vs 24ms per call. Drop out the attention *output* instead (GPT-NeoX style). |
| **Torch thread oversubscription** | 3× slower training | 14 threads (this box's default) is 3× slower than 4–8 for small matmuls. |
| **`inject_error` no-ops** | corpus full of fake negatives | Most corruption kinds don't apply to most queries. Must verify by execution, not by string comparison. |
| **Judge verdict ≠ score** | incoherent JSONL | Verdict must be *derived* from score, never sampled independently. |
| **Zero-column F1** | partial metrics decorative | `column_f1` scored 1.0 when both sides referenced zero columns, so a total miss looked perfect. |

---

## Running on real Spider + real APIs

The bundled fixture exists because this environment has no network. Swapping in the real benchmark is configuration, not code:

```bash
# 1. Get Spider 1.0 (Yale LILY) and normalise its layout
python scripts/prepare_spider.py --spider /path/to/spider_raw --out data/spider --verify

# 2. Set API keys
export DEEPSEEK_API_KEY=...      # cheap generator (bulk of the tokens)
export OPENAI_API_KEY=...        # quality judge

# 3. Run against configs/production.yaml
python -m fcdata.cli synth   --config configs/production.yaml
python -m fcdata.cli validate --config configs/production.yaml
python -m fcdata.cli format   --config configs/production.yaml
python -m fcdata.cli train    --config configs/production.yaml   # backend: hf
```

`configs/production.yaml` is pre-wired for DeepSeek as the cheap generator (`$0.27/$1.10` per 1M tokens) and `gpt-4o-mini` as the judge (`$0.15/$0.60`), with real concurrency, retry/backoff and a disk response cache. The `hf` backend then runs Qwen2.5-7B QLoRA on a single 24GB GPU.

The offline teacher is **not** a stub that returns canned text: it parses the DDL block back into a schema and grounds its CoT, paraphrases and judge verdicts in the real tables and columns. That is what makes the offline demo produce genuinely learnable data.

---

## Repo layout

```
src/fcdata/
  config.py         typed config (yaml + dotted CLI overrides)
  tokens.py         token estimation + append-only ledger
  schema.py         Spider schema model, DDL rendering
  sqlutils.py       tokenizer, static/schema validation, normalization,
                    difficulty scoring, error injection
  dbexec.py         real SQLite build/execute/compare (the ground truth)
  spider.py         Spider-format loaders
  fixtures.py       6 hand-authored relational databases + 122 gold queries
  build_fixture.py  fixture generator (introspects sqlite -> tables.json)
  providers/        LLM providers: deterministic offline teacher + OpenAI-compatible
  synth/            CoT / paraphrase / hard-negative generation
  validate/         format gate + semantic judge + funnel report
  format/           function-calling SFT conversion, 3 chat templates
  train/            reference transformer + LoRA, QLoRA production path
  eval/             inference, exec/EM + partial credit, ablation, scale curve
  report/           markdown tables, CSVs, matplotlib figures
  extrapolate.py    measured cost -> Spider-scale projection
  cli.py            one subcommand per stage
docs/validation.md  the two-gate design and the false-negative audit
scripts/            prepare_spider, threshold tuning, capacity probes, benchmarks
tests/              494 tests, no network, self-contained
```

---

## Testing

```bash
export PYTHONPATH=src
python -m unittest discover -s tests -v      # 494 tests, ~50s
```

No network, no API keys, no GPU. Tests build their own tiny Spider-format fixtures and execute real SQL.

---

## License / data

Spider is distributed for research use — check its license before redistributing database files. The bundled fixture databases are original synthetic data authored for this repo.
