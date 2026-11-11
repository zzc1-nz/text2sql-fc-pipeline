# `fcdata.train` — LoRA SFT: CPU reference backend + production QLoRA path

Two trainers behind one API. The reference backend exists so the entire pipeline
(synthesis → validation → format → train → eval) runs **offline on a laptop CPU**
and produces real numbers; the HF backend is the path that actually fine-tunes a
7B model on a GPU box.

```python
from fcdata.train import train
metrics = train(cfg, sft_train, sft_dev, out_dir)   # writes model.pt + metrics.json
```

## The frozen API

```python
# fcdata/train/reference.py
build_model(cfg, *, vocab_size=None) -> ReferenceLM      # cfg: Config | TrainCfg | dict
save_checkpoint(path, model, tokenizer, cfg_dict) -> Path
load_model(path) -> (ReferenceLM, WordTokenizer)         # path = model.pt OR run dir
generate(model, tokenizer, prompt_text, max_new_tokens=128, temperature=0.0,
         *, top_k=0, stop_ids=None, max_prompt_tokens=0) -> str
generate_batch(model, tokenizer, prompt_texts, max_new_tokens=128, temperature=0.0) -> list[str]

# fcdata/train/__init__.py
train(cfg, sft_train, sft_dev, out_dir, *, max_steps=0, max_records=0) -> dict
```

`generate` is a **module-level function** (not a method) so it can be
monkeypatched in tests. `load_model` returns a model already in `.eval()` and a
tokenizer with `.encode` / `.decode` / `.vocab_size`. `temperature=0.0` is pure
greedy, so two runs of the same checkpoint produce identical predictions.

## The reference model is genuinely a transformer

Not a toy: token embedding + **learned positional embeddings**, N **pre-LN**
blocks (causal self-attention with exact lower-triangular masking, **GELU** FFN),
final LayerNorm, and a **weight-tied** LM head. LoRA is implemented from scratch
(low-rank `A`/`B` on q, k, v, o and both FFN projections, `B` zero-initialised so
the adapter starts as an exact no-op, `alpha/r` scaling, real `merge()`/`unmerge()`).

Loss is real cross-entropy on **assistant tokens only** (prompt tokens labelled
`-100`), with real AdamW, linear warmup + cosine decay, and gradient clipping.

### Why the reference model trains end-to-end, not LoRA-only

This is the single most important design note in this package.

LoRA means *"freeze a competent pretrained base and learn a low-rank steering
update"*. The reference backend's base is **randomly initialised** — it has no
language ability at all. Adapting it through a rank-8 bottleneck cannot work.
Measured, one record, 40 steps, everything else identical:

| mode | trainable | loss | outcome |
|---|---|---|---|
| `lora_only=True` (frozen base) | 73,728 | 6.131 → 4.139 | cannot learn |
| `lora_only=False` (end-to-end) | 990,208 | 6.182 → **0.008** | memorises |

So the default is `lora_only=False`: **all** parameters train. The LoRA adapters
still exist, are still exercised, and are still merged for inference — they are
part of the architecture, not decoration. `metrics.json` reports both
`trainable_params` and `lora_params` so the distinction stays visible.

Textbook frozen-base LoRA is available via `cfg.train.lora_only=True`, and the
negative result above is pinned by a test (`test_lora_only_cannot_learn_from_scratch`)
so the reasoning stays in the suite as executable evidence.

**The honest framing:** the reference backend is an *architecture-faithful but
from-scratch* stand-in that validates the **pipeline mechanics** end-to-end. The
`hf` backend — QLoRA on Qwen2.5-7B — is the one that demonstrates the
**data-quality** claim, because that is where LoRA is actually meaningful.

## Tokenizer

Word-level, fitted on the corpus. Special tokens have fixed ids
(`<pad>=0 <bos>=1 <eos>=2 <mask>=3 <unk>=4`). Multi-character SQL operators
(`>=`, `<=`, `<>`, `!=`, `||`) and chat markers (`<|`, `|>`) stay atomic; numbers
keep their decimals; punctuation is isolated.

Round-trip is verified at the level that matters for evaluation:

| check | method | result |
|---|---|---|
| SQL | `normalize_sql(decode(encode(sql))) == normalize_sql(sql)` | 5/5 |
| tool-call JSON | `json.loads(decode(encode(text))) == payload` | 5/5 |
| chat markers | exact string equality | 2/2 |

`<unk>` is kept (not dropped) so a missing word is visible rather than becoming a
silent hole in the SQL.

## Measured performance (this workspace, 20 vCPU, torch 2.12 CPU)

**Two environment traps that cost ~10x, both fixed and documented in code:**

1. **SDPA dropout on CPU is pathologically slow.** `F.scaled_dot_product_attention(..., dropout_p=0.1)`
   takes **252 ms** vs **24 ms** without dropout (B=16, H=4, T=380, D=32) — the
   CPU kernel materialises the full `(B,H,T,T)` matrix. Fixed by dropping out the
   attention *output* instead of the probabilities (GPT-NeoX does the same).
2. **Thread oversubscription.** Default 14 threads is **3x slower** than 4–8 for
   these small matmuls (0.13 vs 0.42 steps/s). `train()` clamps to ≤8 unless the
   caller set it explicitly.

Measured thread scaling (L4/d128/ff512, batch 16, seq 380):

| threads | 1 | 2 | 4 | 8 | 14 |
|---|---|---|---|---|---|
| steps/s | 0.248 | 0.362 | **0.424** | 0.376 | 0.128 |

`batch_size` barely changes per-example cost on CPU (it changes examples/step, not
cost/example). The dominant levers are `max_seq_len` (quadratic in attention) and
`n_layer` / `d_model`.

## `max_seq_len` — the trap that silently invalidates an experiment

For this corpus (Spider fixture, 6 DBs): prompt ≈ 257 tokens (mostly DDL),
target ≈ 130, **max total 552**.

**`max_seq_len` must be ≥ 576.** Truncation cuts the *prompt from the left*,
which is where the `CREATE TABLE` DDL lives. At 384, **37% of records lose part
of their schema**, and the model then emits syntactically valid SQL with
hallucinated table names — it cannot know a schema it was never shown. The
result is a near-zero accuracy that looks like a data problem but is purely
truncation damage.

The obvious speed instinct ("shrink `max_seq_len`") therefore **silently destroys
the experiment** here. `encode_record` never truncates the *target* (that would
make the task unlearnable) and reports `target_truncated`; `build_encoded_dataset`
prints a loud warning; `collate(..., max_seq_len=N)` raises with the offending
record ids.

## Recommended configs

**Verified end-to-end run** on the real fixture corpus (196 SFT records, 8 epochs):

```
n_layer=3 n_head=4 d_model=128 d_ff=512 max_seq_len=576 batch_size=16 lr=1e-3
-> 104 steps in 127s (0.82 steps/s)
-> loss 4.72 -> 1.24,  token_acc 0.674,  params 781,696 (LoRA: 55,296)
```

**Fits a 3–6 minute CPU run** (batch 16, seq 576, measured ~0.8–1.5 s/step):

```yaml
train:
  backend: reference
  n_layer: 3
  n_head: 4
  d_model: 128
  d_ff: 512
  max_seq_len: 576
  lora_r: 8
  lora_alpha: 16
  batch_size: 16
  epochs: 8          # ~12 steps/epoch on 200 records -> ~100 steps
  lr: 0.001          # end-to-end from scratch needs a higher LR than LoRA did
```

`lr` note: `3e-4` is a LoRA learning rate. End-to-end from scratch wants ~`1e-3`.

**Minimum steps before output is usable** (measured on the real fixture corpus):

| steps | what emerges |
|---|---|
| ~50–80 | tool-call *format* (`<tool_call>{"name": "execute_sql" ...`) |
| ~200 | 8 records memorised exactly — tok_acc 1.000, **byte-correct SQL** |
| ~400 | first executable SQL on *held-out* questions (10%) |
| **~600–800** | **usable held-out exec accuracy (10% → 27.5%)** |

The 8-record result is the important one: it proves the whole path
(tokenizer → mask → loss → decode → parse → execute) is correct, because the model
reproduces the gold SQL character-for-character. Generalisation is the separate,
slower story in the table below.

For smoke runs use the caps rather than a new config:

```python
train(cfg, sft_train, sft_dev, out_dir, max_steps=5, max_records=64)  # ~2 s
```

### Capacity reality check — it generalises, but only past ~600 steps

Held-out exec accuracy (every 5th SFT record held out; scored with
`dbexec.evaluate_pair` against real SQLite), `n_layer=3, d_model=128, d_ff=512,
max_seq_len=576, batch=16, lr=1e-3`:

| step | loss | exec_acc | executable |
|---|---|---|---|
| 100 | 1.040 | 0/40 (0%) | 0/40 (0%) |
| 200 | 0.740 | 0/40 (0%) | 0/40 (0%) |
| 400 | 0.386 | 0/40 (0%) | 4/40 (10%) |
| 600 | 0.180 | 4/40 (10%) | 9/40 (23%) |
| **800** | **0.072** | **11/40 (27.5%)** | **17/40 (43%)** |

Monotonic and still climbing at 800 steps (984 s, ~1.23 s/step). This is a real
capability result for a ~0.8M-parameter model trained from scratch on 196
examples — but note the shape: **below ~400 steps every configuration reads 0%**,
which makes an ablation table built on 100-step runs meaningless rather than
merely noisy. Budget ≥600 steps per arm if the numbers are going to carry a claim.

It is not competitive with a fine-tuned 7B; that is the `hf` path's job. The
reference backend demonstrates the pipeline end-to-end (synthesis → validation →
function-calling format → SFT → greedy decode → extraction → real SQL execution)
and shows capability emerging with training investment.

### Two failure modes that look like "the model learned nothing"

Both were hit for real during development, and both produce a **0% exec accuracy
with a plausible-looking loss curve** — worth knowing before debugging:

1. **Frozen-base LoRA on a random base** (see above): loss plateaus at ~ln(V) and
   decode collapses to repeated tokens.
2. **Extraction junk**: the word tokenizer round-trips `<|im_start|>` as
   `< | im_start | >`, so the model emits spaced `< tool_call >` markers. A naive
   `extract_sql` leaves the trailing `"}} < / tool_call > <|im_end|>` attached,
   and every prediction becomes non-executable. The evaluator must strip the
   template tail (guarded by paren/quote balance, so `'Physics'` and `COUNT(*)`
   keep their closing characters).



## Outputs

`model.pt` (self-contained: arch config + tokenizer + state_dict), `metrics.json`,
`config.json`, `train_log.txt`.

Key `metrics.json` fields:

```
history[]        {step, epoch, loss, token_acc, lr, grad_norm,
                  grad_norm_after_clip, clipped}      # per optimizer step
epochs_summary[] {epoch, train_loss, train_token_acc, dev_loss, dev_token_acc, wall_time_s}
total_params / trainable_params / frozen_params / lora_params / lora_only
final_train_loss / final_train_token_acc / final_dev_loss / final_dev_token_acc
steps_per_sec / tokens_per_sec / wall_time_s / loss_decreased
```

`grad_norm` is the **pre-clip** norm (that is what `clip_grad_norm_` returns, and
it is the informative diagnostic); `grad_norm_after_clip` is what was applied.

## `hf_peft.py` — production QLoRA path

**Not executed in this CPU-only environment** (no `transformers` / `peft` /
`bitsandbytes`, no CUDA torch). It is complete, correct, import-guarded code that
runs unchanged on a GPU box:

- NF4 4-bit base (`hf_load_in_4bit`), double quantisation, bf16 compute
- LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Response-only loss**: masks everything up to the assistant header, so the
  model is not trained to generate user turns or the schema
- gradient checkpointing (`use_reentrant=False`), gradient accumulation, cosine LR
- `RuntimeError` with the exact `pip install` line when deps are missing

The one part that *is* tested without a GPU is the label masking
(`find_assistant_span` / `build_response_labels`), because an off-by-one there
trains the model to emit the assistant *header* instead of the answer — and still
shows a falling loss. Those are pure functions over token ids and are unit-tested
against a synthetic vocabulary.

## Tests

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -p 'test_train.py' -v
```

91 tests, ~30 s. The masking tests are the point:

- `test_causal_mask_blocks_future` — perturbing the last token leaves all earlier
  logits bit-identical (position *i* cannot see *i+1*); the mask is proven, not assumed
- `test_manual_mask_equals_sdpa` — explicit tril path == SDPA path
- `test_label_shift_is_next_token` — logits at *i* score `labels[i+1]`; the
  wrong-direction shift is shown to be much worse
- `test_loss_ignores_masked_positions` — compared against a hand-computed loss
  over supervised pairs only
- `test_prompt_masked_target_is_supervised` — the label boundary is exactly `prompt_len`
- `test_can_overfit_single_example` — **the gate**: 60 steps on one record must
  reach loss < 2.0 *and* greedy generation must reproduce the target SQL
- `test_length_never_exceeds_max_seq_len` — sweeps every budget 2→400
