"""Definitive steps/sec measurement for the reference backend (dev tool).

Measures real optimizer steps at the configured sequence length on the real
fixture-derived SFT corpus, and projects how long a full training run takes.

Reports the *steady-state* cost (first step excluded: allocator warm-up), which is
what a training run actually experiences.

Run:  $env:PYTHONPATH='src'; python scripts/bench_steps.py [threads]
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from fcdata.config import TrainCfg, load_config  # noqa: E402
from fcdata.format import build_sft_records  # noqa: E402
from fcdata.records import SynthRecord, ValidationReport, ValidatedRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train.data import (  # noqa: E402
    build_encoded_dataset,
    collate,
    corpus_texts,
)
from fcdata.train.reference import build_model, describe_model, masked_lm_loss  # noqa: E402
from fcdata.train.vocab import WordTokenizer  # noqa: E402


def build_validated(data) -> list[ValidatedRecord]:
    """A realistic mix of raw / cot / paraphrase / hard_negative records."""
    out: list[ValidatedRecord] = []
    for i, ex in enumerate(data.train):
        kinds = ["raw"]
        if i % 3 == 0:
            kinds.append("cot")
        if i % 4 == 0:
            kinds.append("paraphrase")
        if i % 5 == 0:
            kinds.append("hard_negative")
        for kind in kinds:
            rec = SynthRecord(
                id=f"s_{i}_{kind}", source_id=ex.id, db_id=ex.db_id, kind=kind,
                question=ex.question, sql=ex.sql,
                reasoning=("First locate the table, then apply the filter, then "
                           "choose the aggregate." if kind == "cot" else ""),
                bad_sql="SELECT wrong FROM nope" if kind == "hard_negative" else "",
                error_kind="unknown_table" if kind == "hard_negative" else "",
                correction_note=("That table is absent from the schema; use the one "
                                 "listed in the DDL." if kind == "hard_negative" else ""),
                difficulty=ex.difficulty, prompt_tokens=120, completion_tokens=60)
            out.append(ValidatedRecord(
                record=rec,
                report=ValidationReport(sql_parses=True, schema_refs_ok=True,
                                        executes=True, semantic_checked=True,
                                        semantic_score=0.9, passed=True),
                id=f"v_{i}_{kind}"))
    return out


def measure(cfg: TrainCfg, encoded, weights, n_steps: int = 10) -> dict:
    """Time ``n_steps`` real steps; exclude the first (warm-up) from the average."""
    model = build_model(cfg, vocab_size=cfg.vocab_size)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    bs = cfg.batch_size

    batches = []
    for i in range(n_steps):
        idx = [(i * bs + j) % len(encoded) for j in range(bs)]
        batches.append(collate([encoded[k] for k in idx], [weights[k] for k in idx]))

    with torch.no_grad():
        init_loss, _, _ = masked_lm_loss(model(batches[0].input_ids), batches[0].labels)

    times: list[float] = []
    losses: list[float] = []
    for batch in batches:
        t0 = time.perf_counter()
        loss, _, _ = masked_lm_loss(model(batch.input_ids), batch.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.item()))

    steady = times[1:] or times
    mean_s = sum(steady) / len(steady)
    n_tok = sum(int(b.input_ids.numel()) for b in batches[1:])
    tok_s = sum(int(b.input_ids.numel()) for b in batches[1:]) / max(1e-9, sum(steady))
    return {
        "params": describe_model(model)["total_params"],
        "trainable": describe_model(model)["trainable_params"],
        "seq_len_mean": round(sum(int(b.input_ids.shape[1]) for b in batches) / len(batches), 1),
        "seq_len_max": max(int(b.input_ids.shape[1]) for b in batches),
        "batch_size": bs,
        "sec_per_step": round(mean_s, 4),
        "steps_per_sec": round(1.0 / mean_s, 3),
        "tokens_per_sec": round(tok_s, 1),
        "init_loss": round(float(init_loss.item()), 3),
        "ln_vocab": round(float(torch.log(torch.tensor(float(cfg.vocab_size))).item()), 3),
        "loss_first": round(losses[0], 3),
        "loss_last": round(losses[-1], 3),
    }


def main() -> int:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if threads:
        torch.set_num_threads(threads)
    print(f"torch threads: {torch.get_num_threads()}")

    cfg = load_config("configs/demo.yaml", root=".")
    data = load_spider("data/fixture")
    sft, _ = build_sft_records(cfg, build_validated(data), data, data.train)
    template = cfg.fmt.chat_template
    tok = WordTokenizer().fit(corpus_texts(sft, template=template), cfg.train.vocab_size)
    encoded, weights = build_encoded_dataset(
        sft, tok, max_seq_len=cfg.train.max_seq_len, template=template)
    lens = [len(e.input_ids) for e in encoded]
    print(f"sft records: {len(sft)}  vocab: {tok.vocab_size}  "
          f"seq p50={int(st.median(lens))} max={max(lens)}  "
          f"max_seq_len={cfg.train.max_seq_len}")

    tcfg = cfg.train
    tcfg.vocab_size = tok.vocab_size          # the fitted size, not the request
    r = measure(tcfg, encoded, weights)
    print("\n=== shipped demo.yaml config ===")
    print(json.dumps(r, indent=2))

    print("\n=== projection: full training runs ===")
    print(f"{'records':>8}{'epochs':>8}{'steps':>8}{'minutes':>10}")
    for n_records in (1000, 2000, 3000, 4000):
        for epochs in (2, 3, 4):
            steps = (n_records // r["batch_size"]) * epochs
            minutes = steps * r["sec_per_step"] / 60.0
            print(f"{n_records:>8}{epochs:>8}{steps:>8}{minutes:>10.1f}")

    print("\n=== smaller/faster variants (same data) ===")
    print(f"{'variant':<24}{'params':>10}{'s/step':>9}{'steps/s':>9}")
    base = {f: getattr(tcfg, f) for f in
            ("n_layer", "n_head", "d_model", "d_ff", "vocab_size", "max_seq_len",
             "lora_r", "lora_alpha", "batch_size", "epochs", "lr", "dropout",
             "weight_decay", "max_grad_norm")}
    for name, patch in (
        ("n3_d128_ff512", dict(n_layer=3)),
        ("n4_d128_ff512 (shipped)", {}),
        ("n4_d96_ff384", dict(d_model=96, d_ff=384)),
        ("n4_d128_ff256", dict(d_ff=256)),
    ):
        c = TrainCfg(**{**base, **patch})
        c.vocab_size = tok.vocab_size
        m = measure(c, encoded, weights, n_steps=6)
        print(f"{name:<24}{m['params']:>10}{m['sec_per_step']:>9}{m['steps_per_sec']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
