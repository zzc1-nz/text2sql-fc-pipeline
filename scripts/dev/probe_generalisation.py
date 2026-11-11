"""Long-run probe: at what step count does held-out generation become usable?

Trains on the real fixture SFT corpus for a configurable number of steps and, at
intervals, measures exec accuracy on **held-out** records (records the model never
trained on), using the evaluator's parser and real sqlite execution.

This answers the only question that matters for budgeting: "how many steps until
the model produces SQL that actually executes on unseen questions?"

Run:  $env:PYTHONPATH='src'; python scripts/probe_generalisation.py [steps]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from fcdata.config import load_config  # noqa: E402
from fcdata.dbexec import evaluate_pair  # noqa: E402
from fcdata.format import assistant_sql, build_sft_records, render_record_prompt  # noqa: E402
from fcdata.records import SynthRecord, ValidationReport, ValidatedRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train.data import build_encoded_dataset, collate, corpus_texts  # noqa: E402
from fcdata.train.reference import (  # noqa: E402
    build_model,
    describe_model,
    generate_batch,
    masked_lm_loss,
)
from fcdata.train.vocab import WordTokenizer  # noqa: E402


def build_validated(data) -> list[ValidatedRecord]:
    out = []
    for i, ex in enumerate(data.train):
        rec = SynthRecord(id=f"s_{i}", source_id=ex.id, db_id=ex.db_id, kind="raw",
                          question=ex.question, sql=ex.sql, difficulty=ex.difficulty,
                          prompt_tokens=120, completion_tokens=60)
        out.append(ValidatedRecord(record=rec, report=ValidationReport(
            sql_parses=True, schema_refs_ok=True, executes=True, passed=True),
            id=f"v_{i}"))
    return out


def main() -> int:
    steps_target = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    cfg = load_config("configs/demo.yaml", root=".")
    data = load_spider("data/fixture")
    sft, _ = build_sft_records(cfg, build_validated(data), data, data.train)
    template = cfg.fmt.chat_template

    # held-out split: every 5th record, so the probe measures generalisation
    held = [r for i, r in enumerate(sft) if i % 5 == 0]
    train = [r for i, r in enumerate(sft) if i % 5 != 0]
    print(f"total={len(sft)} train={len(train)} held_out={len(held)}")

    tok = WordTokenizer().fit(corpus_texts(sft, template=template), cfg.train.vocab_size)
    tcfg = cfg.train
    tcfg.vocab_size = tok.vocab_size
    tcfg.max_seq_len = 576
    tcfg.n_layer = 3
    tcfg.n_head = 4
    tcfg.d_model = 128
    tcfg.d_ff = 512
    tcfg.batch_size = 16
    tcfg.lr = 1e-3

    encoded, weights = build_encoded_dataset(train, tok, max_seq_len=tcfg.max_seq_len,
                                             template=template)
    print(f"vocab={tok.vocab_size} train_encoded={len(encoded)} "
          f"max_len={max(len(e.input_ids) for e in encoded)}")

    model = build_model(tcfg, vocab_size=tok.vocab_size)
    print(f"params={describe_model(model)['total_params']:,} "
          f"trainable={describe_model(model)['trainable_params']:,}")
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    bs = tcfg.batch_size

    def probe(tag: str) -> dict:
        model.eval()
        prompts = [render_record_prompt(r, template) for r in held]
        outs = generate_batch(model, tok, prompts, max_new_tokens=110, temperature=0.0)
        exec_match = executable = 0
        for rec, out in zip(held, outs):
            sql = assistant_sql({"role": "assistant", "content": out}) or ""
            if not sql.strip():
                continue
            try:
                r = evaluate_pair(data.db_path(rec.db_id), sql, rec.target_sql)
            except FileNotFoundError:
                continue
            executable += int(r["executable"])
            exec_match += int(r["exec_match"])
        n = len(held)
        print(f"  [{tag}] exec_acc {exec_match}/{n} ({exec_match / n:.1%})  "
              f"executable {executable}/{n} ({executable / n:.0%})")
        return {"exec_match": exec_match, "executable": executable, "n": n}

    model.train()
    t0 = time.perf_counter()
    step = 0
    import random
    rng = random.Random(17)
    checks = sorted({s for s in (100, 200, 400, 600, 800, 1200) if s <= steps_target})
    losses: list[float] = []
    curve = []
    while step < steps_target:
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            batch = collate([encoded[i] for i in idx], [weights[i] for i in idx])
            loss, acc, _ = masked_lm_loss(model(batch.input_ids), batch.labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg.max_grad_norm)
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step in checks:
                recent = sum(losses[-30:]) / len(losses[-30:])
                print(f"\n=== step {step}  loss {recent:.4f}  "
                      f"({time.perf_counter() - t0:.0f}s) ===")
                p = probe(f"step {step}")
                curve.append({"step": step, "loss": round(recent, 4), **p})
                model.train()
            if step >= steps_target:
                break

    print(f"\n=== done {step} steps in {time.perf_counter() - t0:.0f}s ===")
    print(json.dumps(curve, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
