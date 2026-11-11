"""Can the reference model memorise a SMALL set? (separates bug from data-scale)

Trains on 8 records and checks whether greedy generation reproduces their gold SQL.
If it can, the train/generate path is sound and held-out failure is a capacity /
data-scale story. If it cannot, there is a parity bug.

Run:  $env:PYTHONPATH='src'; python scripts/check_memorize_small.py [n_records] [steps]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from fcdata.config import load_config  # noqa: E402
from fcdata.dbexec import execute_sql  # noqa: E402
from fcdata.format import build_sft_records, render_record_prompt  # noqa: E402
from fcdata.records import SynthRecord, ValidationReport, ValidatedRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.sqlutils import extract_sql, normalize_sql  # noqa: E402
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
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    steps_target = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    cfg = load_config("configs/demo.yaml", root=".")
    data = load_spider("data/fixture")
    sft, _ = build_sft_records(cfg, build_validated(data), data, data.train)
    template = cfg.fmt.chat_template
    tok = WordTokenizer().fit(corpus_texts(sft, template=template),
                              cfg.train.vocab_size)

    tcfg = cfg.train
    tcfg.vocab_size = tok.vocab_size
    tcfg.max_seq_len = 576
    subset = sft[:n_records]
    encoded, weights = build_encoded_dataset(subset, tok,
                                             max_seq_len=tcfg.max_seq_len,
                                             template=template)
    print(f"n_records={len(encoded)} vocab={tok.vocab_size} "
          f"seq_len_max={max(len(e.input_ids) for e in encoded)}")

    model = build_model(tcfg, vocab_size=tok.vocab_size)
    print(f"params={describe_model(model)['total_params']:,} "
          f"lr={tcfg.lr} bs={tcfg.batch_size}")
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    bs = min(tcfg.batch_size, len(encoded))

    def probe(tag: str) -> tuple[int, int]:
        model.eval()
        prompts = [render_record_prompt(r, template) for r in subset]
        outs = generate_batch(model, tok, prompts, max_new_tokens=90,
                              temperature=0.0)
        exact = exec_ok = 0
        for rec, out in zip(subset, outs):
            sql = extract_sql(out)
            if sql and normalize_sql(sql) == normalize_sql(rec.target_sql):
                exact += 1
            if sql:
                try:
                    if execute_sql(data.db_path(rec.db_id), sql).ok:
                        exec_ok += 1
                except FileNotFoundError:
                    pass
        print(f"  [{tag}] exact {exact}/{len(subset)}  executes {exec_ok}/{len(subset)}")
        return exact, exec_ok

    model.train()
    t0 = time.perf_counter()
    step = 0
    losses: list[float] = []
    import random
    rng = random.Random(17)
    checks = sorted({s for s in (50, 100, 200, 300, 400, 600) if s <= steps_target})
    best_exact = 0
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
                recent = sum(losses[-20:]) / len(losses[-20:])
                print(f"\nstep {step}  loss {recent:.4f}  tok_acc {acc:.3f}  "
                      f"({time.perf_counter() - t0:.0f}s)")
                ex, _ = probe(f"step {step}")
                best_exact = max(best_exact, ex)
                model.train()
            if step >= steps_target:
                break

    print(f"\n=== {step} steps, {time.perf_counter() - t0:.0f}s, "
          f"best exact {best_exact}/{len(subset)} ===")
    model.eval()
    outs = generate_batch(model, tok,
                          [render_record_prompt(r, template) for r in subset[:3]],
                          max_new_tokens=90, temperature=0.0)
    for rec, out in list(zip(subset[:3], outs)):
        print(f"\n  gold={rec.target_sql}")
        print(f"  got ={extract_sql(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
