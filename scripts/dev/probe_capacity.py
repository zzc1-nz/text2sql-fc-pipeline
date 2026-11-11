"""Capacity probe: how many training steps does the reference model need?

The pipeline's numbers are only meaningful if the model can actually learn the
task. This script answers three questions with measurements, not guesses:

1. Can the model memorise a small slice (i.e. is capacity sufficient)?
2. How many steps does that take?
3. Does a bigger model or a shorter target help more?

Run:  $env:PYTHONPATH='src'; python scripts/probe_capacity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fcdata.config import load_config
from fcdata.dbexec import evaluate_pair
from fcdata.records import read_sft
from fcdata.spider import load_spider
from fcdata.sqlutils import extract_sql
from fcdata.train import data as tdata
from fcdata.train.reference import (build_model, generate, masked_lm_loss,
                                    save_checkpoint)
from fcdata.train.vocab import WordTokenizer


def probe(n_layer: int, d_model: int, d_ff: int, n_head: int,
          n_records: int, steps: int, lr: float, batch: int = 8,
          kind_filter: str | None = None) -> dict:
    cfg = load_config("configs/demo.yaml")
    cfg.train.n_layer = n_layer
    cfg.train.d_model = d_model
    cfg.train.d_ff = d_ff
    cfg.train.n_head = n_head

    data = load_spider(cfg.paths.fixture_dir)
    sft = read_sft(cfg.paths.sft_dir / "train.jsonl")
    if kind_filter:
        sft = [r for r in sft if r.kind == kind_filter]
    sub = sft[:n_records]

    tok = WordTokenizer().fit(tdata.corpus_texts(sft), 2048)
    encs, _w = tdata.build_encoded_dataset(sub, tok, max_seq_len=cfg.train.max_seq_len)
    model = build_model(cfg, vocab_size=tok.vocab_size)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    n_params = sum(p.numel() for p in model.parameters())

    batch_list = [encs[i:i + batch] for i in range(0, len(encs), batch)]
    t0 = time.perf_counter()
    history = []
    step = 0
    while step < steps:
        for b in batch_list:
            if step >= steps:
                break
            maxlen = max(len(e.input_ids) for e in b)
            ids = torch.full((len(b), maxlen), 0, dtype=torch.long)
            lab = torch.full((len(b), maxlen), -100, dtype=torch.long)
            for i, e in enumerate(b):
                ids[i, :len(e.input_ids)] = torch.tensor(e.input_ids)
                lab[i, :len(e.labels)] = torch.tensor(e.labels)
            logits = model(ids)
            loss, acc, _n = masked_lm_loss(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % max(1, steps // 8) == 0 or step == steps:
                history.append({"step": step, "loss": round(float(loss), 4),
                                "tok_acc": round(float(acc), 4)})
                print(f"   step {step:4d}/{steps}  loss {float(loss):7.4f}  "
                      f"tok_acc {float(acc):.3f}")
    elapsed = time.perf_counter() - t0

    # evaluate on the training slice itself: this is memorisation, not generalisation
    model.eval()
    n_exec = n_match = 0
    sample = None
    for r in sub:
        prompt = tdata.render_prompt_text(r, template="qwen")
        out = generate(model, tok, prompt, max_new_tokens=140)
        sql = extract_sql(out)
        ok = bool(sql) and evaluate_pair(data.db_path(r.db_id), sql, r.target_sql)["exec_match"]
        n_match += int(ok)
        n_exec += int(bool(sql))
        if sample is None:
            sample = {"gold": r.target_sql[:110], "pred": sql[:110], "raw": out[:160]}

    return {
        "n_layer": n_layer, "d_model": d_model, "n_head": n_head,
        "params": n_params, "n_records": len(sub), "steps": steps,
        "lr": lr, "batch": batch, "kind_filter": kind_filter,
        "final_loss": history[-1]["loss"] if history else None,
        "final_tok_acc": history[-1]["tok_acc"] if history else None,
        "memorised_exec_match": n_match, "memorised_exec_acc": round(n_match / max(1, len(sub)), 3),
        "steps_per_sec": round(steps / elapsed, 2), "wall_s": round(elapsed, 1),
        "sample": sample,
    }


PROBES = [
    # name,                       kwargs
    ("small_40rec_200steps", dict(n_layer=4, d_model=128, d_ff=512, n_head=4,
                                  n_records=40, steps=200, lr=1e-3)),
    ("small_40rec_600steps", dict(n_layer=4, d_model=128, d_ff=512, n_head=4,
                                  n_records=40, steps=600, lr=1e-3)),
]


def main(argv: list[str] | None = None) -> int:
    results = []
    for name, kwargs in PROBES:
        print(f"\n=== {name} ===")
        r = probe(**kwargs)
        r["name"] = name
        results.append(r)
        print(f"   -> params {r['params']:,}  loss {r['final_loss']}  "
              f"tok_acc {r['final_tok_acc']}  memorised exec_acc {r['memorised_exec_acc']:.1%}")
        print(f"   -> sample pred: {r['sample']['pred']!r}")
    out = Path("artifacts/diag/probe_capacity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
