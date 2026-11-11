"""Benchmark the reference trainer at *realistic* sequence length and batch size.

The first version of this script used seq=128, which badly over-estimated
throughput: real SFT sequences are ~283-415 tokens (system prompt + DDL block +
question + tool-call target). Since CPU training is the binding constraint on the
whole experiment matrix (1 main run + 3 ablation rungs + 4 scale points), the
config must be chosen from realistic numbers.

    $env:PYTHONPATH='src'; python scripts/bench_train.py --realistic
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fcdata.config import load_config
from fcdata.train.reference import build_model, masked_lm_loss


def bench_one(nl: int, nh: int, dm: int, dff: int, vocab: int,
              batch: int, seq: int, iters: int = 6) -> dict:
    cfg = load_config("configs/demo.yaml")
    cfg.train.n_layer = nl
    cfg.train.n_head = nh
    cfg.train.d_model = dm
    cfg.train.d_ff = dff
    cfg.train.vocab_size = vocab
    model = build_model(cfg)
    model.train()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    v = min(300, vocab)
    x = torch.randint(0, v, (batch, seq))
    y = torch.randint(0, v, (batch, seq))

    def step() -> None:
        logits = model(x)
        loss, _acc, _n = masked_lm_loss(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    for _ in range(2):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = (time.perf_counter() - t0) / iters
    return {
        "n_layer": nl, "n_head": nh, "d_model": dm, "d_ff": dff,
        "vocab_size": vocab, "batch": batch, "seq": seq,
        "params": total, "trainable": trainable,
        "ms_per_step": round(dt * 1000, 1),
        "steps_per_sec": round(1 / dt, 2),
        "records_per_sec": round(batch / dt, 1),
    }


REALISTIC = [
    # (n_layer, n_head, d_model, d_ff, vocab, batch, seq)
    (4, 4, 128, 512, 2048, 16, 448),
    (4, 4, 128, 512, 2048, 32, 448),
    (4, 4, 128, 512, 2048, 48, 448),
    (4, 4, 128, 512, 2048, 64, 448),
    (3, 4, 128, 384, 2048, 32, 448),
    (4, 4, 96, 384, 2048, 32, 448),
]


def main(argv: list[str] | None = None) -> int:
    rows = [bench_one(*c) for c in REALISTIC]
    print(f"{'config':30s} {'B':>4s} {'S':>4s} {'params':>10s} {'ms/step':>8s} "
          f"{'steps/s':>8s} {'rec/s':>7s}")
    print("-" * 84)
    for r in rows:
        name = f"L{r['n_layer']} d{r['d_model']} f{r['d_ff']} v{r['vocab_size']}"
        print(f"{name:30s} {r['batch']:>4d} {r['seq']:>4d} {r['params']:>10,} "
              f"{r['ms_per_step']:>8.1f} {r['steps_per_sec']:>8.2f} "
              f"{r['records_per_sec']:>7.1f}")

    # translate to wall-clock for the actual experiment matrix
    print("\nprojected wall clock (490-record corpus, 4 epochs = 124 steps):")
    for r in rows:
        per_run = 124 * r["ms_per_step"] / 1000
        print(f"  L{r['n_layer']} d{r['d_model']} B{r['batch']}: "
              f"{per_run:5.1f}s per run x 8 runs = {per_run * 8 / 60:5.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
