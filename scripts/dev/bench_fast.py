"""Find a training configuration that is BOTH fast enough and actually learnable.

Two hard constraints in this CPU-only environment:

* the model must be trained long enough to generalise (the capacity probe shows
  ~600 steps are needed before SQL generation becomes coherent), and
* the full experiment matrix is 1 main run + 3 ablation rungs + 4 scale points,
  so a run must cost minutes, not hours.

This script measures throughput for smaller architectures at the real sequence
length (576) so the demo config is chosen from data.

    $env:PYTHONPATH='src'; python scripts/bench_fast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fcdata.config import load_config
from fcdata.records import read_sft
from fcdata.train import data as tdata
from fcdata.train.reference import build_model, masked_lm_loss
from fcdata.train.vocab import WordTokenizer

CANDS = [
    # (n_layer, n_head, d_model, d_ff)
    (4, 4, 128, 512),
    (3, 4, 128, 512),
    (2, 4, 128, 512),
    (3, 4, 96, 384),
    (2, 4, 96, 384),
]


def main() -> int:
    torch.set_num_threads(4)
    cfg = load_config("configs/demo.yaml")
    sft = read_sft(cfg.paths.sft_dir / "train.jsonl")
    tok = WordTokenizer().fit(tdata.corpus_texts(sft), 2048)
    encs, _w = tdata.build_encoded_dataset(sft, tok, max_seq_len=cfg.train.max_seq_len)

    # build one real padded batch at the true sequence length
    batch = encs[:16]
    maxlen = max(len(e.input_ids) for e in batch)
    ids = torch.zeros((len(batch), maxlen), dtype=torch.long)
    lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, :len(e.input_ids)] = torch.tensor(e.input_ids)
        lab[i, :len(e.labels)] = torch.tensor(e.labels)
    print(f"real batch shape: {tuple(ids.shape)} (max_seq_len {cfg.train.max_seq_len})")
    print(f"{'config':22s} {'params':>10s} {'ms/step':>9s} {'steps/s':>8s} "
          f"{'min/600':>8s} {'min/1500':>9s}")
    print("-" * 74)
    for nl, nh, dm, dff in CANDS:
        cfg.train.n_layer, cfg.train.n_head = nl, nh
        cfg.train.d_model, cfg.train.d_ff = dm, dff
        model = build_model(cfg, vocab_size=tok.vocab_size)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        n = sum(p.numel() for p in model.parameters())

        def step() -> None:
            logits = model(ids)
            loss, _a, _s = masked_lm_loss(logits, lab)
            loss.backward()
            opt.step()
            opt.zero_grad()

        for _ in range(2):
            step()
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            step()
        dt = (time.perf_counter() - t0) / iters
        print(f"L{nl} d{dm} f{dff} v{tok.vocab_size:<4d} {n:>10,} {dt * 1000:>9.0f} "
              f"{1 / dt:>8.2f} {600 * dt / 60:>8.1f} {1500 * dt / 60:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
