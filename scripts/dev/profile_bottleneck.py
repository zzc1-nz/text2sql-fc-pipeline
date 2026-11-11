"""Find the actual bottleneck in the reference step (dev tool).

Suspicion: for a ~1M-parameter model, 0.37 steps/s is far slower than the FLOP
count implies, so something structural is wrong (thread oversubscription, the
LM head over all positions, or the loss recomputing argmax over the full vocab).

Run:  $env:PYTHONPATH='src'; python scripts/profile_bottleneck.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fcdata.train.reference import build_model, masked_lm_loss  # noqa: E402


def bench(fn, n: int = 5) -> float:
    """Mean seconds per call, first call excluded (warm-up)."""
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> int:
    torch.manual_seed(0)
    vocab, bs, seq = 473, 16, 380
    cfg = dict(n_layer=4, n_head=4, d_model=128, d_ff=512, max_seq_len=512,
               lora_r=8, lora_alpha=16, dropout=0.1)
    model = build_model(cfg, vocab_size=vocab)
    ids = torch.randint(5, vocab, (bs, seq))
    labels = ids.clone()
    labels[:, : seq // 3] = -100

    print(f"params={sum(p.numel() for p in model.parameters()):,} "
          f"batch={bs} seq={seq} vocab={vocab}")

    for n_threads in (1, 4, 8, 14):
        torch.set_num_threads(n_threads)
        model.train()
        fwd = bench(lambda: model(ids))
        def step() -> None:
            model.zero_grad(set_to_none=True)
            loss, _, _ = masked_lm_loss(model(ids), labels)
            loss.backward()
        full = bench(step, 3)
        print(f"\nthreads={n_threads}")
        print(f"  forward          {1000 * fwd:8.1f} ms")
        print(f"  fwd+bwd (no opt) {1000 * full:8.1f} ms")
        print(f"  steps/s          {1.0 / full:8.3f}")

    # ---- where inside the forward? ----
    torch.set_num_threads(8)
    model.eval()
    with torch.no_grad():
        emb = bench(lambda: model.tok_emb(ids) + model.pos_emb(
            torch.arange(seq))[None, :, :])
        x = model.tok_emb(ids) + model.pos_emb(torch.arange(seq))[None, :, :]
        blk = model.blocks[0]
        blk_t = bench(lambda: blk(x, model.causal_mask))
        blocks_all = bench(lambda: [b(x, model.causal_mask) for b in model.blocks])
        head_t = bench(lambda: model.lm_head(model.ln_f(x)))
        full_fwd = bench(lambda: model(ids))
    print("\n=== forward breakdown (threads=8, eval/no_grad) ===")
    print(f"  embeddings       {1000 * emb:8.1f} ms")
    print(f"  1 block          {1000 * blk_t:8.1f} ms")
    print(f"  all {len(model.blocks)} blocks     {1000 * blocks_all:8.1f} ms")
    print(f"  lm_head+ln       {1000 * head_t:8.1f} ms")
    print(f"  total forward    {1000 * full_fwd:8.1f} ms")
    print(f"  -> head share    {100.0 * head_t / full_fwd:8.1f} %")

    # ---- cost of the loss itself ----
    logits = model(ids)
    loss_t = bench(lambda: masked_lm_loss(logits, labels))
    ce_only = bench(lambda: F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1), ignore_index=-100))
    print("\n=== loss cost ===")
    print(f"  masked_lm_loss   {1000 * loss_t:8.1f} ms")
    print(f"  ce only          {1000 * ce_only:8.1f} ms")

    # ---- how much would only-supervised logits save? ----
    n_sup = int((labels != -100).sum())
    print(f"\n  supervised positions: {n_sup} of {bs * seq} "
          f"({100.0 * n_sup / (bs * seq):.0f}%)")

    # ---- batch/seq padding waste ----
    print("\n=== padding waste in the real corpus ===")
    lens = [313] * 16
    print(f"  mean seq 313, padded to max in batch -> "
          f"{100.0 * (sum(lens) / (16 * max(lens))):.0f}% utilisation (batch of 16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
