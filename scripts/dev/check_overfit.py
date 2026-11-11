"""Prove the reference backend can actually learn (the overfit gate).

This is the experiment that exposed the LoRA-only design flaw: a randomly
initialised model adapted through a rank-8 bottleneck cannot learn the task, while
end-to-end training can.

Run:  $env:PYTHONPATH='src'; python scripts/check_overfit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from fcdata.config import load_config  # noqa: E402
from fcdata.format import build_sft_records  # noqa: E402
from fcdata.records import SynthRecord, ValidationReport, ValidatedRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train.data import build_encoded_dataset, collate, corpus_texts  # noqa: E402
from fcdata.train.reference import (  # noqa: E402
    build_model,
    describe_model,
    generate,
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


def overfit(cfg, encoded, weights, n_steps: int, label: str) -> dict:
    """Train on ONE record and report whether the model can memorise it."""
    model = build_model(cfg, vocab_size=cfg.vocab_size)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    batch = collate(encoded[:1], weights[:1])
    model.train()

    losses = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss, acc, _ = masked_lm_loss(model(batch.input_ids), batch.labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0

    model.eval()
    prompt_len = encoded[0].prompt_len
    prompt_ids = encoded[0].input_ids[:prompt_len]
    gen = generate(model, TOK, TOK.decode(prompt_ids), max_new_tokens=40,
                   temperature=0.0)
    target = TOK.decode(encoded[0].input_ids[prompt_len:])

    print(f"\n--- {label} ---")
    print(f"  trainable      : {describe_model(model)['trainable_params']:,} "
          f"of {describe_model(model)['total_params']:,}")
    print(f"  loss           : {losses[0]:.3f} -> {losses[-1]:.3f}  "
          f"(ln(V)={torch.log(torch.tensor(float(cfg.vocab_size))):.2f})")
    print(f"  wall           : {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    print(f"  target         : {target[:150]!r}")
    print(f"  generated      : {gen[:150]!r}")
    overlap = len(set(gen.split()) & set(target.split())) / max(1, len(set(target.split())))
    print(f"  token overlap  : {overlap:.0%}")
    return {"label": label, "first": losses[0], "last": losses[-1],
            "gen": gen, "target": target, "overlap": overlap,
            "trainable": describe_model(model)["trainable_params"]}


def main() -> int:
    global TOK
    cfg_all = load_config("configs/demo.yaml", root=".")
    data = load_spider("data/fixture")
    sft, _ = build_sft_records(cfg_all, build_validated(data), data, data.train)
    template = cfg_all.fmt.chat_template
    TOK = WordTokenizer().fit(corpus_texts(sft, template=template),
                              cfg_all.train.vocab_size)
    print(f"sft={len(sft)} vocab={TOK.vocab_size}")

    base = cfg_all.train
    base.vocab_size = TOK.vocab_size
    base.max_seq_len = 512
    encoded, weights = build_encoded_dataset(sft, TOK, max_seq_len=base.max_seq_len,
                                             template=template)
    print(f"encoded={len(encoded)} seq_len={len(encoded[0].input_ids)}")

    results = []
    for lora_only, n_steps in ((True, 40), (False, 40)):
        import copy
        c = copy.deepcopy(base)
        c.lora_only = lora_only
        c.lr = 3e-3
        results.append(overfit(c, encoded, weights, n_steps,
                               f"lora_only={lora_only}"))

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for r in results:
        verdict = "LEARNS" if r["last"] < 2.0 else "CANNOT LEARN"
        print(f"  {r['label']:<20} loss {r['first']:.2f} -> {r['last']:.2f}   {verdict}")
    best = min(results, key=lambda r: r["last"])
    print(f"\n  best: {best['label']} (loss {best['last']:.3f})")
    return 0 if best["last"] < 2.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
