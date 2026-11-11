"""Probe: can the reference CPU model reach non-zero execution accuracy at all?

This is a diagnostic, not part of the pipeline. It trains on a *reduced*,
higher-repetition subset of the SFT corpus (fewer databases => fewer distinct
gold SQL => the mapping is easier to learn) and then scores the dev split with
the real evaluator. The point is to separate two very different explanations for
an all-zero ablation:

* the evaluation harness is broken (it would report 0.0 even for a capable
  model), or
* the reference model simply has not learned the task within the CPU budget.

If this probe reaches non-zero exec accuracy, the harness is proven to surface
capability when it exists, and the ablation's zeros are a model/budget finding
rather than a measurement bug.

Usage::

    $env:PYTHONPATH='src'
    python scripts/probe_learning.py --dbs college,restaurant --max-steps 700
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from fcdata.config import load_config
from fcdata.eval.infer import evaluate_checkpoint
from fcdata.records import SFTRecord, read_sft
from fcdata.spider import load_spider
from fcdata.train import train
from fcdata.train.reference import load_model


def _pick(records: list[SFTRecord], dbs: set[str]) -> list[SFTRecord]:
    return [r for r in records if r.db_id in dbs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--fixture", default="data/fixture")
    ap.add_argument("--dbs", default="college,restaurant",
                    help="comma-separated db_ids to train on")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=576)
    ap.add_argument("--out", default=".scratch/probe_learning")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, root=".")
    cfg.train.n_layer = args.n_layer
    # n_head must divide d_model; pick the largest divisor <= 8 that is >= 2.
    cfg.train.n_head = next((h for h in (8, 4, 2) if args.d_model % h == 0), 2)
    cfg.train.d_model = args.d_model
    cfg.train.d_ff = args.d_model * 4
    cfg.train.max_seq_len = args.max_seq_len
    cfg.train.batch_size = 16
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr

    spider = load_spider(args.fixture)
    dbs = {d.strip() for d in args.dbs.split(",") if d.strip()}
    train_records = _pick(read_sft(Path(cfg.paths.sft_dir) / "train.jsonl"), dbs)
    if not train_records:
        print(f"[probe] no SFT records for dbs={sorted(dbs)}", file=sys.stderr)
        return 2
    distinct_sql = len({r.target_sql for r in train_records})
    print(f"[probe] training on {len(train_records)} records "
          f"({distinct_sql} distinct gold SQL) from dbs={sorted(dbs)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics = train(cfg, train_records, None, out / "train",
                    max_steps=args.max_steps)
    print(f"[probe] trained {len(metrics['history'])} steps in "
          f"{time.perf_counter() - t0:.1f}s, final loss "
          f"{metrics['final_train_loss']:.3f}, token_acc "
          f"{metrics['final_train_token_acc']:.3f}")

    model, tokenizer = load_model(out / "train" / "model.pt")
    # score the dev examples that belong to the trained databases
    dev = [e for e in spider.dev if e.db_id in dbs]
    if not dev:
        print("[probe] no dev examples for those dbs", file=sys.stderr)
        return 2
    result = evaluate_checkpoint(cfg, out / "train", spider, dev, out / "eval",
                                 model=model, tokenizer=tokenizer, progress=True)
    print(f"[probe] exec_acc={result['exec_acc']:.4f} "
          f"exact_match={result['exact_match']:.4f} "
          f"executable_rate={result['executable_rate']:.4f} n={result['n']}")
    return 0 if result["exec_acc"] > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
