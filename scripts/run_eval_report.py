"""Developer driver: run the eval -> ablation -> scale -> report chain for real.

This is a thin orchestration script used while developing the report artifacts.
It is intentionally NOT part of the package: the CLI (owned by the Lead) calls
the same public functions. Keeping the driver here means the exact command that
produced ``reports/`` is reproducible and reviewable.

Usage (from the repo root)::

    $env:PYTHONPATH='src'
    python scripts/run_eval_report.py --max-steps 60 --eval-examples 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from fcdata.config import ensure_dirs, load_config
from fcdata.eval.ablation import run_ablation
from fcdata.eval.scale import DEFAULT_FRACTIONS, run_scale_curve
from fcdata.report import (
    build_report,
    collect_artifacts,
    collect_ledger_summary,
    find_train_metrics,
)
from fcdata.spider import load_spider


def _bump_budget(cfg: Any, max_steps: int, eval_examples: int,
                 max_seq_len: int) -> None:
    """Apply a realistic-but-bounded training/eval budget for the artifact run.

    The budget is chosen from measured facts, not guesswork:

    * ``max_seq_len`` must cover prompt+target for the *median* record, otherwise
      the truncation silently removes part of the DDL from the system message and
      the model cannot know which tables exist. Measured on the fixture corpus:
      total mean 343 tokens, max 555, so 576 is the floor.
    * the reference CPU model trains from scratch, so step count (not width) is
      the binding constraint; a wide-but-undertrained model learns only the
      output format.

    The per-arm epoch count needed to make ``max_steps`` reachable is derived in
    :func:`fcdata.eval.pipeline._cfg_with_reachable_steps`, because each ablation
    arm has a different record count.
    """
    cfg.train.max_steps = int(max_steps)
    cfg.train.max_seq_len = int(max_seq_len)
    cfg.ev.max_examples = int(eval_examples)
    cfg.ablation.dev_sample = int(eval_examples)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--fixture", default="data/fixture")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-examples", type=int, default=24)
    ap.add_argument("--max-seq-len", type=int, default=576,
                    help="must cover prompt+target; 576 is the measured floor")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fractions", default="0,0.25,0.5,1.0")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-scale", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, root=".")
    ensure_dirs(cfg)
    spider = load_spider(args.fixture)
    _bump_budget(cfg, args.max_steps, args.eval_examples, args.max_seq_len)
    cfg.train.n_layer = args.n_layer
    cfg.train.d_model = args.d_model
    cfg.train.n_head = next((h for h in (8, 4, 2) if args.d_model % h == 0), 2)
    cfg.train.d_ff = args.d_model * 4
    cfg.train.lr = args.lr
    print(f"[driver] fixture: {spider.stats()}")
    print(f"[driver] budget: max_steps={cfg.train.max_steps} "
          f"seq={cfg.train.max_seq_len} model=L{cfg.train.n_layer}/"
          f"d{cfg.train.d_model} lr={cfg.train.lr} "
          "(per-arm epochs derived to make max_steps reachable)")

    eval_dir = Path(cfg.paths.eval_dir)
    ablation: dict[str, Any] | None = None
    scale: dict[str, Any] | None = None

    if not args.skip_ablation:
        t0 = time.perf_counter()
        ablation = run_ablation(cfg, spider, cfg.ablation, eval_dir)
        print(f"[driver] ablation done in {time.perf_counter() - t0:.1f}s")

    if not args.skip_scale:
        fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
        t0 = time.perf_counter()
        scale = run_scale_curve(cfg, spider, fractions or list(DEFAULT_FRACTIONS),
                                eval_dir)
        print(f"[driver] scale done in {time.perf_counter() - t0:.1f}s")

    summary = build_report(
        cfg, ablation, scale, collect_ledger_summary(cfg),
        Path(cfg.paths.report_path),
        artifacts=collect_artifacts(cfg),
        train_metrics=find_train_metrics(cfg),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "csv"}, indent=2,
                     default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
