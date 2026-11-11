"""Command-line entry point: one subcommand per pipeline stage.

Design: every stage is independently runnable and idempotent, reading its inputs
from ``artifacts/`` and writing its outputs back there. That makes the pipeline
debuggable (re-run just the validator after changing a judge prompt) and lets the
ablation runner reuse cached synthesis instead of paying for it three times.

    python -m fcdata.cli fixture                    # build the offline dataset
    python -m fcdata.cli synth                      # CoT + paraphrase + hard negatives
    python -m fcdata.cli validate                   # format + semantic checks
    python -m fcdata.cli format                     # -> function-calling SFT records
    python -m fcdata.cli train                      # LoRA SFT
    python -m fcdata.cli eval                       # exec accuracy / exact match
    python -m fcdata.cli ablation                   # raw_only vs +CoT vs +hard negatives
    python -m fcdata.cli scale                      # data-scale curve
    python -m fcdata.cli report                     # tables + figures
    python -m fcdata.cli all                        # the whole thing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .config import Config, dump_config, ensure_dirs, load_config, scratch_dir
from .records import (Example, SynthRecord, read_json, read_jsonl, read_sft,
                      read_synth_records, read_validated, write_json,
                      write_jsonl)
from .tokens import TokenLedger, summarize_ledger


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}", flush=True)


def _step(msg: str) -> None:
    print(f"  -> {msg}", flush=True)


def _prices(cfg: Config) -> dict[str, tuple[float, float]]:
    return {role: (p.price_in_per_mtok, p.price_out_per_mtok)
            for role, p in cfg.providers.items()}


def _ledger(cfg: Config, name: str = "ledger.jsonl", reset: bool = False) -> TokenLedger:
    return TokenLedger(cfg.paths.artifacts_dir + "/" + name, prices=_prices(cfg),
                       reset_file=reset)


def _spider(cfg: Config):
    from .spider import load_spider
    root = Path(cfg.paths.spider_dir)
    if not (root / "tables.json").exists():
        fixture = Path(cfg.paths.fixture_dir)
        if not (fixture / "tables.json").exists():
            raise SystemExit(
                f"No dataset found. Looked for {root}/tables.json and {fixture}/tables.json.\n"
                f"Run: python -m fcdata.cli fixture")
        print(f"  [info] {root} not found; using bundled fixture {fixture}", flush=True)
        root = fixture
    return load_spider(root)


def _raw_examples(cfg: Config) -> tuple[list[Example], list[Example]]:
    """Load (train, dev) examples, caching them into artifacts/01_raw."""
    train_p = cfg.paths.raw_dir / "train.jsonl"
    dev_p = cfg.paths.raw_dir / "dev.jsonl"
    if train_p.exists() and dev_p.exists():
        train = [Example(**d) for d in read_jsonl(train_p)]
        dev = [Example(**d) for d in read_jsonl(dev_p)]
        return train, dev
    data = _spider(cfg)
    write_jsonl(train_p, data.train)
    write_jsonl(dev_p, data.dev)
    return data.train, data.dev


# --------------------------------------------------------------------------- #
# stage: fixture
# --------------------------------------------------------------------------- #
def cmd_fixture(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .build_fixture import build
    _banner("STAGE 0  build offline Spider-format dataset")
    summary = build(Path(cfg.paths.fixture_dir), dev_ratio=args.dev_ratio, seed=cfg.seed)
    print(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# stage: synth
# --------------------------------------------------------------------------- #
def cmd_synth(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .synth import build_synth_records
    _banner("STAGE 1  data synthesis (CoT / paraphrase / hard negatives)")
    train, _dev = _raw_examples(cfg)
    if cfg.synth.max_source_examples:
        train = train[: cfg.synth.max_source_examples]
    _step(f"{len(train)} source examples; plan = "
          f"{cfg.synth.cot_per_example} CoT + {cfg.synth.paraphrase_per_example} paraphrase "
          f"+ {cfg.synth.hard_negative_per_example} hard negative per example")
    data = _spider(cfg)
    ledger = _ledger(cfg, "synth_ledger.jsonl", reset=True)
    t0 = time.time()
    records = build_synth_records(cfg, data, ledger)
    dt = time.time() - t0
    out = cfg.paths.synth_dir / "synth.jsonl"
    n = write_jsonl(out, records)
    agg = ledger.aggregate()
    totals = ledger.totals()
    ledger.dump(cfg.paths.synth_dir / "ledger_summary.json")
    _step(f"wrote {n} synthetic records -> {out}  ({dt:.1f}s)")
    _step(f"tokens spent: {totals['total_tokens']:,} "
          f"(prompt {totals['prompt_tokens']:,} / completion {totals['completion_tokens']:,}) "
          f"= ${totals['cost_usd']:.4f}")
    for stage, st in agg.items():
        _step(f"  stage {stage}: {st['total_tokens']:,} tokens over {st['calls']} calls")
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    _step(f"by kind: {kinds}")
    summary = {"n_records": n, "by_kind": kinds, "tokens": totals,
               "elapsed_s": round(dt, 2), "ledger": agg}
    write_json(cfg.paths.synth_dir / "synth_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# stage: validate
# --------------------------------------------------------------------------- #
def cmd_validate(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .validate import validate_and_write
    _banner("STAGE 2  multi-stage validation (format + semantic)")
    synth_p = cfg.paths.synth_dir / "synth.jsonl"
    if not synth_p.exists():
        raise SystemExit(f"missing {synth_p}; run `synth` first")
    records = read_synth_records(synth_p)
    _step(f"{len(records)} synthetic records to validate")
    data = _spider(cfg)
    ledger = _ledger(cfg, "validate_ledger.jsonl", reset=True)
    t0 = time.time()
    report = validate_and_write(cfg, records, data, ledger=ledger,
                                out_dir=cfg.paths.validated_dir)
    dt = time.time() - t0
    ledger.dump(cfg.paths.validated_dir / "ledger_summary.json")
    headline = report.get("headline", {})
    _step(f"validated -> {cfg.paths.validated_dir}  ({dt:.1f}s)")
    _step(f"format pass rate: {headline.get('format_pass_rate_pct', 0):.1f}%   "
          f"combined pass rate: {headline.get('combined_pass_rate_pct', 0):.1f}%")
    for row in report.get("funnel", []):
        _step(f"  funnel {row.get('stage', '?'):18s} {row.get('n', 0):6d}  "
              f"{row.get('pct', 0):5.1f}%")
    _step(f"judge tokens: {ledger.totals()['total_tokens']:,} "
          f"(${ledger.totals()['cost_usd']:.4f})")
    return report


# --------------------------------------------------------------------------- #
# stage: format
# --------------------------------------------------------------------------- #
def cmd_format(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .format import build_and_write
    _banner("STAGE 3  convert to function-calling SFT records")
    val_p = cfg.paths.validated_dir / "validated.jsonl"
    if not val_p.exists():
        raise SystemExit(f"missing {val_p}; run `validate` first")
    validated = read_validated(val_p)
    data = _spider(cfg)
    stats = build_and_write(cfg, validated, data,
                            raw_examples=data.train + data.dev,
                            out_dir=cfg.paths.sft_dir)
    _step(f"SFT records written -> {cfg.paths.sft_dir}")
    _step(f"kind mix: {stats.get('by_kind', {})}")
    _step(f"tool-call validity: {stats.get('tool_call_valid_rate', 0):.3f}")
    return stats


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def cmd_train(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .train import train
    _banner("STAGE 4  LoRA SFT")
    tr_p = cfg.paths.sft_dir / "train.jsonl"
    dv_p = cfg.paths.sft_dir / "dev.jsonl"
    if not tr_p.exists():
        raise SystemExit(f"missing {tr_p}; run `format` first")
    train_recs = read_sft(tr_p)
    dev_recs = read_sft(dv_p) if dv_p.exists() else []
    if cfg.train.backend == "reference" and args.smoke:
        train_recs = train_recs[: max(8, len(train_recs) // 4)]
        cfg.train.epochs = 1
    out = Path(cfg.paths.train_dir) / (args.run_name or "main")
    _step(f"backend={cfg.train.backend}  train={len(train_recs)}  dev={len(dev_recs)}  "
          f"epochs={cfg.train.epochs}  out={out}")
    t0 = time.time()
    metrics = train(cfg, train_recs, dev_recs, out)
    dt = time.time() - t0
    _step(f"trained in {dt:.1f}s; train_loss={metrics.get('final_train_loss'):.4f} "
          f"train_token_acc={metrics.get('final_train_token_acc')} "
          f"dev_loss={metrics.get('final_dev_loss')} "
          f"dev_token_acc={metrics.get('final_dev_token_acc')}")
    _step(f"params: total={metrics.get('total_params'):,} "
          f"trainable={metrics.get('trainable_params'):,} "
          f"({metrics.get('trainable_pct')}%)")
    return metrics


# --------------------------------------------------------------------------- #
# stage: eval
# --------------------------------------------------------------------------- #
def cmd_eval(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .eval import evaluate_model
    _banner("STAGE 5  evaluation (real sqlite execution)")
    run_dir = Path(cfg.paths.train_dir) / (args.run_name or "main")
    if not (run_dir / "model.pt").exists():
        raise SystemExit(f"missing checkpoint in {run_dir}; run `train` first")
    data = _spider(cfg)
    examples = data.dev
    if cfg.ev.max_examples:
        examples = examples[: cfg.ev.max_examples]
    out = Path(cfg.paths.eval_dir) / (args.run_name or "main")
    _step(f"evaluating {len(examples)} dev examples")
    metrics = evaluate_model(cfg, run_dir, data, examples, out)
    _step(f"exec_acc={metrics.get('exec_acc', 0):.3f}  "
          f"exact_match={metrics.get('exact_match', 0):.3f}  "
          f"executable={metrics.get('executable_rate', 0):.3f}")
    return metrics


# --------------------------------------------------------------------------- #
# stage: ablation
# --------------------------------------------------------------------------- #
def cmd_ablation(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .eval.ablation import run_ablation
    _banner("STAGE 6  ablation: raw only vs +CoT vs +hard negatives")
    data = _spider(cfg)
    out = Path(cfg.paths.eval_dir) / "ablation"
    _step(f"configs={cfg.ablation.configs}")
    result = run_ablation(cfg, data, cfg.ablation, out, quick=args.smoke)
    for row in result.get("rows", []):
        _step(f"  {row.get('config', '?'):22s} n={row.get('n_sft_records', 0):5d} "
              f"exec_acc={row.get('exec_acc', 0):.3f}  "
              f"em={row.get('exact_match', 0):.3f}  "
              f"synth_tokens={row.get('synth_tokens', 0):,}")
    return result


# --------------------------------------------------------------------------- #
# stage: scale
# --------------------------------------------------------------------------- #
def cmd_scale(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .eval.scale import run_scale_curve
    _banner("STAGE 7  data-scale curve")
    data = _spider(cfg)
    out = Path(cfg.paths.eval_dir) / "scale"
    fractions = [float(x) for x in (args.fractions or "0,0.25,0.5,1.0").split(",")]
    _step(f"fractions={fractions}")
    result = run_scale_curve(cfg, data, fractions, out, quick=args.smoke)
    for p in result.get("points", []):
        _step(f"  frac={p.get('fraction', 0):.2f}  n={p.get('n_sft_records')}  "
              f"tokens={p.get('synth_tokens', 0):,}  exec_acc={p.get('exec_acc', 0):.3f}")
    return result


# --------------------------------------------------------------------------- #
# stage: report
# --------------------------------------------------------------------------- #
def cmd_report(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .report import build_report
    _banner("STAGE 8  report (tables + figures)")
    ablation = read_json(Path(cfg.paths.eval_dir) / "ablation" / "ablation.json", {})
    scale = read_json(Path(cfg.paths.eval_dir) / "scale" / "scale_curve.json", {})
    ledger_summary = {}
    for name in ("synth_ledger.jsonl", "validate_ledger.jsonl"):
        p = Path(cfg.paths.artifacts_dir) / name
        if p.exists():
            ledger_summary[name] = summarize_ledger(p)
    out = Path(cfg.paths.reports_dir)
    result = build_report(cfg, ablation, scale, ledger_summary, out)
    _step(f"wrote report artifacts -> {out}")
    return result


# --------------------------------------------------------------------------- #
# stage: extrapolate
# --------------------------------------------------------------------------- #
def cmd_extrapolate(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from .extrapolate import (build_extrapolation_report, markdown_table,
                              measure_from_ledger)
    _banner("STAGE 9  token-investment extrapolation to Spider scale")
    ledger_summary: dict[str, Any] = {}
    stages: dict[str, Any] = {}
    for name, stage_key in (("synth_ledger.jsonl", "synth"),
                            ("validate_ledger.jsonl", "validate")):
        p = Path(cfg.paths.artifacts_dir) / name
        if p.exists():
            s = summarize_ledger(p)
            stages.update(s.get("by_stage", {}))
    ledger_summary = {"by_stage": stages}
    train_p = Path(cfg.paths.raw_dir) / "train.jsonl"
    synth_p = cfg.paths.synth_dir / "synth.jsonl"
    n_source = sum(1 for _ in read_jsonl(train_p)) if train_p.exists() else 0
    n_synth = sum(1 for _ in read_jsonl(synth_p)) if synth_p.exists() else 0
    measured = measure_from_ledger(ledger_summary, n_source, n_synth)
    report = build_extrapolation_report(measured, out_dir=Path(cfg.paths.reports_dir))
    print(markdown_table(report))
    _step(f"measured: {measured.total_tokens:,} tokens on {n_source} sources "
          f"/ {n_synth} records")
    return report


# --------------------------------------------------------------------------- #
# stage: all
# --------------------------------------------------------------------------- #
def cmd_all(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs(cfg)
    t0 = time.time()
    out: dict[str, Any] = {}
    out["synth"] = cmd_synth(cfg, args)
    out["validate"] = cmd_validate(cfg, args)
    out["format"] = cmd_format(cfg, args)
    out["train"] = cmd_train(cfg, args)
    out["eval"] = cmd_eval(cfg, args)
    out["ablation"] = cmd_ablation(cfg, args)
    out["scale"] = cmd_scale(cfg, args)
    out["extrapolate"] = cmd_extrapolate(cfg, args)
    out["report"] = cmd_report(cfg, args)
    _banner(f"PIPELINE COMPLETE in {time.time() - t0:.1f}s")
    return out


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
COMMANDS = {
    "fixture": cmd_fixture,
    "synth": cmd_synth,
    "validate": cmd_validate,
    "format": cmd_format,
    "train": cmd_train,
    "eval": cmd_eval,
    "ablation": cmd_ablation,
    "scale": cmd_scale,
    "report": cmd_report,
    "extrapolate": cmd_extrapolate,
    "all": cmd_all,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fcdata", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, repeatable")
    ap.add_argument("--run-name", default="", help="run directory name")
    ap.add_argument("--smoke", action="store_true", help="fast reduced run")
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    ap.add_argument("--fractions", default="")
    ap.add_argument("--root", default=None, help="project root (default: cwd)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root) if args.root else Path.cwd()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        alt = root / args.config
        if alt.exists():
            cfg_path = alt
    cfg = load_config(cfg_path if cfg_path.exists() else None,
                      args.overrides, root=root)
    ensure_dirs(cfg)
    dump_config(cfg, Path(cfg.paths.artifacts_dir) / "resolved_config.json")
    try:
        COMMANDS[args.command](cfg, args)
    except SystemExit:
        raise
    except Exception:
        print("\n!! stage failed:", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
