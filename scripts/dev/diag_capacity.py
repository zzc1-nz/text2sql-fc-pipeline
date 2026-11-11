"""Diagnostic: can the reference model learn text2sql at all?

Trains on a small slice and evaluates on the *same* slice (train-set accuracy).
If exec accuracy stays at 0 even on data the model has seen, the problem is
capacity/representation, not generalisation. If it rises, the pipeline is sound
and the demo simply needs more steps.

    $env:PYTHONPATH='src'; python scripts/diag_capacity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.config import load_config
from fcdata.records import read_sft
from fcdata.spider import load_spider
from fcdata.train import train
from fcdata.train.reference import generate, load_model
from fcdata.train import data as tdata
from fcdata.dbexec import evaluate_pair
from fcdata.sqlutils import extract_sql


def eval_on(run_dir: Path, records, data, tag: str) -> dict:
    model, tok = load_model(run_dir)
    n_ok = n_exec = n_match = 0
    samples = []
    for r in records:
        prompt = tdata.render_prompt_text(r, template="qwen")
        out = generate(model, tok, prompt, max_new_tokens=120)
        sql = extract_sql(out)
        gold = r.target_sql
        db_path = data.db_path(r.db_id)
        res = evaluate_pair(db_path, sql, gold) if sql else {"executable": False, "exec_match": False}
        n_ok += 1
        n_exec += int(res.get("executable", False))
        n_match += int(res.get("exec_match", False))
        if len(samples) < 3:
            samples.append({"gold": gold[:100], "pred": sql[:100], "raw": out[:120]})
    out = {"tag": tag, "n": n_ok, "executable": n_exec, "exec_match": n_match,
           "exec_acc": round(n_match / max(1, n_ok), 3), "samples": samples}
    print(f"[{tag}] n={n_ok} executable={n_exec} exec_match={n_match} "
          f"acc={out['exec_acc']:.1%}")
    for s in samples:
        print(f"   gold: {s['gold']}")
        print(f"   pred: {s['pred']!r}")
        print(f"   raw : {s['raw']!r}")
    return out


def main() -> int:
    cfg = load_config("configs/demo.yaml")
    data = load_spider(cfg.paths.fixture_dir)
    sft = read_sft(cfg.paths.sft_dir / "train.jsonl")

    # a tiny slice the model should be able to memorise
    slice_n = 40
    sub = sft[:slice_n]
    cfg.train.epochs = 12
    cfg.train.lr = 1e-3
    run = Path("artifacts/diag/capacity")
    metrics = train(cfg, sub, None, run)
    print(f"train: {metrics.get('steps')} steps, "
          f"final loss {metrics.get('final_train_loss'):.4f}, "
          f"tok_acc {metrics.get('final_train_token_acc'):.3f}")

    result = eval_on(run, sub, data, "train_slice_memorised")
    Path("artifacts/diag").mkdir(parents=True, exist_ok=True)
    Path("artifacts/diag/capacity_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["exec_match"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
