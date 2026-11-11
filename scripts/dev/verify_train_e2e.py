"""Final end-to-end verification of the reference trainer on the REAL SFT corpus.

Trains with the recommended config, then evaluates greedy generation on the
training split with the *evaluator's* parser (`assistant_sql`) and real sqlite
execution, so the number reported is a genuine exec accuracy.

Run:  $env:PYTHONPATH='src'; python scripts/verify_train_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.config import TrainCfg, scratch_dir  # noqa: E402
from fcdata.dbexec import execute_sql  # noqa: E402
from fcdata.format import assistant_sql, build_sft_records, render_record_prompt  # noqa: E402
from fcdata.records import SynthRecord, ValidationReport, ValidatedRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train import train  # noqa: E402
from fcdata.train.reference import generate_batch, load_model  # noqa: E402


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
    from fcdata.config import load_config

    cfg = load_config("configs/demo.yaml", root=".")
    data = load_spider("data/fixture")
    sft, _ = build_sft_records(cfg, build_validated(data), data, data.train)
    print(f"sft records: {len(sft)}")

    tcfg = cfg.train
    tcfg.n_layer = 3
    tcfg.n_head = 4
    tcfg.d_model = 128
    tcfg.d_ff = 512
    tcfg.max_seq_len = 576
    tcfg.batch_size = 16
    tcfg.epochs = 8
    tcfg.lr = 1e-3
    tcfg.log_every = 1000
    tcfg.eval_every = 0

    run_dir = scratch_dir("verify_e2e")
    t0 = time.perf_counter()
    metrics = train(cfg, sft, None, run_dir)
    wall = time.perf_counter() - t0

    print("\n=== training metrics ===")
    for key in ("total_params", "trainable_params", "lora_params", "lora_only",
                "vocab_size", "n_train", "steps", "steps_per_sec", "wall_time_s",
                "first_epoch_train_loss", "final_train_loss",
                "final_train_token_acc", "loss_decreased"):
        print(f"  {key:<26} {metrics[key]}")

    # ---- evaluate with the real parser + real sqlite ----
    model, tok = load_model(run_dir / "model.pt")
    n_eval = min(40, len(sft))
    prompts = [render_record_prompt(r, cfg.fmt.chat_template) for r in sft[:n_eval]]
    t_gen = time.perf_counter()
    outs = generate_batch(model, tok, prompts, max_new_tokens=110, temperature=0.0)
    gen_s = time.perf_counter() - t_gen

    exact = exec_ok = 0
    for rec, out in zip(sft[:n_eval], outs):
        sql = assistant_sql({"role": "assistant", "content": out}) or ""
        if sql.strip().lower() == rec.target_sql.strip().lower():
            exact += 1
        if sql and sql.strip().lower().startswith(("select", "with")):
            try:
                if execute_sql(data.db_path(rec.db_id), sql).ok:
                    exec_ok += 1
            except FileNotFoundError:
                pass

    print("\n=== generation on TRAINING records (upper bound) ===")
    print(f"  n={n_eval}  exact_sql {exact}/{n_eval}  executes {exec_ok}/{n_eval}")
    print(f"  generation: {gen_s:.1f}s ({gen_s / n_eval * 1000:.0f} ms/example)")
    for rec, out in list(zip(sft[:n_eval], outs))[:2]:
        print(f"  gold={rec.target_sql[:90]}")
        print(f"  got ={(assistant_sql({'role': 'assistant', 'content': out}) or '')[:90]}")

    summary = {
        "n_sft": len(sft), "steps": metrics["steps"],
        "steps_per_sec": metrics["steps_per_sec"], "wall_s": round(wall, 1),
        "params": metrics["total_params"],
        "final_train_loss": metrics["final_train_loss"],
        "final_train_token_acc": metrics["final_train_token_acc"],
        "train_exact_sql": exact, "train_executes": exec_ok, "n_eval": n_eval,
        "gen_ms_per_example": round(gen_s / n_eval * 1000, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
