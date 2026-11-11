"""Benchmark the reference backend on realistic fixture data (dev tool, not a test).

Run:  $env:PYTHONPATH='src'; python scripts/bench_train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.config import TrainCfg, scratch_dir  # noqa: E402
from fcdata.records import SFTRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train import train  # noqa: E402
from fcdata.train.reference import count_parameters, describe_model  # noqa: E402


def build_records(data, n: int) -> list[SFTRecord]:
    """Realistic function-calling SFT records from the real fixture."""
    out: list[SFTRecord] = []
    examples = list(data.train)
    for i in range(n):
        ex = examples[i % len(examples)]
        db = data.databases[ex.db_id]
        ddl = db.render_ddl(max_tables=12)
        tools = [{"type": "function", "function": {
            "name": "execute_sql",
            "description": "Execute a read-only SQLite query against the database.",
            "parameters": {"type": "object", "properties": {
                "sql": {"type": "string", "description": "A single SQLite SELECT."}},
                "required": ["sql"]}}}]
        msgs = [
            {"role": "system", "content": "You are a SQL assistant. Convert the user's "
                                          "question into a single SQLite SELECT statement "
                                          "using the provided database schema."},
            {"role": "user", "content": f"Database schema:\n{ddl}\n\nQuestion: {ex.question}"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "execute_sql",
                             "arguments": json.dumps({"sql": ex.sql})}}]},
        ]
        out.append(SFTRecord(id=f"bench_{i}", source_id=ex.id, kind="raw",
                             db_id=ex.db_id, messages=msgs, tools=tools,
                             target_sql=ex.sql, difficulty=ex.difficulty))
    return out


def run(name: str, cfg: TrainCfg, records: list[SFTRecord], dev: list[SFTRecord]) -> dict:
    d = scratch_dir(f"bench_{name}")
    t0 = time.perf_counter()
    metrics = train(cfg, records, dev, d)
    wall = time.perf_counter() - t0
    return {
        "name": name,
        "wall_s": round(wall, 2),
        "params": metrics["total_params"],
        "trainable": metrics["trainable_params"],
        "vocab": metrics["vocab_size"],
        "steps": metrics["steps"],
        "steps_per_sec": metrics["steps_per_sec"],
        "tokens_per_sec": metrics["tokens_per_sec"],
        "final_train_loss": metrics["final_train_loss"],
        "first_epoch_loss": metrics["first_epoch_train_loss"],
        "final_dev_token_acc": metrics["final_dev_token_acc"],
        "loss_decreased": metrics["loss_decreased"],
        "avg_sup_tokens": round(
            sum(h["step"] for h in metrics["history"]) / max(1, len(metrics["history"])), 2),
    }


def main() -> int:
    data = load_spider("data/fixture")
    print(f"fixture: {len(data.train)} train / {len(data.dev)} dev examples, "
          f"{len(data.databases)} DBs")
    n_records = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    records = build_records(data, n_records)
    dev = build_records(data, 24)
    print(f"records: {len(records)} train, {len(dev)} dev")
    print(f"avg message chars: "
          f"{sum(len(json.dumps(r.messages)) for r in records) / len(records):.0f}")

    configs = [
        ("A_n4_d128_v2048", dict(n_layer=4, n_head=4, d_model=128, d_ff=512,
                                 vocab_size=2048, max_seq_len=384, lora_r=8,
                                 lora_alpha=16, batch_size=16, epochs=2, lr=3e-4,
                                 log_every=100000)),
        ("B_n6_d256_v4096", dict(n_layer=6, n_head=8, d_model=256, d_ff=1024,
                                 vocab_size=4096, max_seq_len=384, lora_r=8,
                                 lora_alpha=16, batch_size=16, epochs=2, lr=3e-4,
                                 log_every=100000)),
        ("C_n4_d192_v2048", dict(n_layer=4, n_head=6, d_model=192, d_ff=768,
                                 vocab_size=2048, max_seq_len=384, lora_r=8,
                                 lora_alpha=16, batch_size=16, epochs=2, lr=3e-4,
                                 log_every=100000)),
    ]
    results = []
    for name, kw in configs:
        cfg = TrainCfg(**kw)
        r = run(name, cfg, records, dev)
        results.append(r)
        print(json.dumps(r, indent=2))
    print("\n=== SUMMARY ===")
    print(f"{'config':<18}{'params':>10}{'wall_s':>9}{'steps/s':>9}"
          f"{'tok/s':>9}{'loss0':>9}{'lossN':>9}{'dev_acc':>9}")
    for r in results:
        print(f"{r['name']:<18}{r['params']:>10}{r['wall_s']:>9}{r['steps_per_sec']:>9}"
              f"{r['tokens_per_sec']:>9}{r['first_epoch_loss']:>9}"
              f"{r['final_train_loss']:>9}{str(r['final_dev_token_acc']):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
