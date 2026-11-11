"""Measure real SFT sequence lengths against max_seq_len (dev diagnostic).

Run:  $env:PYTHONPATH='src'; python scripts/check_seqlen.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.records import SFTRecord  # noqa: E402
from fcdata.spider import load_spider  # noqa: E402
from fcdata.train.data import (  # noqa: E402
    encode_record,
    render_prompt_text,
    render_target_text,
)
from fcdata.train.vocab import WordTokenizer  # noqa: E402

SYSTEM = ("You are a SQL assistant. Convert the user's natural-language question "
          "into a single SQLite SELECT statement using the provided database schema.")


def make_record(i: int, db_id: str, ddl: str, question: str, sql: str) -> SFTRecord:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Database schema:\n{ddl}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "execute_sql",
                         "arguments": json.dumps({"sql": sql})}}]},
    ]
    return SFTRecord(id=f"seq_{i}", source_id=f"ex_{i}", kind="raw", db_id=db_id,
                     messages=msgs, target_sql=sql)


def main() -> int:
    data = load_spider("data/fixture")
    for max_tables in (12, 8, 6):
        recs = [make_record(i, ex.db_id, data.databases[ex.db_id].render_ddl(max_tables),
                            ex.question, ex.sql)
                for i, ex in enumerate(data.train)]
        texts = [render_prompt_text(r) + render_target_text(r) for r in recs]
        tok = WordTokenizer().fit(texts, 2048)
        pl = [len(tok.encode(render_prompt_text(r), add_bos=True)) for r in recs]
        tl = [len(tok.encode(render_target_text(r), add_eos=True)) for r in recs]
        tot = [a + b for a, b in zip(pl, tl)]
        print(f"=== max_schema_tables={max_tables} (vocab={tok.vocab_size}) ===")
        for name, vals in (("prompt", pl), ("target", tl), ("total", tot)):
            p95 = sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.95))]
            print(f"  {name:7s} min={min(vals):4d} p50={int(st.median(vals)):4d} "
                  f"p95={p95:4d} max={max(vals):4d}")
        for cap in (384, 512):
            fits = sum(1 for t in tot if t <= cap)
            print(f"  fits<= {cap}: {fits}/{len(tot)}")
        # Would encode_record truncate the target at 512?
        truncated = 0
        for r, t_len in zip(recs, tl):
            enc = encode_record(r, tok, max_seq_len=512)
            if enc.target_len != t_len:
                truncated += 1
        print(f"  target truncated at max_seq_len=512: {truncated}/{len(recs)}")

    # tokenizer robustness on tool-call JSON
    print("\n=== tokenizer round-trip on tool-call JSON ===")
    samples = [
        '{"tool_calls":[{"type":"function","function":{"name":"execute_sql",'
        '"arguments":{"sql":"SELECT * FROM student WHERE gpa >= 3.5"}}}]}',
        "<|im_start|>assistant\n",
        "SELECT t1.name, COUNT(*) FROM student AS t1 JOIN dept AS t2 "
        "ON t1.dept_id = t2.id WHERE t1.gpa <> 2.0 GROUP BY t1.name LIMIT 5",
    ]
    ok = 0
    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        same = back == s.strip()
        ok += int(same)
        print(f"  exact={same} ntok={len(ids):3d} unk={ids.count(tok.unk_id)}  {back[:90]}")
    print(f"  round-trip exact: {ok}/{len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
