"""Print the exact token-ledger totals used in the README."""
import json
from pathlib import Path

from fcdata.tokens import summarize_ledger

grand = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "total_tokens": 0, "cost_usd": 0.0}

for name in ("synth_ledger.jsonl", "validate_ledger.jsonl"):
    p = Path("artifacts") / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    s = summarize_ledger(p)
    t = s["totals"]
    print(f"{name}")
    print(f"   calls={t['calls']:,}  prompt={t['prompt_tokens']:,}  "
          f"completion={t['completion_tokens']:,}  total={t['total_tokens']:,}  "
          f"cost=${t['cost_usd']:.4f}")
    for stage, stg in s["by_stage"].items():
        print(f"   stage '{stage}': {stg['total_tokens']:,} tokens / {stg['calls']} calls")
        for role, r in stg["by_role"].items():
            print(f"       role '{role}' ({','.join(r['models'])}): "
                  f"{r['total_tokens']:,} tokens, ${r['cost_usd']:.4f}")
    for k in grand:
        grand[k] += t.get(k, 0)

print()
print("GRAND TOTAL")
print(f"   calls={grand['calls']:,}  prompt={grand['prompt_tokens']:,}  "
      f"completion={grand['completion_tokens']:,}  total={grand['total_tokens']:,}  "
      f"cost=${grand['cost_usd']:.4f}")
