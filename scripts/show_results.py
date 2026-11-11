"""Print the authoritative ablation + scale results from eval-engineer's run.

Reads `artifacts/06_eval/ablation/ablation.json` (the run that includes the
step-parity and nested-partial fixes) and the scale curve beside it, falling back
to the flat `artifacts/06_eval/` location if needed.
"""
import json
from pathlib import Path

CANDIDATES = [
    Path("artifacts/06_eval/ablation"),
    Path("artifacts/06_eval"),
]


def find(name: str) -> Path | None:
    for d in CANDIDATES:
        p = d / name
        if p.exists():
            return p
    return None


def main() -> int:
    p = find("ablation.json")
    if p is None:
        print("no ablation.json found")
        return 1
    a = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 104)
    print(f"ABLATION  ({p})")
    print("=" * 104)
    print(f"split={a.get('eval_split')}  n_eval={a.get('n_eval_examples')}  "
          f"n_sft_available={a.get('n_sft_available')}  quick={a.get('quick')}")
    if a.get("warnings"):
        print("warnings:", a["warnings"])
    print()
    hdr = (f"{'config':22s} {'n_sft':>6s} {'synth_tok':>10s} {'exec':>7s} {'EM':>7s} "
           f"{'execble':>8s} {'tok_acc':>8s} {'skel':>6s} {'tbl':>6s} {'colF1':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in a.get("rows", []):
        print(f"{r['config']:22s} {r['n_sft_records']:>6d} {r['synth_tokens']:>10,} "
              f"{r['exec_acc']:>7.3f} {r['exact_match']:>7.3f} {r['executable_rate']:>8.3f} "
              f"{r.get('token_acc', 0):>8.3f} {r.get('skeleton_match', 0):>6.3f} "
              f"{r.get('table_set_match', 0):>6.3f} {r.get('column_f1', 0):>7.3f}")
    print()
    print("failure taxonomy (error class x config, counts)")
    classes = ["wrong_column", "not_executable", "wrong_table", "wrong_values",
               "missing_join", "extra_join", "missing_group_by", "missing_order_by",
               "wrong_aggregate", "unparseable"]
    print(f"{'config':22s} " + " ".join(f"{c[:11]:>11s}" for c in classes))
    for r in a.get("rows", []):
        ft = (r.get("partial_failures")
              or (r.get("partial") or {}).get("failure_taxonomy")
              or {})

        def count(c: str) -> int:
            v = ft.get(c, 0)
            if isinstance(v, dict):
                return int(v.get("n", 0))
            return int(v or 0)

        print(f"{r['config']:22s} " + " ".join(f"{count(c):>11d}" for c in classes))

    sp = find("scale_curve.json")
    if sp is not None:
        s = json.loads(sp.read_text(encoding="utf-8"))
        print()
        print("=" * 104)
        print(f"DATA-SCALE CURVE  ({sp})")
        print("=" * 104)
        hdr2 = (f"{'frac':>6s} {'n_sft':>6s} {'synth_tok':>10s} {'exec':>7s} {'EM':>7s} "
                f"{'execble':>8s} {'tok_acc':>8s} {'tbl':>6s}")
        print(hdr2)
        print("-" * len(hdr2))
        for pt in s.get("points", []):
            part = pt.get("partial") or pt
            print(f"{pt.get('fraction', 0):>6.2f} {pt.get('n_sft_records', 0):>6d} "
                  f"{pt.get('synth_tokens', 0):>10,} {pt.get('exec_acc', 0):>7.3f} "
                  f"{pt.get('exact_match', 0):>7.3f} {pt.get('executable_rate', 0):>8.3f} "
                  f"{part.get('token_acc', 0):>8.3f} {part.get('table_set_match', 0):>6.3f}")
    else:
        print("\n(scale curve not yet available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
