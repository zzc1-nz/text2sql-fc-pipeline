"""Establish ground truth directly from each arm's own eval/train metrics files.

Reads `artifacts/06_eval/ablation/<arm>/{eval,train}/metrics.json` -- the files a
real run wrote -- rather than any aggregated report, so nothing is inherited from
a possibly-stale or concurrently-written summary.
"""
import json
from pathlib import Path

ARMS = ["raw_only", "plus_cot", "plus_hard_negatives"]
ROOT = Path("artifacts/06_eval/ablation")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    print("=" * 110)
    print("GROUND TRUTH per arm (from <arm>/eval/metrics.json and <arm>/train/metrics.json)")
    print("=" * 110)
    hdr = (f"{'arm':22s} {'steps':>6s} {'tr_loss':>8s} {'dev_tok':>8s} {'n':>4s} "
           f"{'exec':>6s} {'execble':>8s} {'tok_acc':>8s} {'skel':>6s} {'tbl':>6s} {'colF1':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        d = ROOT / arm
        tr = load(d / "train" / "metrics.json")
        ev = load(d / "eval" / "metrics.json")
        pa = ev.get("partial") or {}
        print(f"{arm:22s} {str(tr.get('steps')):>6} "
              f"{_f(tr.get('final_train_loss')):>8} "
              f"{_f(tr.get('final_dev_token_acc')):>8} "
              f"{str(ev.get('n')):>4} "
              f"{_f(ev.get('exec_acc')):>6} "
              f"{_f(ev.get('executable_rate')):>8} "
              f"{_f(pa.get('token_acc')):>8} "
              f"{_f(pa.get('skeleton_match')):>6} "
              f"{_f(pa.get('table_set_match')):>6} "
              f"{_f(pa.get('column_f1')):>7}")

    print()
    print("failure taxonomy (counts)")
    classes = ["wrong_column", "not_executable", "wrong_table", "wrong_values",
               "missing_join", "extra_join", "missing_group_by", "missing_order_by",
               "wrong_aggregate", "unparseable"]
    print(f"{'arm':22s} " + " ".join(f"{c[:11]:>11s}" for c in classes))
    for arm in ARMS:
        ev = load(ROOT / arm / "eval" / "metrics.json")
        pa = ev.get("partial") or {}
        ft = pa.get("failure_taxonomy") or ev.get("failure_taxonomy") or {}

        def count(c: str) -> int:
            v = ft.get(c, 0)
            if isinstance(v, dict):
                return int(v.get("n", 0))
            return int(v or 0)

        print(f"{arm:22s} " + " ".join(f"{count(c):>11d}" for c in classes))
    return 0


def _f(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
