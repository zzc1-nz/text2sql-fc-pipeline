import glob
import json

for d in ["raw_only", "plus_cot", "plus_hard_negatives",
          "frac_0p00", "frac_0p25", "frac_0p50", "frac_1p00"]:
    paths = (glob.glob(f"artifacts/06_eval/{d}/train/metrics.json")
             + glob.glob(f"artifacts/06_eval/{d}/train/*/metrics.json"))
    if not paths:
        print(f"{d:22s} NO metrics.json")
        continue
    for p in paths:
        m = json.load(open(p))
        print(f"{d:22s} steps={str(m.get('steps')):>5} "
              f"loss={m.get('final_train_loss'):.3f} "
              f"dev_tokacc={m.get('final_dev_token_acc')} "
              f"wall={m.get('wall_time_s')}s")
