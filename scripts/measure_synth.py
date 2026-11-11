"""End-to-end synthesis measurement on a 50-example sample.

Reports the real token counts, per-task splits, judge calibration and determinism
that go into the task report. Run with ``$env:PYTHONPATH='src'; python
scripts/measure_synth.py``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter

from fcdata.config import Config, ProviderCfg, load_config, scratch_dir
from fcdata.providers import Job, build_provider, AsyncBatchRunner
from fcdata.providers import prompts
from fcdata.spider import load_spider
from fcdata.sqlutils import inject_error, normalize_sql
from fcdata.synth import synthesize
from fcdata.synth.hard_negative import corrupt_candidate
from fcdata.synth.common import build_sources
from fcdata.tokens import TokenLedger

N = 50


def main() -> None:
    cfg = load_config("configs/demo.yaml", root=".")
    cfg.synth.max_source_examples = N
    data = load_spider("data/fixture")

    # ---------------- run 1 ----------------
    ledger = TokenLedger(None)
    records, summary = synthesize(cfg, data, ledger, verbose=True)

    # ---------------- run 2: determinism ----------------
    ledger2 = TokenLedger(None)
    records2, summary2 = synthesize(cfg, data, ledger2, verbose=False)
    identical = ([r.id for r in records] == [r.id for r in records2]
                 and [r.question for r in records] == [r.question for r in records2]
                 and [r.reasoning for r in records] == [r.reasoning for r in records2]
                 and [r.bad_sql for r in records] == [r.bad_sql for r in records2]
                 and [r.total_tokens for r in records] == [r.total_tokens for r in records2])

    def _stable(recs) -> str:
        """Serialized records minus the wall-clock provenance stamp."""
        return json.dumps([{k: v for k, v in r.as_dict().items() if k != "created_at"}
                           for r in recs], sort_keys=True)

    byte_identical = _stable(records) == _stable(records2)
    differing = sorted({k for a, b in zip(records, records2)
                        for k in a.as_dict() if a.as_dict()[k] != b.as_dict()[k]})

    print("\n=== RECORDS ===")
    print(f"source examples      : {N}")
    print(f"records produced     : {len(records)}  {summary['by_kind']}")

    print("\n=== TOKENS (real counts from fcdata.tokens) ===")
    print(f"{'kind':<16}{'planned':>8}{'ok':>6}{'prompt':>10}{'compl':>9}"
          f"{'total':>10}{'tok/rec':>9}")
    for k in summary["kinds"]:
        per = k["total_tokens"] / max(1, k["ok"])
        print(f"{k['kind']:<16}{k['planned']:>8}{k['ok']:>6}{k['prompt_tokens']:>10}"
              f"{k['completion_tokens']:>9}{k['total_tokens']:>10}{per:>9.0f}")
    tot = summary["total_tokens"]
    print(f"{'TOTAL':<16}{sum(k['planned'] for k in summary['kinds']):>8}"
          f"{len(records):>6}{summary['prompt_tokens']:>10}"
          f"{summary['completion_tokens']:>9}{tot:>10}"
          f"{tot / max(1, len(records)):>9.0f}")
    print(f"ledger agrees        : {ledger.spent('synth') == tot}")
    print(f"cost (demo prices)   : ${summary['ledger_spent_usd']:.6f}")
    print(f"tokens/record (all)  : {tot / max(1, len(records)):.0f}")

    # ---------------- hard-negative error kinds ----------------
    kinds = Counter(r.error_kind for r in records if r.kind == "hard_negative")
    print("\n=== HARD NEGATIVES ===")
    print(f"records              : {sum(kinds.values())}")
    print(f"error kinds          : {dict(kinds)}")
    plans = build_sources(cfg, data)
    db_paths = {}
    for p in plans:
        try:
            db_paths[p.example.id] = data.db_path(p.db_id)
        except FileNotFoundError:
            pass
    # how many kinds were attempted vs accepted (yield of the verification gate)
    from fcdata.synth.hard_negative import PREFERRED_KINDS
    attempted = accepted = 0
    for p in plans[:N]:
        for k in PREFERRED_KINDS:
            attempted += 1
            ok, _, _ = corrupt_candidate(p, k, db_paths.get(p.example.id))
            accepted += int(ok)
    print(f"kind attempts        : {attempted}, accepted: {accepted} "
          f"({100.0 * accepted / max(1, attempted):.1f}% yield)")
    identical_bad = [r for r in records
                     if r.kind == "hard_negative"
                     and normalize_sql(r.bad_sql) == normalize_sql(r.sql)]
    print(f"bad_sql == gold      : {len(identical_bad)} (must be 0)")

    # ---------------- judge calibration ----------------
    print("\n=== JUDGE CALIBRATION ===")
    from fcdata.providers.mock import MockProvider
    from fcdata.providers.mockwriters import verdict_for_score
    judge = MockProvider(name="mock", model="mock-judge-1", role="judge",
                         ledger=TokenLedger(None), stage="validate",
                         temperature=0.0, max_tokens=256)
    good_scores, bad_scores = [], []
    incoherent = 0
    verdicts = Counter()
    for p in plans[:N]:
        good = judge.complete(prompts.judge_messages(
            p.ddl, p.question, p.gold_sql, p.gold_sql, p.db_id), task="judge")
        parsed = prompts.parse_judge_json(good.text)
        if parsed:
            good_scores.append(parsed["score"])
            verdicts[parsed["verdict"]] += 1
            if parsed["verdict"] != verdict_for_score(parsed["score"]):
                incoherent += 1
        ok, bad, _ = corrupt_candidate(p, "wrong_table", db_paths.get(p.example.id))
        if ok:
            comp = judge.complete(prompts.judge_messages(
                p.ddl, p.question, bad, p.gold_sql, p.db_id), task="judge")
            parsed_bad = prompts.parse_judge_json(comp.text)
            if parsed_bad:
                bad_scores.append(parsed_bad["score"])
    print(f"gold queries         : mean={statistics.mean(good_scores):.3f} "
          f"min={min(good_scores):.3f} max={max(good_scores):.3f}")
    print(f"corrupted queries    : mean={statistics.mean(bad_scores):.3f} "
          f"min={min(bad_scores):.3f} max={max(bad_scores):.3f}")
    print(f"separation (gap)     : {statistics.mean(good_scores) - statistics.mean(bad_scores):.3f}")
    print(f"verdict distribution : {dict(verdicts)}")
    print(f"score/verdict incoherent: {incoherent} (must be 0)")
    print(f"gold >= 0.5 threshold pass rate: "
          f"{100.0 * sum(1 for s in good_scores if s >= 0.5) / len(good_scores):.1f}%")

    # ---------------- determinism ----------------
    print("\n=== DETERMINISM ===")
    print(f"ids/questions/reasoning/bad_sql/tokens identical : {identical}")
    print(f"records byte-identical ignoring created_at       : {byte_identical}")
    print(f"fields that differ between runs                  : {differing or 'none'}")

    # ---------------- per-task token split (for the README) ----------------
    print("\n=== TOKENS BY TASK (all calls in this run) ===")
    by_task: dict[str, list[int]] = {}
    for rec in ledger.records:
        by_task.setdefault(rec.task, []).append(rec.total_tokens)
    for task, toks in sorted(by_task.items()):
        print(f"{task:<16}{len(toks):>6} calls  mean={statistics.mean(toks):>7.0f}  "
              f"total={sum(toks):>8}")


if __name__ == "__main__":
    main()
