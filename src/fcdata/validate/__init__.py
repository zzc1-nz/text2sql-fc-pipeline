"""Validation stage -- the quality gate between synthesis and the SFT corpus.

WHY this package exists
-----------------------
Synthesis turns ~100 gold examples into thousands of augmented records; the
student model then imitates whatever is in that corpus.  A single bad target is
therefore not a wasted sample, it is *actively harmful*: it teaches a confidently
wrong behaviour that evaluation will later reward-agnostically reproduce.

So every synthesized record passes two independent gates before it can become
training data:

``format`` (:mod:`fcdata.validate.format_check`)
    Does the SQL parse, resolve against the Spider schema, and really execute on
    the fixture sqlite?  Cheap, deterministic, run on *every* record.  For hard
    negatives the premise is checked too: the wrong SQL must differ from gold and
    must actually be wrong.

``semantic`` (:mod:`fcdata.validate.semantic`)
    Does the SQL answer the *question*?  Expensive (an LLM judge), so run on a
    deterministic sample controlled by ``cfg.validate.semantic_sample``.

``passed`` is the conjunction, with one deliberate exception: a *judge
infrastructure* failure (provider error, unparseable reply) does not fail the
record -- the record is format-valid and a transient API hiccup must not silently
delete training data.  Those are reported separately as ``judge.errors`` so the
report can never hide them.

``build_validation_report`` writes ``validation_report.json``: the funnel
(synth -> format-pass -> semantic-checked -> kept), per-kind and per-error-kind
pass rates, judge token spend, and the failure-code histogram.  The report is the
input to the "token investment -> capability" table, so it must account for
tokens exactly once (the provider owns the ledger; we only mirror usage).
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ..records import (SynthRecord, ValidatedRecord, ValidationReport, write_json,
                       write_jsonl)
from ..spider import SpiderData

from .format_check import (BLOCKING_CODES, HN_SYMPTOM_CODES, SCHEMA_CODES,
                           SYNTAX_CODES, WARNING_CODES, check_format, check_records,
                           format_ok, normalize_question)
from .semantic import (SEMANTIC_CODES, VERDICTS, SemanticVerdict, apply_verdict,
                       build_judge_messages, judge_cost_usd, judge_one,
                       parse_judge_json, provider_judge_messages,
                       render_relevant_ddl, run_semantic, select_semantic_sample)

#: issue codes that record *success* rather than a problem
_SUCCESS_CODES: frozenset[str] = frozenset({"semantic_consistent"})

__all__ = [
    # submodules
    "format_check", "semantic",
    # format stage
    "check_format", "check_records", "format_ok", "normalize_question",
    "BLOCKING_CODES", "SYNTAX_CODES", "SCHEMA_CODES", "WARNING_CODES", "HN_SYMPTOM_CODES",
    # semantic stage
    "SemanticVerdict", "judge_one", "parse_judge_json", "apply_verdict",
    "build_judge_messages", "provider_judge_messages", "render_relevant_ddl",
    "select_semantic_sample", "run_semantic", "VERDICTS", "SEMANTIC_CODES",
    # orchestration
    "validate_records", "build_validation_report", "validate_and_write", "combined_pass",
]

# --------------------------------------------------------------------------- #
# provider resolution
# --------------------------------------------------------------------------- #
def _resolve_provider(cfg: Any, ledger: Any, provider: Any) -> Any:
    """Lazily build the judge provider.

    Imported inside the function on purpose: ``fcdata.providers`` is owned by a
    different stage and may not exist yet (or may pull in httpx), while the format
    stage must stay importable offline.  A missing module becomes an actionable
    error instead of an import-time crash for the whole package.
    """
    if provider is not None:
        return provider
    try:
        from ..providers import build_provider  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - depends on build order
        raise RuntimeError(
            "fcdata.validate.semantic needs a provider: fcdata.providers is not "
            "importable yet. Pass provider=<object with .complete(...)> explicitly "
            "or wait for the providers stage to land."
        ) from exc
    return build_provider(cfg.validate.judge_role, cfg, ledger)


# --------------------------------------------------------------------------- #
# the combined verdict
# --------------------------------------------------------------------------- #
def combined_pass(report: ValidationReport, cfg: Any, *, judge_failed: bool = False) -> bool:
    """Format must pass; the judge must pass *unless it could not run*.

    ``judge_failed`` means "we tried to judge and the judge itself broke" (provider
    error / unparseable JSON), which is an infrastructure problem, not a data
    problem, so it never fails a format-valid record.
    """
    if not format_ok(report):
        return False
    if not report.semantic_checked:
        return True
    if judge_failed:
        return True
    threshold = float(getattr(cfg.validate, "semantic_threshold", 0.5))
    return report.semantic_score >= threshold


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def validate_records(cfg: Any,
                     records: Sequence[SynthRecord],
                     spider_data: SpiderData,
                     ledger: Any = None,
                     *,
                     provider: Any = None,
                     record_usage_in_ledger: bool = False,
                     timeout_s: float | None = None) -> list[ValidatedRecord]:
    """Run both validation gates over ``records``.

    Parameters
    ----------
    cfg:
        Pipeline config; ``cfg.validate`` drives the gates.
    records:
        Output of the synthesis stage (``SynthRecord`` list).
    spider_data:
        Provides ``.databases`` (schema) and ``.db_path(db_id)`` (real sqlite).
    ledger:
        Token ledger. The judge provider records into it (one ``UsageRecord`` per
        call, per the provider contract). Set ``record_usage_in_ledger=True`` only
        for a provider that does *not* record, to avoid double counting.
    provider:
        Optional explicit judge provider (tests / dry runs).

    Returns
    -------
    list[ValidatedRecord]
        **One entry per input record** -- failures are kept with ``passed=False``
        so the funnel (synth -> kept) stays auditable.  ``validate_and_write``
        applies ``cfg.validate.drop_failed`` when writing the artifact.
    """
    vcfg = cfg.validate
    timeout = float(timeout_s if timeout_s is not None else
                    getattr(cfg.ev, "sql_timeout_s", 5.0))
    databases = spider_data.databases

    def _db_path(db_id: str) -> Path | None:
        try:
            return spider_data.db_path(db_id)
        except FileNotFoundError:
            return None

    # ---- gate 1: format (all records, cheap, deterministic) -----------
    reports: list[ValidationReport] = []
    for rec in records:
        db = databases.get(rec.db_id)
        if db is None:
            rep = ValidationReport()
            rep.issue_codes.append("unknown_db_id")
            rep.reason = "unknown_db_id"
            reports.append(rep)
            continue
        path = _db_path(rec.db_id)
        if path is None:
            rep = ValidationReport()
            rep.issue_codes.append("missing_database")
            rep.reason = "missing_database"
            reports.append(rep)
            continue
        reports.append(check_format(rec, db, path, timeout_s=timeout))

    # ---- gate 2: semantic judge (deterministic sample of format-pass) --
    verdicts: dict[str, SemanticVerdict] = {}
    if getattr(vcfg, "enable_semantic", True):
        candidates = [rec for rec, rep in zip(records, reports) if format_ok(rep)]
        if candidates:
            judge = _resolve_provider(cfg, ledger, provider)
            verdicts = run_semantic(candidates, cfg, judge, databases)

    # ---- combine ------------------------------------------------------
    out: list[ValidatedRecord] = []
    for rec, rep in zip(records, reports):
        verdict = verdicts.get(rec.id)
        if verdict is not None:
            apply_verdict(rep, verdict)
            if record_usage_in_ledger and ledger is not None:
                ledger.add(
                    "validate", vcfg.judge_role,
                    cfg.provider(vcfg.judge_role).model,
                    verdict.prompt_tokens, verdict.completion_tokens,
                    task="judge", ok=verdict.ok, cached=False,
                    sample_id=rec.id)
        rep.passed = combined_pass(rep, cfg, judge_failed=bool(
            verdict is not None and not verdict.ok))
        if not rep.passed and not rep.reason:
            rep.reason = "failed"
        out.append(ValidatedRecord(record=rec, report=rep))
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _funnel(n_raw: int, n_format: int, n_semantic_checked: int, n_semantic_pass: int,
            n_kept: int, n_raw_examples: int) -> list[dict[str, Any]]:
    """The README funnel: raw -> synth -> format-pass -> semantic-pass -> SFT.

    ``pct`` is relative to the *previous* row, so each row answers "what survived
    this gate".  One row is deliberately not a survival rate: ``raw -> synth`` is
    an **amplification** (one Spider example yields several augmented records), so
    ``pct`` there legitimately exceeds 100% and the chain is not monotonic until
    ``format_pass``.  ``pct_of_raw`` anchors every row to the seed corpus so both
    the amplification and the attrition are readable from the same table.

    Rows are omitted rather than faked when the count is unknown
    (``n_raw_examples == 0`` means the caller did not tell us the seed size).
    """
    rows: list[dict[str, Any]] = []
    prev = n_raw_examples
    if n_raw_examples:
        rows.append({"stage": "raw", "n": n_raw_examples, "pct": 100.0, "pct_of_raw": 100.0})
    stages = [("synth", n_raw), ("format_pass", n_format),
              ("semantic_checked", n_semantic_checked),
              ("semantic_pass", n_semantic_pass), ("sft", n_kept)]
    for name, value in stages:
        denom = prev if prev else value
        rows.append({
            "stage": name,
            "n": value,
            "pct": round(_rate(value, denom) * 100, 2) if denom else 0.0,
            "pct_of_raw": round(_rate(value, n_raw_examples) * 100, 2) if n_raw_examples else 0.0,
            # explicit flag so a report generator does not have to guess which row
            # is allowed to exceed 100%
            "is_amplification": name == "synth",
        })
        prev = value
    return rows


def _ground_truth_flags(validated: Sequence[ValidatedRecord],
                        spider_data: Any | None) -> dict[str, bool] | None:
    """Per-record ground truth: is this record's SQL provably correct?

    ``SynthRecord.sql`` is by contract the **gold** SQL of a real Spider example,
    so a record is *provably correct* when its SQL is byte-identical to its source
    example's gold and executes to the same result set.  That check is independent
    of the judge and of any threshold, which is exactly what makes it usable as an
    audit reference.

    Computed for **every** judged record (not only the ones the configured
    threshold happened to drop): the threshold sweep has to be able to re-score
    alternative thresholds without re-running the judge, so the flags cannot be
    filtered by the current decision.

    Returns ``None`` when no independent ground truth is available (no
    ``spider_data``, or no source examples), never a guessed default.
    """
    if spider_data is None:
        return None
    sources = {e.id: e for e in (list(getattr(spider_data, "train", []) or [])
                                 + list(getattr(spider_data, "dev", []) or []))}
    if not sources:
        return None

    from ..dbexec import evaluate_pair

    flags: dict[str, bool] = {}
    for v in validated:
        if not v.report.semantic_checked:
            continue
        src = sources.get(v.record.source_id)
        if src is None:
            continue
        if v.record.sql.strip() != src.sql.strip():
            flags[v.id] = False
            continue
        try:
            flags[v.id] = bool(evaluate_pair(spider_data.db_path(src.db_id),
                                             v.record.sql, src.sql)["exec_match"])
        except (FileNotFoundError, KeyError):
            flags[v.id] = False
    return flags


def _threshold_sweep(validated: Sequence[ValidatedRecord],
                     flags: dict[str, bool] | None,
                     threshold: float,
                     thresholds: Sequence[float] = (0.30, 0.40, 0.50, 0.60)) -> dict[str, Any]:
    """What each candidate ``semantic_threshold`` would keep, with precision.

    WHY it ships inside the report: the threshold is a *policy* decision, and a
    policy is only auditable if the data behind it travels with the artifact.  A
    reader must be able to see that moving the bar from 0.50 to 0.40 keeps 83 more
    records without admitting a single wrong one -- otherwise the number in the
    config looks arbitrary and a future maintainer will re-guess it.

    Definitions (standard, and the ones the README quotes):

    * ``kept``             records scoring ``>= threshold``
    * ``kept_correct``     kept records that are *provably correct*
    * ``precision``        ``kept_correct / kept`` -- of what we kept, how much is right
    * ``false_negatives``  correct records the threshold threw away
    * ``recall``           ``kept_correct / total_correct``

    When ground truth is unavailable, the correctness columns are ``None`` (an
    unknown is not a measurement) and only the kept counts are reported.
    """
    judged = [v for v in validated if v.report.semantic_checked]
    total = len(judged)
    if not total:
        return {"available": False, "reason": "no records were semantically judged"}

    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        kept_recs = [v for v in judged if v.report.semantic_score >= thr]
        row: dict[str, Any] = {
            "threshold": float(thr),
            "kept": len(kept_recs),
            "keep_rate": _rate(len(kept_recs), total),
            "dropped": total - len(kept_recs),
            "is_configured": abs(float(thr) - float(threshold)) < 1e-9,
        }
        if flags is None:
            row.update({"kept_correct": None, "precision": None,
                        "false_negatives": None, "false_negative_rate": None,
                        "recall": None})
        else:
            total_correct = sum(1 for v in judged if flags.get(v.id))
            kept_correct = sum(1 for v in kept_recs if flags.get(v.id))
            fn = total_correct - kept_correct
            row.update({
                "kept_correct": kept_correct,
                # precision is undefined when nothing is kept -- reporting 0.0 would
                # look like "everything kept was wrong" rather than "nothing kept"
                "precision": _rate(kept_correct, len(kept_recs)) if kept_recs else None,
                "false_negatives": fn,
                "false_negative_rate": _rate(fn, total_correct),
                "recall": _rate(kept_correct, total_correct),
            })
        rows.append(row)

    configured = next((r for r in rows if r["is_configured"]), None)
    return {
        "available": True,
        "n_judged": total,
        "configured_threshold": float(threshold),
        "configured": configured,
        "rows": rows,
        "precision_defined": flags is not None,
        "note": ("precision = share of kept records proven correct against execution "
                 "ground truth. When it is 1.0 at every threshold, the gate is not "
                 "trading precision for recall: the threshold only decides how much "
                 "verified-correct data is discarded."),
    }


def _ground_truth_audit(validated: Sequence[ValidatedRecord],
                        spider_data: Any | None,
                        threshold: float) -> dict[str, Any]:
    """Measure how many semantic drops are *provably correct* SQL.

    WHY this exists: a pass rate alone cannot distinguish "the judge filtered bad
    data" from "the judge deleted good data".  The pipeline has independent ground
    truth -- ``SynthRecord.sql`` is by contract the **gold** SQL of a real Spider
    example -- so a dropped record whose SQL is byte-identical to its source gold
    and which executes to the same result set is a judge **false negative** by
    definition, not filtering.

    Reporting that number keeps the funnel honest in both directions: a high drop
    rate is only defensible if most drops are genuinely wrong, and when they are
    not, the report says so instead of letting the percentage imply rigour.

    Delegates the per-record judgement to :func:`_ground_truth_flags` so the audit
    and the threshold sweep can never disagree about which records are correct.
    """
    flags = _ground_truth_flags(validated, spider_data)
    if flags is None:
        return {"available": False,
                "reason": "no independent ground truth (spider_data/source examples)"}

    dropped = [v for v in validated if not v.report.passed and v.report.semantic_checked]
    total_correct = sum(1 for v in validated if flags.get(v.id))
    kept = [v for v in validated if v.report.passed]
    kept_correct = sum(1 for v in kept if flags.get(v.id))
    precision_of_kept = _rate(kept_correct, len(kept)) if kept else None
    if not dropped:
        return {"available": True, "n_dropped": 0, "n_correct": total_correct,
                "n_false_negative": 0, "n_false_negative_rate": 0.0,
                "precision_of_kept": precision_of_kept,
                "by_kind": {}, "examples": []}

    by_kind: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    n_fn = 0
    for v in dropped:
        k = by_kind.setdefault(v.kind, {"dropped": 0, "provably_correct": 0})
        k["dropped"] += 1
        if flags.get(v.id):
            k["provably_correct"] += 1
            n_fn += 1
            if len(examples) < 10:
                examples.append({"id": v.id, "kind": v.kind,
                                 "score": v.report.semantic_score,
                                 "verdict": v.report.semantic_verdict,
                                 "question": v.record.question[:120],
                                 "rationale": v.report.semantic_rationale[:160]})

    kept = [v for v in validated if v.report.passed]
    kept_correct = sum(1 for v in kept if flags.get(v.id))
    return {
        "available": True,
        "n_dropped": len(dropped),
        "n_correct": total_correct,
        "n_false_negative": n_fn,
        "n_false_negative_rate": _rate(n_fn, total_correct),
        # kept records that are correct / kept records: the precision of the gate
        "precision_of_kept": _rate(kept_correct, len(kept)) if kept else None,
        "by_kind": by_kind,
        "examples": examples,
        "note": ("A drop whose SQL is byte-identical to the source Spider gold and "
                 "executes to the same result set is a judge false negative, not "
                 "filtering. Compare this against the semantic drop rate before "
                 "describing the funnel as a quality barrier."),
    }


def build_validation_report(cfg: Any,
                            validated: Sequence[ValidatedRecord],
                            ledger: Any = None,
                            out_dir: str | Path | None = None,
                            *,
                            kept: Sequence[ValidatedRecord] | None = None,
                            budget_exhausted: bool | None = None,
                            n_raw: int = 0,
                            spider_data: Any | None = None) -> dict[str, Any]:
    """Aggregate a validation run into ``validation_report.json``.

    ``validated`` is the full output of :func:`validate_records` (including
    failures); ``kept`` is the subset that will become SFT data (defaults to
    ``[v for v in validated if v.report.passed]``).  Passing ``kept`` explicitly
    keeps the funnel honest when ``cfg.validate.drop_failed`` is false and the
    caller decides later.

    ``n_raw`` is the number of *raw* Spider examples the synthesis stage started
    from.  It is optional because validation never sees them, but supplying it
    (``len(spider_data.train)``) lets the report render the full funnel
    ``raw -> synth -> format-pass -> semantic-pass -> SFT`` that the README table
    needs, with the amplification factor (``synth / raw``) made explicit.

    Everything in the returned dict is derived from the reports -- no estimate is
    presented as a measurement.
    """
    vcfg = cfg.validate
    threshold = float(getattr(vcfg, "semantic_threshold", 0.5))
    n = len(validated)
    kept = list(kept) if kept is not None else [v for v in validated if v.report.passed]

    by_kind: dict[str, dict[str, Any]] = {}
    by_db: dict[str, dict[str, Any]] = {}
    code_counter: Counter[str] = Counter()
    judge_prompt = judge_completion = judge_calls = judge_errors = 0
    parse_errors = 0
    hn_by_error_kind: dict[str, dict[str, Any]] = {}
    difficulty_buckets: dict[str, dict[str, int]] = {}
    score_histogram: Counter[str] = Counter()
    score_by_verdict: dict[str, list[float]] = {v: [] for v in VERDICTS}

    for v in validated:
        rep, rec, kind = v.report, v.record, v.kind
        ok_format = format_ok(rep)
        judge_failed = (not rep.semantic_checked) and any(
            c in {"judge_error", "judge_parse_error"} for c in rep.issue_codes)
        if not judge_failed and rep.semantic_checked:
            judge_failed = any(c in {"judge_error", "judge_parse_error"} for c in rep.issue_codes)
        semantic_ok = (not rep.semantic_checked) or judge_failed or \
            rep.semantic_score >= threshold

        for code in rep.issue_codes:
            code_counter[code] += 1
        if "judge_error" in rep.issue_codes:
            judge_errors += 1
        if "judge_parse_error" in rep.issue_codes:
            parse_errors += 1
        if rep.semantic_checked:
            judge_calls += 1
            judge_prompt += rep.judge_prompt_tokens
            judge_completion += rep.judge_completion_tokens
            bucket = f"{min(9, int(rep.semantic_score * 10)) / 10:.1f}"
            score_histogram[bucket] += 1
            if rep.semantic_verdict in score_by_verdict:
                score_by_verdict[rep.semantic_verdict].append(rep.semantic_score)

        k = by_kind.setdefault(kind, {
            "n": 0, "format_pass": 0, "semantic_checked": 0, "semantic_pass": 0,
            "kept": 0, "judge_tokens": 0, "semantic_score_sum": 0.0,
            "difficulty_sum": 0.0, "blocking": 0,
        })
        k["n"] += 1
        k["format_pass"] += int(ok_format)
        k["blocking"] += int(bool(set(rep.issue_codes) & BLOCKING_CODES))
        k["semantic_checked"] += int(rep.semantic_checked)
        k["semantic_pass"] += int(rep.semantic_checked and semantic_ok)
        k["kept"] += int(rep.passed)
        k["judge_tokens"] += rep.judge_tokens
        k["semantic_score_sum"] += rep.semantic_score if rep.semantic_checked else 0.0
        k["difficulty_sum"] += float(getattr(rec, "difficulty", 0.0) or 0.0)

        d = by_db.setdefault(rec.db_id, {"n": 0, "format_pass": 0, "kept": 0})
        d["n"] += 1
        d["format_pass"] += int(ok_format)
        d["kept"] += int(rep.passed)

        bucket = f"{int(float(getattr(rec, 'difficulty', 0.0) or 0.0) * 5) / 5:.1f}"
        b = difficulty_buckets.setdefault(bucket, {"n": 0, "format_pass": 0, "kept": 0})
        b["n"] += 1
        b["format_pass"] += int(ok_format)
        b["kept"] += int(rep.passed)

        if kind == "hard_negative":
            ek = rec.error_kind or "unspecified"
            e = hn_by_error_kind.setdefault(ek, {
                "n": 0, "format_pass": 0, "kept": 0,
                "symptom_exec_error": 0, "symptom_wrong_result": 0,
                "bad_sql_same_result": 0,
            })
            e["n"] += 1
            e["format_pass"] += int(ok_format)
            e["kept"] += int(rep.passed)
            e["symptom_exec_error"] += int("hn_symptom_exec_error" in rep.issue_codes)
            e["symptom_wrong_result"] += int("hn_symptom_wrong_result" in rep.issue_codes)
            e["bad_sql_same_result"] += int("hn_bad_sql_same_result" in rep.issue_codes)

    for k in by_kind.values():
        checked = k["semantic_checked"]
        k["format_pass_rate"] = _rate(k["format_pass"], k["n"])
        k["semantic_pass_rate"] = _rate(k["semantic_pass"], checked)
        k["keep_rate"] = _rate(k["kept"], k["n"])
        k["avg_semantic_score"] = round(k["semantic_score_sum"] / checked, 4) if checked else 0.0
        k["avg_difficulty"] = round(k["difficulty_sum"] / k["n"], 4) if k["n"] else 0.0
        del k["semantic_score_sum"], k["difficulty_sum"]

    for d in by_db.values():
        d["format_pass_rate"] = _rate(d["format_pass"], d["n"])
        d["keep_rate"] = _rate(d["kept"], d["n"])

    n_format_pass = sum(1 for v in validated if format_ok(v.report))
    n_semantic_checked = sum(1 for v in validated if v.report.semantic_checked)
    n_semantic_pass = sum(
        1 for v in validated
        if v.report.semantic_checked and (
            v.report.semantic_score >= threshold or any(
                c in {"judge_error", "judge_parse_error"} for c in v.report.issue_codes)))
    n_kept = sum(1 for v in validated if v.report.passed)

    budget = int(getattr(vcfg, "max_total_tokens", 0) or 0)
    judge_total = judge_prompt + judge_completion
    if budget_exhausted is None:
        budget_exhausted = bool(budget and judge_total >= budget and
                                n_semantic_checked < n_format_pass)
    judge_cost = judge_cost_usd(cfg, judge_prompt, judge_completion)

    top_codes = [{"code": c, "count": k, "blocking": c in BLOCKING_CODES,
                  "stage": ("format" if c in BLOCKING_CODES or c in WARNING_CODES
                            else "semantic")}
                 for c, k in code_counter.most_common()
                 # a *success* code is not a failure code; keeping it would make
                 # "top failure codes" read as if the stage were failing
                 if c not in _SUCCESS_CODES][:15]

    ledger_summary: dict[str, Any] | None = None
    if ledger is not None:
        try:
            agg = ledger.aggregate()
            ledger_summary = {
                "totals": ledger.totals(),
                "by_stage": {s: {"calls": v["calls"], "total_tokens": v["total_tokens"],
                                 "cost_usd": v["cost_usd"]}
                             for s, v in agg.items()},
                "validate_stage": agg.get("validate"),
            }
        except Exception:  # noqa: BLE001 - report building must not die on a stub ledger
            ledger_summary = None

    # computed before the report dict so the sweep can score thresholds against
    # the same ground truth the audit used
    _flags = _ground_truth_flags(validated, spider_data)
    _audit = _ground_truth_audit(validated, spider_data, threshold)

    report: dict[str, Any] = {
        "generated_at": time.time(),
        "run_name": getattr(cfg, "run_name", ""),
        "config": {
            "enable_format": bool(getattr(vcfg, "enable_format", True)),
            "enable_semantic": bool(getattr(vcfg, "enable_semantic", True)),
            "semantic_sample": int(getattr(vcfg, "semantic_sample", 0) or 0),
            "semantic_threshold": threshold,
            "drop_failed": bool(getattr(vcfg, "drop_failed", True)),
            "judge_role": getattr(vcfg, "judge_role", "judge"),
            "judge_model": cfg.provider(getattr(vcfg, "judge_role", "judge")).model,
            "max_total_tokens": budget,
        },
        "counts": {
            "n": n,
            "by_kind": {k: v["n"] for k, v in by_kind.items()},
            "n_databases": len(by_db),
        },
        "stages": {
            "format": {
                "checked": n,
                "passed": n_format_pass,
                "pass_rate": _rate(n_format_pass, n),
                "blocking_codes": sorted(BLOCKING_CODES),
            },
            "semantic": {
                "checked": n_semantic_checked,
                "passed": n_semantic_pass,
                "pass_rate": _rate(n_semantic_pass, n_semantic_checked),
                "errors": judge_errors,
                "parse_errors": parse_errors,
            },
            "combined": {
                "passed": n_kept,
                "pass_rate": _rate(n_kept, n),
                "kept_written": len(kept),
            },
        },
        "funnel": _funnel(n, n_format_pass, n_semantic_checked, n_semantic_pass,
                          n_kept, int(n_raw)),
        "by_kind": by_kind,
        "by_db": by_db,
        "by_difficulty_bucket": dict(sorted(difficulty_buckets.items())),
        "judge": {
            "calls": judge_calls,
            "prompt_tokens": judge_prompt,
            "completion_tokens": judge_completion,
            "total_tokens": judge_total,
            "cost_usd": judge_cost,
            "budget_tokens": budget,
            "budget_exhausted": bool(budget_exhausted),
            "avg_tokens_per_call": round(judge_total / judge_calls, 1) if judge_calls else 0.0,
        },
        "top_failure_codes": top_codes,
        "ground_truth_audit": _audit,
        "threshold_sweep": _threshold_sweep(validated, _flags, threshold),
        "judge_score_distribution": {
            # the raw material for choosing `semantic_threshold` instead of
            # inheriting a default: a threshold that is too high silently deletes
            # correct records, which is invisible in a pass rate alone
            "histogram": dict(sorted(score_histogram.items())),
            "by_verdict": {
                v: {"n": len(vals),
                    "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
                    "min": round(min(vals), 4) if vals else 0.0,
                    "max": round(max(vals), 4) if vals else 0.0}
                for v, vals in score_by_verdict.items()},
            "threshold": threshold,
            "n_at_or_above_threshold": sum(1 for v in validated
                                           if v.report.semantic_checked
                                           and v.report.semantic_score >= threshold),
        },
        "hard_negatives": {
            "n": sum(v["n"] for k, v in hn_by_error_kind.items()),
            "by_error_kind": hn_by_error_kind,
            "symptom_totals": {c: code_counter.get(c, 0) for c in HN_SYMPTOM_CODES},
        },
        "ledger": ledger_summary,
        "headline": {
            "format_pass_rate_pct": round(_rate(n_format_pass, n) * 100, 2),
            "combined_pass_rate_pct": round(_rate(n_kept, n) * 100, 2),
            "judge_tokens_per_kept_record": round(judge_total / n_kept, 1) if n_kept else 0.0,
        },
    }
    if out_dir is not None:
        write_json(Path(out_dir) / "validation_report.json", report)
    return report


# --------------------------------------------------------------------------- #
# write the stage artifact
# --------------------------------------------------------------------------- #
def validate_and_write(cfg: Any,
                       records: Sequence[SynthRecord],
                       spider_data: SpiderData,
                       ledger: Any = None,
                       *,
                       provider: Any = None,
                       out_dir: str | Path | None = None,
                       record_usage_in_ledger: bool = False,
                       timeout_s: float | None = None) -> dict[str, Any]:
    """Validate, write ``validated.jsonl`` (+ report), and return the report.

    ``cfg.validate.drop_failed`` decides whether the artifact contains only
    passing records.  The report is always built from the **full** result, so
    dropping never hides the failure statistics.
    """
    out = Path(out_dir) if out_dir is not None else Path(cfg.paths.validated_dir)
    out.mkdir(parents=True, exist_ok=True)
    validated = validate_records(cfg, records, spider_data, ledger, provider=provider,
                                 record_usage_in_ledger=record_usage_in_ledger,
                                 timeout_s=timeout_s)
    drop = bool(getattr(cfg.validate, "drop_failed", True))
    kept = [v for v in validated if v.report.passed] if drop else list(validated)
    write_jsonl(out / "validated.jsonl", kept)
    # the raw corpus is the funnel's origin; validation never sees it, so take it
    # from spider_data (train split) to keep the README funnel complete
    n_raw = len(getattr(spider_data, "train", []) or [])
    report = build_validation_report(cfg, validated, ledger, out, kept=kept, n_raw=n_raw,
                                     spider_data=spider_data)
    return report
