"""Hard-negative synthesis: learn from a wrong query plus its diagnosis.

Why hard negatives
------------------
A model that only ever sees correct SQL never learns to *discriminate*. Hard
negatives add a contrastive signal: the same question with a plausible-but-wrong
query and a short diagnosis of the bug. The format stage renders this as "here is
a wrong query, fix it", which is a genuinely different function-calling behaviour
from "write a query from scratch".

The central risk: a fake negative
---------------------------------
``sqlutils.inject_error`` is a *best effort* rewriter -- it returns the input
unchanged when the requested corruption does not apply (no GROUP BY to remove, no
ORDER BY to flip). Measured on the fixture, several kinds are no-ops on the
majority of queries (``missing_distinct`` ~98%, ``wrong_order`` ~85%). Emitting
those would produce a record whose ``bad_sql`` equals the gold SQL: a training
example that teaches the model that a correct query is wrong. That is strictly
worse than no data.

So this module **never** trusts ``inject_error``. For each source example it walks
a preference list of error kinds and accepts a candidate only when *all three*
checks pass:

1. ``normalize_sql(bad) != normalize_sql(gold)``   -- textually a different query;
2. the corrupted SQL still passes static schema analysis (it must be a *plausible*
   mistake, not a syntax error, or the diagnosis becomes trivial);
3. **execution check** -- the corrupted SQL either fails to execute, or executes
   and returns a result set that differs from the gold result set
   (:func:`fcdata.dbexec.result_match`). This is the only check that proves the
   corruption is *semantically* wrong rather than merely spelled differently.

Check 3 runs **before** the teacher is called, so a rejected kind costs no tokens;
that is the whole point of verifying at generation time rather than letting the
downstream validator drop the record. If no kind survives, the example is skipped
and counted -- a skipped example is fine, a fabricated negative is not.

Difficulty stratification
-------------------------
When ``cfg.synth.hard_negative_use_difficulty`` is set, the fixed negative budget
is concentrated on the hardest examples (by ``Example.difficulty``): the hardest
``hard_negative_hard_fraction`` of slots go to the top of the difficulty ranking,
the rest are spread over the remaining examples. Hard queries are where the model
needs contrastive signal most, and it keeps the corpus from being dominated by
easy single-table negatives.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from ..config import Config
from ..dbexec import evaluate_pair, execute_sql
from ..records import SynthRecord
from ..sqlutils import (ERROR_KINDS, analyze_sql, inject_error, normalize_sql)
from ..tokens import TokenLedger
from ..providers import AsyncBatchRunner, Job, build_provider
from ..providers.prompts import correction_messages
from .common import (SYNTH_STAGE, KindReport, SourcePlan, build_sources,
                     make_record, stable_rng)

#: Kinds tried first: these always apply and produce semantically different queries.
PREFERRED_KINDS: tuple[str, ...] = (
    "wrong_table",
    "wrong_column",
    "wrong_aggregate",
    "wrong_operator",
    "wrong_literal",
    "missing_join",
    "missing_group_by",
    "wrong_order",
    "missing_distinct",
)

#: structural corruptions that need the query to already have that structure
_STRUCTURAL = frozenset({"missing_join", "missing_group_by", "wrong_order",
                         "missing_distinct"})


def _ordered_kinds(source_id: str) -> list[str]:
    """Preference list, rotated per source so the corpus is not all ``wrong_table``.

    The first four kinds apply to almost any query; rotating only their order
    keeps yield high while giving variety across examples.
    """
    kinds = list(PREFERRED_KINDS)
    head, tail = kinds[:4], kinds[4:]
    rng = stable_rng("hn-kinds", source_id)
    rng.shuffle(head)
    return head + tail


def _kinds_for_slot(source_id: str, slot: int) -> list[str]:
    """Rotate the preference list further for repeated slots on one example."""
    kinds = _ordered_kinds(source_id)
    if slot <= 0:
        return kinds
    shift = slot % max(1, len(kinds))
    return kinds[shift:] + kinds[:shift]


#: Static-analysis error codes that mean the SQL is *syntactically* broken rather
#: than semantically wrong. A broken statement is a different task ("fix the
#: syntax"), so those corruptions are rejected. Schema-level mistakes
#: (``unknown_column``, ``unknown_table_qualifier``, ...) are *accepted*: a
#: hallucinated column is one of the most realistic mistakes a text2sql model
#: makes, and a hard negative that shows it is exactly what we want to teach.
_SYNTAX_ERROR_CODES = frozenset({
    "empty_sql", "unbalanced_paren", "unterminated_string", "not_select",
    "destructive_statement", "no_from_clause",
})


def corrupt_candidate(plan: SourcePlan, kind: str,
                      db_path: Path | None) -> tuple[bool, str, str]:
    """Try one error kind; return ``(accepted, bad_sql, reason)``.

    Acceptance requires three independent things: the text changed, the statement
    is still syntactically valid SQL, and -- when a database is available -- the
    result is observably different. See the module docstring for why each matters.
    """
    bad = inject_error(plan.gold_sql, plan.db, kind)
    if not bad or normalize_sql(bad) == normalize_sql(plan.gold_sql):
        return False, "", "identical_after_injection"
    analysis = analyze_sql(bad, plan.db)
    syntax_errors = [i for i in analysis.errors if i.code in _SYNTAX_ERROR_CODES]
    if syntax_errors:
        return False, "", f"syntactically_broken:{syntax_errors[0].code}"
    if db_path is not None:
        differs, reason = _execution_differs(db_path, bad, plan.gold_sql)
        if not differs:
            return False, "", reason
        return True, bad, reason
    return True, bad, "unverified"


def _execution_differs(db_path: Path, bad_sql: str, gold_sql: str
                       ) -> tuple[bool, str]:
    """True when the corrupted query is *observably* wrong against real sqlite.

    Uses :func:`fcdata.dbexec.evaluate_pair` -- the same function the downstream
    validator uses -- rather than a bespoke comparison. That is deliberate: the
    validator drops a hard negative with ``hn_bad_sql_same_result`` when
    ``exec_match`` is true, and ``evaluate_pair`` treats row order as significant
    only for ``ORDER BY ... LIMIT`` queries (Spider convention). A corruption whose
    difference is invisible under that convention (e.g. flipping ``ORDER BY``
    direction without a ``LIMIT``) would be generated and then discarded
    downstream, wasting the teacher call. So generation-time verification must use
    the *same* notion of "different result" as validation.
    """
    pair = evaluate_pair(db_path, bad_sql, gold_sql)
    if not pair.get("gold_ok"):
        # Cannot compare without a working gold query; be conservative and reject
        # rather than emit a negative we cannot justify.
        return False, f"gold_not_executable:{str(pair.get('gold_error'))[:80]}"
    if not pair.get("executable"):
        return True, "exec_error"          # it breaks: clearly wrong
    if pair.get("exec_match"):
        return False, "same_result_set"
    reason = str(pair.get("reason") or "")
    if "row count" in reason:
        return True, "row_count_differs"
    return True, "result_differs"


def pick_negatives(plan: SourcePlan, kinds: Sequence[str],
                   db_path: Path | None) -> list[tuple[str, str, str]]:
    """Every acceptable ``(kind, bad_sql, exec_reason)`` for one example."""
    out: list[tuple[str, str, str]] = []
    for kind in kinds:
        ok, bad, reason = corrupt_candidate(plan, kind, db_path)
        if ok:
            out.append((kind, bad, reason))
    return out


# --------------------------------------------------------------------------- #
# selection / stratification
# --------------------------------------------------------------------------- #
def _slots_per_example(cfg: Config, plans: Sequence[SourcePlan]) -> dict[str, int]:
    """How many hard negatives each source example contributes.

    * ``hard_negative_use_difficulty = False`` -> uniform ``per_example`` slots for
      every example (full coverage, easy and hard alike).
    * ``True`` -> the *same total budget* is concentrated on hard examples: the
      hardest examples receive up to ``2 * per_example`` slots while easy ones may
      receive none. The total stays ``per_example * len(plans)`` so the configured
      token budget is unchanged -- stratification reallocates, it does not inflate.

    Concentration requires a budget smaller than the maximum capacity, which is
    why the hard slots are capped at ``2 * per_example`` rather than
    ``per_example``: with a flat cap every example would fill to the cap and the
    distribution would be identical to the uniform case.
    """
    per_example = int(cfg.synth.hard_negative_per_example)
    if per_example <= 0 or not plans:
        return {}
    total = per_example * len(plans)
    if not cfg.synth.hard_negative_use_difficulty:
        return {str(p.example.id): per_example for p in plans}

    ranked = sorted(plans, key=lambda p: (-float(p.difficulty), str(p.example.id)))
    fraction = max(0.0, min(1.0, float(cfg.synth.hard_negative_hard_fraction)))
    hard_budget = min(total, int(round(total * fraction)))
    hard_cap = per_example * 2
    slots: dict[str, int] = {str(p.example.id): 0 for p in ranked}

    # pass 1: concentrate on the hardest examples, up to hard_cap each
    assigned = 0
    while assigned < hard_budget:
        progressed = False
        for plan in ranked:
            if assigned >= hard_budget:
                break
            sid = str(plan.example.id)
            if slots[sid] < hard_cap:
                slots[sid] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break

    # pass 2: distribute any remainder round-robin, hardest first, up to hard_cap
    remaining = total - assigned
    while remaining > 0:
        progressed = False
        for plan in ranked:
            if remaining <= 0:
                break
            sid = str(plan.example.id)
            if slots[sid] < hard_cap:
                slots[sid] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return slots


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def _db_path_for(spider_data: Any, db_id: str) -> Path | None:
    """Resolve the sqlite path for a db, or ``None`` when unavailable."""
    try:
        p = spider_data.db_path(db_id)
    except (FileNotFoundError, KeyError, AttributeError):
        return None
    return Path(p) if Path(p).exists() else None


def synth_hard_negative(cfg: Config, spider_data: Any, ledger: TokenLedger,
                        *, plans: Sequence[SourcePlan] | None = None,
                        runner: AsyncBatchRunner | None = None
                        ) -> tuple[list[SynthRecord], KindReport]:
    """Generate verified hard negatives plus their correction notes."""
    report = KindReport(kind="hard_negative")
    if int(cfg.synth.hard_negative_per_example) <= 0:
        return [], report
    if plans is None:
        plans = build_sources(cfg, spider_data)
    if not plans:
        return [], report

    slots = _slots_per_example(cfg, plans)
    jobs: list[Job] = []
    keys: list[tuple[SourcePlan, int, str, str, str]] = []
    planned_examples = 0
    for plan in plans:
        n_slots = slots.get(str(plan.example.id), 0)
        if n_slots <= 0:
            continue
        db_path = _db_path_for(spider_data, plan.db_id)
        candidates: list[tuple[str, str, str]] = []
        for slot in range(n_slots):
            # try the rotated preference list, skipping kinds already used here
            fresh = [k for k in _kinds_for_slot(str(plan.example.id), slot)
                     if k not in {c[0] for c in candidates}]
            found = pick_negatives(plan, fresh, db_path)
            if not found:
                continue
            candidates.append(found[0])
            kind, bad_sql, reason = found[0]
            jobs.append(Job(
                messages=correction_messages(plan.ddl, plan.question, bad_sql,
                                             plan.gold_sql, kind, plan.db_id),
                task="hard_negative",
                sample_id=f"{plan.example.id}#hn{slot}",
                temperature=0.2,
                max_tokens=160,
                meta={"source_id": plan.example.id, "index": slot,
                      "error_kind": kind, "verified_by": reason},
            ))
            keys.append((plan, slot, kind, bad_sql, reason))
        if candidates:
            planned_examples += 1
        else:
            # No error kind produced an observably different query: the example is
            # unusable for contrastive training. Counted separately from a *failed
            # teacher call* -- nothing was dispatched, no token was spent.
            report.examples_skipped += 1
            if "no_viable_error_kind" not in report.errors:
                report.errors.append("no_viable_error_kind")
    report.planned = len(jobs)
    if not jobs:
        return [], report

    provider = build_provider("cheap", cfg, ledger, stage=SYNTH_STAGE)
    active_runner = runner or AsyncBatchRunner(provider, ledger, cfg, stage=SYNTH_STAGE)
    completions = active_runner.run(jobs, stage=SYNTH_STAGE)

    records: list[SynthRecord] = []
    for (plan, slot, kind, bad_sql, reason), comp in zip(keys, completions):
        report.prompt_tokens += comp.prompt_tokens
        report.completion_tokens += comp.completion_tokens
        if comp.error == "budget_exhausted":
            report.skipped_budget += 1
            continue
        if not comp.ok:
            report.failed += 1
            if comp.error:
                report.errors.append(comp.error[:160])
            continue
        note = clean_note(comp.text)
        if not note:
            report.dropped_invalid += 1
            report.errors.append("empty_correction_note")
            continue
        records.append(make_record(
            kind="hard_negative", plan=plan, index=slot, question=plan.question,
            completion=comp, role="cheap", model=provider.model,
            bad_sql=bad_sql, error_kind=kind, correction_note=note,
            meta={"verified_by": reason,
                  "gold_executes": True,
                  "source_question": plan.question,
                  "difficulty": plan.difficulty},
        ))
        report.ok += 1
    report.examples_covered = planned_examples
    return records, report


def clean_note(text: str) -> str:
    """Normalize a correction note to one tidy paragraph."""
    body = (text or "").strip()
    body = re.sub(r"^```(?:text)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    body = re.sub(r"^\s*(?:diagnosis|correction note|note|explanation)\s*[:\-]\s*", "",
                  body, flags=re.IGNORECASE)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def verify_negative(db_path: Path | None, gold_sql: str, bad_sql: str) -> dict[str, Any]:
    """Re-verify a hard negative end to end (used by tests and by the report).

    Mirrors the generation-time checks so a caller can audit an existing corpus
    without re-running synthesis.
    """
    identical = normalize_sql(bad_sql) == normalize_sql(gold_sql)
    if db_path is None:
        return {"identical": identical, "executed": False,
                "exec_differs": None, "reason": "no_database"}
    pair = evaluate_pair(db_path, bad_sql, gold_sql)
    return {
        "identical": identical,
        "executed": bool(pair.get("gold_ok")),
        "pred_executable": bool(pair.get("executable")),
        "exec_differs": not bool(pair.get("exec_match")),
        "reason": pair.get("reason", ""),
    }
