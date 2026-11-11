"""CoT synthesis: teach the model *why* a query looks the way it does.

What a record is
----------------
One ``SynthRecord(kind="cot")`` per (source example, sample index). The record
carries the gold SQL as its target plus a natural-language reasoning trace in
``reasoning``. The format stage later turns that trace into the assistant's
pre-call reasoning, so the model learns the *mapping* from question+schema to
query rather than a question->SQL lookup table.

Why the trace is validated, not trusted
---------------------------------------
A teacher model can produce a fluent trace that is factually wrong ("group by
city" for a query that never groups). Training on it teaches the student to
hallucinate. So every trace is checked against the *parsed gold SQL* before it
becomes a record:

* it must contain a ``Final SQL:`` line whose statement normalizes to the gold SQL;
* it must name at least one real table and one real column from the schema;
* it must not mention a table that is not in the schema (the classic
  hallucination failure mode).

Records that fail are counted and dropped (``dropped_invalid``) with the reason
recorded, never silently repaired -- a repair would hide a teacher-quality
regression instead of reporting it.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..config import Config
from ..records import SynthRecord
from ..sqlutils import normalize_sql
from ..tokens import TokenLedger
from ..providers import AsyncBatchRunner, Job, build_provider
from ..providers.base import Completion
from ..providers.prompts import COT_SYSTEM, cot_messages
from ..providers.sqlread import iter_refs
from .common import (SYNTH_STAGE, KindReport, SourcePlan, build_sources,
                     make_record, summarize)

#: the marker the teacher must emit before the final query
FINAL_SQL_MARKER = "final sql:"


def _plan_jobs(plans: Sequence[SourcePlan], per_example: int,
               role: str) -> tuple[list[Job], list[tuple[SourcePlan, int]]]:
    jobs: list[Job] = []
    keys: list[tuple[SourcePlan, int]] = []
    for plan in plans:
        for idx in range(per_example):
            jobs.append(Job(
                messages=cot_messages(plan.ddl, plan.question, plan.gold_sql,
                                      plan.db_id),
                task="cot",
                sample_id=f"{plan.example.id}#cot{idx}",
                # CoT is the highest-value signal in the corpus: keep it factual
                # rather than creative, so temperature is lower than paraphrase.
                temperature=0.3,
                max_tokens=384,
                meta={"source_id": plan.example.id, "index": idx},
            ))
            keys.append((plan, idx))
    return jobs, keys


def validate_trace(text: str, plan: SourcePlan) -> tuple[bool, str]:
    """Check a reasoning trace against the gold SQL and the schema."""
    if not text or not text.strip():
        return False, "empty_trace"
    body = text.strip()
    # 1. the final query must be present and must be the gold query
    lower = body.lower()
    pos = lower.rfind(FINAL_SQL_MARKER)
    if pos < 0:
        return False, "missing_final_sql"
    final = body[pos + len(FINAL_SQL_MARKER):].strip().splitlines()[0].strip()
    if not final:
        return False, "empty_final_sql"
    if normalize_sql(final) != normalize_sql(plan.gold_sql):
        return False, "final_sql_mismatch"
    # 2. the trace must actually explain with schema vocabulary
    prose = body[:pos]
    if len(prose.split()) < 12:
        return False, "trace_too_short"
    schema_tables = {t.norm for t in plan.schema.tables}
    mentioned = {w.lower() for w in _words(prose)}
    if not (mentioned & schema_tables):
        return False, "no_table_mentioned"
    refs = list(iter_refs(plan.facts))
    if refs:
        col_names = {r.column.lower() for r in refs}
        if not (mentioned & col_names):
            return False, "no_column_mentioned"
    # 3. no hallucinated table *in table position*.
    #    Only identifiers used the way a table is used ("from X", "join X",
    #    "table X") are checked, and only when the token is *not* ordinary English
    #    prose: "answered from the following table" is a sentence, not a schema
    #    claim. Two filters keep the check precise -- a prose stop list, and
    #    skipping any token immediately followed by the word "table" (which is how
    #    a writer refers to *the* table rather than to a named one).
    for name in _table_mentions(prose):
        if name in schema_tables or name.rstrip("s") in schema_tables:
            continue
        if name in _PROSE_STOPWORDS:
            continue
        return False, f"hallucinated_table:{name}"
    return True, "ok"


#: English words that legitimately follow "from"/"join"/"table" in prose. Kept
#: deliberately broad: a false *rejection* costs one record, while a false
#: *acceptance* would let a hallucinated table into the training corpus.
_PROSE_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "each", "every", "all",
    "its", "their", "his", "her", "both", "only", "just", "first", "then", "next",
    "keeps", "keep", "returns", "return", "gives", "give", "contains", "contain",
    "matches", "match", "produces", "produce", "rows", "row", "records", "record",
    "values", "value", "columns", "column", "tables", "table", "groups", "group",
    "results", "result", "here", "there", "which", "where", "one", "two", "three",
    "below", "above", "them", "it", "us", "me", "you", "we", "they", "and", "or",
    "not", "no", "any", "some", "more", "less", "than", "as", "by", "to", "in",
    "on", "at", "for", "with", "without", "having", "being", "is", "are", "was",
    "were", "be", "been", "do", "does", "did", "can", "will", "would", "should",
    "must", "may", "might", "same", "different", "other", "another", "such",
    "single", "multiple", "surviving", "remaining", "filtered", "joined",
    "selected", "computed", "aggregated", "ordered", "limited", "counted",
    "following", "preceding", "above", "below", "inside", "outside", "within",
    "across", "between", "under", "over", "into", "onto", "upon", "via",
    "tables_", "sources", "source", "input", "output", "answer", "question",
    "query", "clause", "condition", "conditions", "predicate", "expression",
    "aggregate", "aggregation", "calculation", "comparison", "order", "limit",
    "join", "subquery", "alias", "qualifier", "schema", "database", "dataset",
})


#: ``from X`` / ``join X`` -- the two positions where a SQL writer *must* name a
#: relation. "table X" is deliberately *not* a trigger: in prose "the table holds"
#: is a sentence about the schema block, not a claim that a relation named "holds"
#: exists. The reverse form ("the publisher table") is handled separately below.
_TABLE_MENTION_RE = re.compile(
    r"(?:\bfrom|\bjoin)\s+(?:the\s+|a\s+|an\s+)?[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
#: "the <name> table" -- a named relation spelled backwards.
_NAMED_TABLE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+table\b", re.IGNORECASE)


def _table_mentions(text: str) -> list[str]:
    """Identifiers the trace uses as a relation name, lowercased.

    Two forms are recognised: ``from X`` / ``join X``, and the reverse
    ``the X table``. A token followed by the word "table" in the *forward* form
    ("read from the following table") is prose, not a relation name, and is
    excluded -- otherwise every trace that points at the schema block would be
    rejected as hallucinating a table called "following".
    """
    out: list[str] = []
    for m in _TABLE_MENTION_RE.finditer(text):
        tail = text[m.end():m.end() + 8].lstrip()
        if tail.lower().startswith("table"):
            continue
        out.append(m.group(1).lower())
    for m in _NAMED_TABLE_RE.finditer(text):
        name = m.group(1).lower()
        # "the table" / "a table" are articles, not names
        if name in _PROSE_STOPWORDS:
            continue
        out.append(name)
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def synth_cot(cfg: Config, spider_data: Any, ledger: TokenLedger,
              *, plans: Sequence[SourcePlan] | None = None,
              runner: AsyncBatchRunner | None = None) -> tuple[list[SynthRecord], KindReport]:
    """Generate the CoT records for every planned source example."""
    per_example = int(cfg.synth.cot_per_example)
    report = KindReport(kind="cot")
    if per_example <= 0:
        return [], report
    if plans is None:
        plans = build_sources(cfg, spider_data)
    jobs, keys = _plan_jobs(plans, per_example, "cheap")
    report.planned = len(jobs)
    if not jobs:
        return [], report

    provider = build_provider("cheap", cfg, ledger, stage=SYNTH_STAGE)
    active_runner = runner or AsyncBatchRunner(provider, ledger, cfg, stage=SYNTH_STAGE)
    completions = active_runner.run(jobs, stage=SYNTH_STAGE)

    records: list[SynthRecord] = []
    for (plan, idx), comp in zip(keys, completions):
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
        ok, reason = validate_trace(comp.text, plan)
        if not ok:
            report.dropped_invalid += 1
            if reason not in report.errors:
                report.errors.append(f"invalid_trace:{reason}")
            continue
        records.append(make_record(
            kind="cot", plan=plan, index=idx, question=plan.question,
            completion=comp, role="cheap", model=provider.model,
            reasoning=_clean_trace(comp.text),
            meta={"teacher_system": COT_SYSTEM.splitlines()[0][:80],
                  "n_tables": len(plan.facts.tables),
                  "n_joins": len(plan.facts.joins),
                  "has_aggregate": plan.facts.has_aggregate,
                  "source_question": plan.question},
        ))
        report.ok += 1
    return records, report


def _clean_trace(text: str) -> str:
    """Normalize the trace for storage: strip fences, collapse blank lines."""
    import re

    body = text.strip()
    body = re.sub(r"^```(?:text)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()
