"""Validation stage 1 -- *static* + *dynamic* format checking of synthesized SQL.

WHY this stage exists
---------------------
Synthesis is cheap; an LLM generator happily emits a record that *looks* like
training data but is unusable.  A wrong target is worse than a missing target,
because it teaches the student model to be confidently wrong.  Three independent
failure modes are checked before a record may enter the SFT corpus:

1. **Parse / syntax** -- the string must be a single read-only ``SELECT``/``WITH``.
   Checked by :func:`fcdata.sqlutils.analyze_sql`, which tokenises without a real
   parser.  Catches truncated generations, prose/fence leakage, DML/DDL and
   unbalanced quotes or parentheses.
2. **Schema linkage** -- every table and column must exist in the Spider schema.
   This is the classic generator hallucination ("no such column") and it is
   detectable *without* touching sqlite, so it is cheap enough to run on the
   whole corpus.
3. **Executability** -- the statement must really run on the fixture sqlite
   file.  Static checks cannot see ambiguous column references across joins or
   aggregate misuse; real execution can.  This project never simulates a metric,
   so "valid" means "we actually ran it".

Hard negatives carry a *premise* that must hold: ``bad_sql`` must differ from
the gold query **and** must actually be wrong (fail to execute, or return a
different result set).  A "negative" that returns exactly the gold rows teaches
the model to fix a non-bug, so it is rejected here rather than in the report.

Design notes
------------
* :func:`check_format` returns a :class:`~fcdata.records.ValidationReport` with
  only the *format* half populated; the semantic judge fills the rest.  That
  split lets the expensive LLM stage run on a sample while the format stage runs
  on everything.
* ``schema_refs_ok`` is ``False`` when no :class:`~fcdata.schema.Database` was
  supplied -- we never report a check we did not perform.  In the pipeline the
  caller always supplies one (or the record is rejected with ``missing_database``).
* Codes are stable identifiers (not prose) so the report can aggregate them;
  :data:`BLOCKING_CODES` is the authoritative "this record must be dropped" set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

from ..dbexec import execute_sql, result_match
from ..records import SynthRecord, ValidationReport
from ..schema import Database
from ..sqlutils import ERROR_KINDS, analyze_sql, extract_sql, normalize_sql

# --------------------------------------------------------------------------- #
# failure taxonomy
# --------------------------------------------------------------------------- #
#: static-analysis codes that mean "this is not a well-formed SELECT statement"
SYNTAX_CODES: frozenset[str] = frozenset({
    "empty_sql",
    "unterminated_string",
    "unbalanced_paren",
    "not_select",
    "destructive_statement",
})

#: static-analysis codes that mean "the SQL references the wrong schema"
SCHEMA_CODES: frozenset[str] = frozenset({
    "unknown_table",
    "unknown_column",
    "unknown_table_qualifier",
    "column_used_as_table",
    "no_from_clause",
})

#: codes that make a record unusable as SFT data
BLOCKING_CODES: frozenset[str] = SYNTAX_CODES | SCHEMA_CODES | frozenset({
    "sql_unparseable",
    "exec_failed",
    "missing_database",
    "unknown_db_id",
    "hn_bad_sql_missing",
    "hn_bad_sql_identical",
    "hn_bad_sql_same_result",
})

#: codes that are recorded but do not fail the record
WARNING_CODES: frozenset[str] = frozenset({
    "schema_check_unavailable",
    "exec_check_unavailable",
    "sql_normalized_from_text",
    "hn_error_kind_unknown",
    "hn_symptom_exec_error",
    "hn_symptom_wrong_result",
    "hn_bad_sql_executes",
    "cot_reasoning_missing",
    "paraphrase_identical",
    "aggregate_without_group_by",
})

#: hard-negative symptom codes, kept separate so the report can break them down
HN_SYMPTOM_CODES: tuple[str, ...] = ("hn_symptom_exec_error", "hn_symptom_wrong_result")


# --------------------------------------------------------------------------- #
# result helpers
# --------------------------------------------------------------------------- #
def _add(report: ValidationReport, code: str) -> None:
    """Append ``code`` once, preserving discovery order (stable for diffs)."""
    if code not in report.issue_codes:
        report.issue_codes.append(code)


def format_ok(report: ValidationReport) -> bool:
    """True iff the format stage passed: parses, links, executes, no blocker.

    ``ValidationReport`` has no dedicated "format passed" flag (the contract's
    ``passed`` is the *combined* verdict), so this predicate is the single
    definition used by both :func:`check_format` and the report builder.
    """
    if not (report.sql_parses and report.schema_refs_ok and report.executes):
        return False
    return not (set(report.issue_codes) & BLOCKING_CODES)


def _summarize(report: ValidationReport, kind: str, error_kind: str = "") -> str:
    """Human-readable one-liner stored on the report (logs / debugging)."""
    bad = [c for c in report.issue_codes if c in BLOCKING_CODES]
    tag = kind + (f"[{error_kind}]" if error_kind else "")
    if not bad:
        return f"{tag}: format ok"
    detail = f" ({report.exec_error})" if report.exec_error else ""
    return f"{tag}: format failed: {', '.join(bad)}{detail}"


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #
def check_format(record: SynthRecord,
                 db: Database | None = None,
                 db_path: str | os.PathLike[str] | None = None,
                 *,
                 timeout_s: float = 5.0) -> ValidationReport:
    """Run the static + schema + execution checks for one synthesized record.

    Parameters
    ----------
    record:
        The synthesized record.  ``record.sql`` is always the *gold* SQL (also
        for hard negatives -- the corrupted query lives in ``record.bad_sql``).
    db:
        Spider schema used for table/column resolution.
    db_path:
        sqlite file used for real execution.  ``None`` downgrades the execution
        check to a recorded warning (see module docstring); the pipeline always
        resolves a path and rejects the record with ``missing_database`` when it
        cannot.

    Returns
    -------
    ValidationReport
        Format-stage fields populated, ``semantic_*`` untouched, and
        ``passed`` set to :func:`format_ok` (provisional until the judge runs).

    Side effect
    -----------
    If ``record.sql`` contains prose/fences, the extracted statement replaces
    it and ``sql_normalized_from_text`` is recorded.  Validation is the right
    place to repair that: every later stage consumes ``record.sql`` verbatim, so
    leaving a fenced blob in place would poison the SFT target.
    """
    report = ValidationReport()
    kind = getattr(record, "kind", "") or ""

    raw_sql = (record.sql or "").strip()
    if not raw_sql:
        _add(report, "empty_sql")
        report.reason = _summarize(report, kind)
        return report

    sql = extract_sql(raw_sql)
    if not sql:
        _add(report, "sql_unparseable")
        report.reason = _summarize(report, kind)
        return report
    if sql != raw_sql:
        # repair in place; downstream stages consume record.sql as the target
        record.sql = sql
        _add(report, "sql_normalized_from_text")

    # ---- 1. static + schema analysis ----------------------------------
    analysis = analyze_sql(sql, db)
    for issue in analysis.issues:
        _add(report, issue.code)
        if issue.severity == "error":
            report.exec_error = report.exec_error or issue.detail

    report.sql_parses = not any(
        i.code in SYNTAX_CODES and i.severity == "error" for i in analysis.issues)
    if db is None:
        _add(report, "schema_check_unavailable")
        report.schema_refs_ok = False
    else:
        report.schema_refs_ok = not any(
            i.code in SCHEMA_CODES and i.severity == "error" for i in analysis.issues)

    # ---- 2. real execution -------------------------------------------
    gold_result = None
    if db_path is None:
        _add(report, "exec_check_unavailable")
    else:
        gold_result = execute_sql(db_path, sql, timeout_s)
        report.executes = bool(gold_result.ok)
        if not gold_result.ok:
            report.exec_error = gold_result.error or "execution failed"
            _add(report, "exec_failed")
        else:
            # a successful run clears a stale static error message
            if not (set(report.issue_codes) & BLOCKING_CODES):
                report.exec_error = ""

    # ---- 3. kind-specific payload checks -----------------------------
    if kind == "hard_negative":
        _check_hard_negative(record, report, db, db_path, gold_result, timeout_s)
    elif kind == "cot":
        if not (record.reasoning or "").strip():
            _add(report, "cot_reasoning_missing")
    elif kind == "paraphrase":
        if (record.original_question or "").strip() and \
                normalize_question(record.question) == normalize_question(record.original_question):
            _add(report, "paraphrase_identical")

    report.passed = format_ok(report)
    report.reason = _summarize(report, kind, getattr(record, "error_kind", ""))
    return report


def _check_hard_negative(record: SynthRecord,
                         report: ValidationReport,
                         db: Database | None,
                         db_path: str | os.PathLike[str] | None,
                         gold_result: Any,
                         timeout_s: float) -> None:
    """Extra premise checks for error -> correction records."""
    bad_sql = extract_sql((record.bad_sql or "").strip()) or (record.bad_sql or "").strip()
    if not bad_sql:
        _add(report, "hn_bad_sql_missing")
        return

    # the negative must be textually different from the gold query
    if normalize_sql(bad_sql) == normalize_sql(record.sql):
        _add(report, "hn_bad_sql_identical")
        return

    error_kind = getattr(record, "error_kind", "") or ""
    if error_kind and error_kind not in ERROR_KINDS:
        _add(report, "hn_error_kind_unknown")

    if db_path is None or gold_result is None or not gold_result.ok:
        return

    bad_result = execute_sql(db_path, bad_sql, timeout_s)
    if not bad_result.ok:
        # the most common and most instructive symptom: sqlite rejects it
        _add(report, "hn_symptom_exec_error")
        return

    _add(report, "hn_bad_sql_executes")
    match = result_match(bad_result, gold_result)
    if match.exec_match:
        # premise is false: "fixing" this changes nothing, so it is not a bug
        _add(report, "hn_bad_sql_same_result")
    else:
        _add(report, "hn_symptom_wrong_result")


def normalize_question(text: str) -> str:
    """Loose question normalisation used to detect no-op paraphrases."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# --------------------------------------------------------------------------- #
# convenience: check many records
# --------------------------------------------------------------------------- #
def check_records(records: Sequence[SynthRecord],
                  databases: dict[str, Database],
                  db_path_for: Callable[[str], Path | None] | None = None,
                  *,
                  timeout_s: float = 5.0) -> list[ValidationReport]:
    """Check a batch; missing schema/database becomes a blocking record, not an error.

    Kept separate from :func:`fcdata.validate.validate_records` so the static
    checks can be unit-tested (or reused by a linter) without any LLM involved.
    """
    out: list[ValidationReport] = []
    for rec in records:
        db = databases.get(rec.db_id)
        path = db_path_for(rec.db_id) if db_path_for is not None else None
        if db is None:
            rep = ValidationReport()
            _add(rep, "unknown_db_id")
            rep.reason = _summarize(rep, rec.kind)
            out.append(rep)
            continue
        if path is None:
            rep = ValidationReport()
            _add(rep, "missing_database")
            rep.reason = _summarize(rep, rec.kind)
            out.append(rep)
            continue
        out.append(check_format(rec, db, path, timeout_s=timeout_s))
    return out
