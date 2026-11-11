"""Partial-credit metrics: the learning signal that strict exec accuracy hides.

WHY this module exists
----------------------
Execution accuracy is **all-or-nothing**: a generated SQL statement counts only
if it runs *and* returns exactly the gold result set. One wrong token anywhere --
a column name, a literal, an alias in a JOIN condition -- makes the statement
either fail or return different rows. For a model with per-token accuracy ``p``
and a target of ``L`` tokens, the probability of a byte-perfect statement is
roughly ``p ** L``; at p=0.73 and L=60 that is ~1e-9. So ``exec_acc`` is
essentially zero for *any* model below ~99% token accuracy, and a small
from-scratch reference model will sit at 0.0% while demonstrably learning.

Reporting only ``exec_acc`` therefore throws away the entire learning signal and
makes a working pipeline look broken. The metrics here recover it:

===================  =========================================================
metric               what it isolates
===================  =========================================================
``token_acc``        raw token-level agreement of the generated SQL (1-edit
                     distance / max length), i.e. "how close was the text"
``skeleton_match``   the *query shape* is right with literals abstracted, i.e.
                     "same structure, possibly different constants"
``table_set_match``  the referenced tables are right, i.e. "understood which
                     tables the question needs" -- the schema-grounding signal
``column_f1``        F1 over referenced (table, column) pairs -- schema linking
                     precision *and* recall, so hallucinated columns are
                     penalised rather than hidden by recall alone
``clause_match``     per-clause agreement (JOIN / aggregate / GROUP BY / ORDER
                     BY / LIMIT), which localises *which* construct is missing
``valid_sql_rate``   parses and executes (same as ``executable_rate``)
===================  =========================================================

Every value is computed from the model's real generated SQL and the real gold
SQL using the project's own tokenizer/analyzer (:mod:`fcdata.sqlutils`), so the
numbers are reproducible and auditable. Nothing here executes SQL itself -- the
execution-based numbers still come from :mod:`fcdata.dbexec` via ``Prediction``.

The failure taxonomy additionally classifies *why* a statement failed in terms a
data engineer can act on (wrong table / wrong column / missing join / wrong
aggregate / unparseable), which maps directly onto the hard-negative error kinds
the synthesis stage injects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..records import Prediction
from ..schema import Database
from ..sqlutils import SqlAnalysis, analyze_sql, sql_skeleton, tokenize

#: Clause flags compared between prediction and gold.
CLAUSE_FLAGS: tuple[str, ...] = (
    "has_join", "has_aggregate", "has_group_by", "has_order_by", "has_limit",
    "has_subquery", "has_set_op",
)

#: Stable order for the "why did it fail" taxonomy (checked in this order).
PARTIAL_FAILURE_TAXONOMY: tuple[str, ...] = (
    "unparseable",       # no SQL extracted, or not a SELECT/WITH statement
    "not_executable",    # parsed but sqlite refused it
    "wrong_table",       # references a table that does not exist in the schema
    "wrong_column",      # references a column that does not exist
    "missing_join",      # gold joins, prediction does not
    "extra_join",        # prediction joins, gold does not
    "wrong_aggregate",   # aggregate function presence/mix differs
    "missing_group_by",  # gold groups, prediction does not
    "missing_order_by",  # gold orders, prediction does not
    "wrong_values",      # executable, same shape, different rows
)


# --------------------------------------------------------------------------- #
# token-level similarity
# --------------------------------------------------------------------------- #
def _token_texts(sql: str) -> list[str]:
    """Lowercased token values; the alignment unit for ``token_acc``."""
    return [t.value.lower() for t in tokenize(sql or "")]


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences (iterative DP).

    Token-level (not character-level) because a single wrong column name is one
    token error, not twelve character errors -- the token view is what a reader
    can interpret as "one identifier wrong".
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def token_accuracy(pred_sql: str, gold_sql: str) -> float:
    """``1 - edit_distance / max(len)`` over token sequences, in [0, 1].

    This is a *normalised edit similarity*, not per-position accuracy: it does
    not punish a correct statement for shifting tokens, which positional
    accuracy would (a single inserted token would misalign everything after it).
    """
    pt = _token_texts(pred_sql)
    gt = _token_texts(gold_sql)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    longest = max(len(pt), len(gt))
    return max(0.0, 1.0 - edit_distance(pt, gt) / longest)


def prefix_match_ratio(pred_sql: str, gold_sql: str) -> float:
    """Fraction of gold tokens matched in the longest common prefix.

    A sharper signal than edit similarity for "how far into the statement did the
    model get before going wrong", which is what a curriculum wants to know.
    """
    pt = _token_texts(pred_sql)
    gt = _token_texts(gold_sql)
    if not gt:
        return 1.0 if not pt else 0.0
    n = 0
    for a, b in zip(pt, gt):
        if a != b:
            break
        n += 1
    return n / len(gt)


# --------------------------------------------------------------------------- #
# schema-linking metrics
# --------------------------------------------------------------------------- #
def _table_set(analysis: SqlAnalysis) -> set[str]:
    return {t.lower() for t in analysis.tables}


def referenced_tables(sql: str, db: Database | None = None) -> set[str]:
    """Tables named after FROM/JOIN, whether or not they exist in the schema.

    ``SqlAnalysis.tables`` only contains *resolved* tables, so a hallucinated
    name vanishes from it -- which would make hallucination invisible. This walks
    the tokens directly to recover every name in a table position, then checks it
    against the schema when one is supplied.
    """
    from ..schema import normalize_ident  # noqa: PLC0415 - local import

    names: set[str] = set()
    toks = tokenize(sql or "")
    for i, tok in enumerate(toks):
        if tok.kind == "keyword" and tok.value.lower() in {"from", "join"}:
            j = i + 1
            while j < len(toks):
                nxt = toks[j]
                if nxt.kind == "punct" and nxt.value == "(":
                    break
                if nxt.kind == "ident":
                    names.add(normalize_ident(nxt.value))
                elif nxt.kind == "punct" and nxt.value == ",":
                    pass
                elif nxt.kind == "keyword" and nxt.value.lower() in {
                        "select", "as", "on", "where", "group", "order", "limit",
                        "join", "inner", "left", "right", "outer", "having"}:
                    break
                j += 1
    if db is not None:
        known = {t.norm for t in db.tables}
        return {n for n in names if n in known or n not in known}
    return names


def hallucinated_tables(sql: str, db: Database | None) -> set[str]:
    """Table names referenced by ``sql`` that do not exist in ``db``."""
    if db is None:
        return set()
    names = referenced_tables(sql)
    known = {t.norm for t in db.tables}
    return {n for n in names if n not in known}


def _column_pairs(analysis: SqlAnalysis) -> set[tuple[str | None, str]]:
    """Referenced ``(table_or_None, column)`` pairs, lowercased."""
    out: set[tuple[str | None, str]] = set()
    for tbl, col in analysis.columns:
        out.add((tbl.lower() if tbl else None, col.lower()))
    return out


def column_f1(pred_an: SqlAnalysis, gold_an: SqlAnalysis) -> tuple[float, float, float]:
    """``(precision, recall, f1)`` over referenced (table, column) pairs.

    Unqualified columns (``table=None``) are compared on the column name alone so
    a correct column is not counted wrong merely because the gold used an alias
    qualifier the prediction omitted.

    **Both sets empty scores 0, not 1.** ``SELECT 1`` references no columns, and
    a gold ``SELECT COUNT(*)`` also references none; treating that agreement as a
    perfect score would award full schema-linking credit to a prediction that
    named no schema object at all -- which is exactly the failure this metric is
    meant to expose. Perfect credit requires at least one matched pair.
    """
    pred = _column_pairs(pred_an)
    gold = _column_pairs(gold_an)
    if not pred and not gold:
        return (0.0, 0.0, 0.0)
    if not pred or not gold:
        return (0.0, 0.0, 0.0)

    def _match(a: set[tuple[str | None, str]], b: set[tuple[str | None, str]]) -> int:
        """Count pairs in ``a`` matched by a pair in ``b`` (ignoring None table)."""
        b_cols = {c for _t, c in b}
        hits = 0
        for tbl, col in a:
            if (tbl, col) in b:
                hits += 1
            elif tbl is None and col in b_cols:
                hits += 1
        return hits

    tp = _match(pred, gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return (round(precision, 6), round(recall, 6), round(f1, 6))


def clause_flags(analysis: SqlAnalysis) -> dict[str, bool]:
    return {flag: bool(getattr(analysis, flag)) for flag in CLAUSE_FLAGS}


def clause_match(pred_an: SqlAnalysis, gold_an: SqlAnalysis) -> dict[str, bool]:
    """Per-clause boolean agreement between prediction and gold.

    Only clauses the **gold actually uses** are scored. A clause the gold does not
    have (e.g. ``has_join`` when the gold is a single-table query) is not a
    meaningful signal: both sides are False, so a prediction that named no schema
    object at all would collect free agreement. Scoring only gold-present clauses
    keeps the metric interpretable as "did the model produce the constructs the
    answer requires".
    """
    return {flag: bool(getattr(pred_an, flag))
            for flag in CLAUSE_FLAGS if bool(getattr(gold_an, flag))}


# --------------------------------------------------------------------------- #
# per-prediction scoring
# --------------------------------------------------------------------------- #
@dataclass
class PartialScore:
    """Partial-credit score for one prediction."""

    token_acc: float = 0.0
    prefix_match: float = 0.0
    skeleton_match: bool = False
    table_set_match: bool = False
    column_precision: float = 0.0
    column_recall: float = 0.0
    column_f1: float = 0.0
    clauses: dict[str, bool] = field(default_factory=dict)
    failure: str = ""
    pred_tables: list[str] = field(default_factory=list)
    gold_tables: list[str] = field(default_factory=list)
    hallucinated_tables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_acc": self.token_acc,
            "prefix_match": self.prefix_match,
            "skeleton_match": self.skeleton_match,
            "table_set_match": self.table_set_match,
            "column_precision": self.column_precision,
            "column_recall": self.column_recall,
            "column_f1": self.column_f1,
            "clauses": dict(self.clauses),
            "failure": self.failure,
        }


def _aggregate_names(sql: str) -> set[str]:
    """Aggregate function names actually used in ``sql`` (lowercased).

    ``SqlAnalysis.has_aggregate`` is a single boolean, so it cannot tell
    ``COUNT`` from ``SUM``. Comparing the *names* lets the taxonomy distinguish
    "no aggregate at all" from "wrong aggregate", which are different mistakes.
    """
    return {t.value.lower() for t in tokenize(sql or "")
            if t.kind in ("ident", "keyword")
            and t.value.lower() in {"count", "sum", "avg", "min", "max"}}


def classify_partial_failure(pred: Prediction, pred_an: SqlAnalysis | None,
                             gold_an: SqlAnalysis | None,
                             pred_sql: str,
                             gold_sql: str = "") -> str:
    """Most specific actionable reason the prediction is not an exec match.

    Ordered from "did not produce SQL at all" outward to "produced valid but
    different results", because that ordering is also the order in which a data
    engineer would fix them. Schema errors outrank structural ones: a
    hallucinated column is a *harder* failure than a missing JOIN, and it is the
    one the hard-negative synthesis stage is designed to fix.
    """
    if pred.exec_match:
        return ""
    if not (pred_sql or "").strip():
        return "unparseable"
    if pred_an is not None and not pred_an.ok:
        # schema-level errors are more actionable than generic "not executable"
        codes = {i.code for i in pred_an.errors}
        if "unknown_table" in codes or "unknown_table_qualifier" in codes:
            return "wrong_table"
        if "unknown_column" in codes:
            return "wrong_column"
        if "not_select" in codes:
            return "unparseable"
    if not pred.executable:
        return "not_executable"
    if pred_an is not None and gold_an is not None:
        if gold_an.has_join and not pred_an.has_join:
            return "missing_join"
        if pred_an.has_join and not gold_an.has_join:
            return "extra_join"
        pred_aggs = _aggregate_names(pred_sql)
        gold_aggs = _aggregate_names(gold_sql)
        if gold_aggs and not pred_aggs:
            return "wrong_aggregate"
        if pred_aggs and gold_aggs and pred_aggs != gold_aggs:
            return "wrong_aggregate"
        if gold_an.has_group_by and not pred_an.has_group_by:
            return "missing_group_by"
        if gold_an.has_order_by and not pred_an.has_order_by:
            return "missing_order_by"
    return "wrong_values"


def score_prediction(pred: Prediction, db: Database | None = None) -> PartialScore:
    """Compute every partial-credit metric for one prediction.

    ``db`` enables schema-aware analysis (unknown-table/column detection). When it
    is unavailable the metrics still compute from the SQL text alone, minus the
    schema-error part of the failure taxonomy.
    """
    pred_sql = pred.pred_sql or ""
    gold_sql = pred.gold_sql or ""
    score = PartialScore(
        token_acc=round(token_accuracy(pred_sql, gold_sql), 6),
        prefix_match=round(prefix_match_ratio(pred_sql, gold_sql), 6),
    )
    pred_an = analyze_sql(pred_sql, db) if pred_sql else None
    gold_an = analyze_sql(gold_sql, db) if gold_sql else None

    if pred_sql and gold_sql:
        score.skeleton_match = sql_skeleton(pred_sql) == sql_skeleton(gold_sql)
    if pred_an is not None and gold_an is not None:
        score.pred_tables = sorted(referenced_tables(pred_sql))
        score.gold_tables = sorted(_table_set(gold_an))
        score.table_set_match = _table_set(pred_an) == _table_set(gold_an)
        p, r, f = column_f1(pred_an, gold_an)
        score.column_precision, score.column_recall, score.column_f1 = p, r, f
        score.clauses = clause_match(pred_an, gold_an)
    elif pred_sql:
        score.pred_tables = sorted(referenced_tables(pred_sql))
    score.hallucinated_tables = sorted(hallucinated_tables(pred_sql, db))
    score.failure = classify_partial_failure(pred, pred_an, gold_an, pred_sql,
                                             gold_sql)
    return score


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def aggregate_partial(scores: Sequence[PartialScore]) -> dict[str, Any]:
    """Aggregate partial-credit scores into a JSON-serializable summary.

    Keys are stable and always present so the report can render a table without
    branching on whether a metric was computable.
    """
    n = len(scores)
    clause_rates: dict[str, float] = {}
    for flag in CLAUSE_FLAGS:
        vals = [1.0 if s.clauses.get(flag) else 0.0 for s in scores if s.clauses]
        clause_rates[flag] = _mean(vals)

    failure_counts: dict[str, int] = {k: 0 for k in PARTIAL_FAILURE_TAXONOMY}
    for s in scores:
        if s.failure:
            failure_counts[s.failure] = failure_counts.get(s.failure, 0) + 1

    return {
        "n": n,
        "token_acc": _mean([s.token_acc for s in scores]),
        "prefix_match": _mean([s.prefix_match for s in scores]),
        "skeleton_match": _mean([1.0 if s.skeleton_match else 0.0 for s in scores]),
        "table_set_match": _mean([1.0 if s.table_set_match else 0.0 for s in scores]),
        "column_precision": _mean([s.column_precision for s in scores]),
        "column_recall": _mean([s.column_recall for s in scores]),
        "column_f1": _mean([s.column_f1 for s in scores]),
        "clause_match": clause_rates,
        "clause_match_mean": _mean(list(clause_rates.values())),
        "failure_taxonomy": {
            k: {"n": v, "rate": round(v / n, 6) if n else 0.0}
            for k, v in failure_counts.items()
        },
        "examples_with_table_hallucination": sum(
            1 for s in scores if s.hallucinated_tables),
    }


def partial_metric_values(scores: Sequence[PartialScore],
                          metric: str) -> list[float]:
    """Per-example 0/1 (or continuous) values for one metric, for bootstrap CIs."""
    out: list[float] = []
    for s in scores:
        if metric == "token_acc":
            out.append(s.token_acc)
        elif metric == "prefix_match":
            out.append(s.prefix_match)
        elif metric == "skeleton_match":
            out.append(1.0 if s.skeleton_match else 0.0)
        elif metric == "table_set_match":
            out.append(1.0 if s.table_set_match else 0.0)
        elif metric == "column_f1":
            out.append(s.column_f1)
        elif metric == "clause_match_mean":
            out.append(_mean([1.0 if v else 0.0 for v in s.clauses.values()]))
        else:
            raise KeyError(f"unknown partial metric {metric!r}")
    return out
