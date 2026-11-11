"""The offline teacher's content generators.

Why this is not a toy
---------------------
The demo run has no network and no API key, so :class:`fcdata.providers.mock.
MockProvider` *is* the teacher that produces the entire synthesis corpus. A mock
that echoed the question back would make the downstream learning curve
meaningless. So every generator here is **grounded**:

* the schema is recovered from the ``CREATE TABLE`` block embedded in the prompt
  (see :mod:`fcdata.providers.sqlread`), never from a hard-coded table list;
* the SQL is parsed into clause facts and the explanation names the *actual*
  tables, columns, operators, literals and aggregates in that statement;
* paraphrase rewrites are validated against the original for entity/number
  preservation and fall back deterministically instead of drifting semantically;
* the judge re-runs the *real* schema validator (:func:`fcdata.sqlutils.analyze_sql`)
  and compares structural facts against the reference query, so its score is
  calibrated rather than random.

Determinism
-----------
Each writer receives a :class:`random.Random` seeded by the provider from
``sha256(task | sample_id | prompt)``. Same prompt in, same text out, byte for
byte -- which is what makes the "run the pipeline twice, diff the artifacts"
check meaningful. Randomness is only ever used to pick between *equivalent*
phrasings, never to decide *content*.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..sqlutils import analyze_sql, normalize_sql
from .prompts import (CONSISTENT_AT, INCONSISTENT_BELOW, verdict_for_score)
from .sqlread import (DdlSchema, Ref, SqlFacts, iter_refs, read_intent,
                      read_sql)

# --------------------------------------------------------------------------- #
# shared phrasing helpers
# --------------------------------------------------------------------------- #
_AGG_WORDS = {"count": "count", "sum": "total", "avg": "average",
              "min": "minimum", "max": "maximum"}


def _q(name: str) -> str:
    """Spell an identifier the way a prompt/schema would (no quoting games)."""
    return name


def _col(ref: Ref) -> str:
    return f"{ref.table}.{ref.column}"


def _cols(refs: Iterable[Ref], limit: int = 4) -> str:
    names = [_col(r) for r in refs][:limit]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _table_list(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _one_of(rng: random.Random, options: Sequence[str]) -> str:
    """Deterministically pick one phrasing variant."""
    return options[rng.randrange(len(options))]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _clean_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";")


# --------------------------------------------------------------------------- #
# CoT writer
# --------------------------------------------------------------------------- #
@dataclass
class _Step:
    text: str


def _scope_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    tables = facts.tables
    picked = [r for item in facts.select_items for r in item.refs]
    lead = _one_of(rng, [
        "Start from the table that holds the records the question asks about",
        "The question is answered from the following table",
        "First locate the data the answer is built from",
    ])
    if len(tables) > 1:
        lead = _one_of(rng, [
            "The question spans several tables",
            "This question cannot be answered from a single table",
            "Several tables have to be combined to answer the question",
        ])
    if picked:
        types = ", ".join(f"{_col(r)} ({r.type})" for r in picked[:3])
        return f"{lead}: {_table_list(tables)}. The columns that appear in the output are {types}."
    return f"{lead}: {_table_list(tables)}."


def _join_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    if not facts.joins:
        return ""
    pieces: list[str] = []
    for join in facts.joins:
        if join.pairs:
            left, right = join.pairs[0]
            cond = f"{_col(left)} = {_col(right)}"
        else:
            cond = join.on or "the join condition"
        if join.kind == "left":
            extra = (f"the left join keeps every row of {join.table} even when no "
                     f"matching row exists on the other side")
        elif join.kind == "cross":
            extra = (f"the cross join pairs every row of {join.table} with every row "
                     f"already in scope")
        else:
            extra = (f"the inner join keeps only rows that find a match on both sides, "
                     f"so the two tables are aligned row by row")
        pieces.append((join.table, cond, extra))
    table, cond, extra = pieces[0]
    lead = _one_of(rng, [
        f"Join {table} on {cond}",
        f"Bring in {table} by matching {cond}",
        f"Relate {table} to the tables already in scope through {cond}",
    ])
    text = f"{lead}; {extra}"
    for table, cond, extra in pieces[1:]:
        text += f". Then join {table} on {cond}, where {extra}"
    return text + "."


def _filter_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    if not facts.where:
        return ""
    parts: list[str] = []
    for idx, pred in enumerate(facts.where):
        connector = pred.connector or "and"
        if idx == 0:
            clause = _describe_predicate(rng, pred)
        else:
            clause = f"{connector} {_describe_predicate(rng, pred)}"
        parts.append(clause)
    if facts.joins:
        lead = _one_of(rng, [
            "Restrict the joined rows to those that satisfy the conditions",
            "Filter the joined rows",
            "Keep only the joined rows the question actually asks about",
        ])
    else:
        lead = _one_of(rng, [
            "Restrict the rows to those that satisfy the conditions",
            "Filter the rows",
            "Keep only the rows the question actually asks about",
        ])
    return f"{lead}: " + " ".join(parts) + "."


def _describe_predicate(rng: random.Random, pred: Any) -> str:
    left = _clean_predicate_left(pred.left)
    right = pred.right.strip()
    if pred.op in ("isnull", "notnull"):
        return f"{left} is {'not ' if pred.op == 'notnull' else ''}null"
    if pred.op in ("in", "not in") and pred.has_subquery:
        inner = re.sub(r"\s+", " ", right).strip()
        inner = inner.strip()
        # strip the outer parens so the inner SELECT is readable
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1].strip()
        m = re.search(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)", inner, re.IGNORECASE)
        sel = re.search(r"\bselect\s+(.+?)\s+from\b", inner, re.IGNORECASE | re.DOTALL)
        source = m.group(1) if m else "the subquery"
        col = sel.group(1).strip() if sel else "the subquery's column"
        neg = "not among" if pred.op == "not in" else "among"
        return (f"{left} is {neg} the {col} values produced by the subquery over "
                f"{source}")
    if pred.op == "between":
        return f"{left} falls between {right}"
    if pred.op in ("like", "not like"):
        return f"{left} {'does not match' if pred.op == 'not like' else 'matches'} {right}"
    if pred.op:
        op_word = {
            "=": "equals", "<>": "differs from", ">": "is greater than",
            ">=": "is at least", "<": "is less than", "<=": "is at most",
        }.get(pred.op, f"compares with {pred.op}")
        if pred.op == "=" and re.match(r"^'", right):
            op_word = "equals the literal"
        return f"{left} {op_word} {right}"
    return _clean_predicate_left(pred.raw)


def _clean_predicate_left(left: str) -> str:
    return re.sub(r"\s+", " ", left.strip())


def _aggregate_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    """Explain grouping and every aggregate, including those only in ORDER BY/HAVING."""
    aggs = [(i.aggregate, i.expr) for i in facts.select_items if i.aggregate]
    for term in facts.order_by:
        if term.aggregate:
            aggs.append((term.aggregate, term.expr))
    if facts.having:
        for m in re.finditer(r"\b(count|sum|avg|min|max)\s*\(([^()]*(?:\([^()]*\))?[^()]*)\)",
                             facts.having, re.IGNORECASE):
            aggs.append((m.group(1).lower(), m.group(0)))
    seen: set[tuple[str, str]] = set()
    unique_aggs: list[tuple[str, str]] = []
    for agg, expr in aggs:
        key = (agg, normalize_sql(expr))
        if key not in seen:
            seen.add(key)
            unique_aggs.append((agg, expr))

    if facts.group_by:
        groups = ", ".join(_clean_sql(g) for g in facts.group_by)
        text = (f"Group the surviving rows by {groups}, so each output row corresponds "
                f"to one distinct {groups}.")
        if unique_aggs:
            text += " Within each group, " + " and ".join(
                _describe_aggregate(agg, expr) for agg, expr in unique_aggs[:3]) + "."
        return text
    if unique_aggs:
        return ("Collapse all surviving rows into a single result row by "
                + " and ".join(_describe_aggregate(agg, expr, gerund=True)
                               for agg, expr in unique_aggs[:3]) + ".")
    if facts.select_distinct:
        return ("Apply DISTINCT to the selected columns so that repeated "
                "combinations collapse into one row each.")
    if facts.set_ops:
        return (f"Combine the two SELECT blocks with {facts.set_ops[0].upper()}, which "
                f"keeps only rows satisfying the set operation.")
    return ""


def _describe_aggregate(agg: str, expr: str, gerund: bool = False) -> str:
    m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", expr, re.DOTALL)
    target = m.group(2).strip() if m else "the rows"
    if target == "*":
        target = "the surviving rows"
    word = _AGG_WORDS.get(agg, agg)
    if gerund:
        if agg == "count":
            return f"counting {target}"
        return f"taking the {word} of {target}"
    if agg == "count":
        return f"count {target}"
    return f"take the {word} of {target}"


def _having_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    if not facts.having:
        return ""
    return (f"Keep only the groups that pass the aggregate condition "
            f"{_clean_sql(facts.having)}.")


def _order_step(rng: random.Random, facts: SqlFacts, schema: DdlSchema) -> str:
    if not facts.order_by:
        if facts.limit is not None:
            return f"Keep only the first {facts.limit} row(s) of the result."
        return ""
    terms = []
    for t in facts.order_by:
        direction = "descending" if t.descending else "ascending"
        terms.append(f"{_clean_sql(t.expr)} in {direction} order")
    text = "Sort the result by " + " then ".join(terms)
    if facts.limit is not None:
        if facts.limit == 1 and any(t.descending for t in facts.order_by):
            text += (", and keep the single top row (LIMIT 1) so only the extreme "
                     "value survives")
        else:
            text += f", and keep only the first {facts.limit} row(s)"
    return text + "."


def write_cot(rng: random.Random, *, schema: DdlSchema, question: str,
              gold_sql: str, facts: SqlFacts) -> str:
    """Emit a numbered reasoning trace grounded in ``facts``."""
    builders = (
        lambda: _scope_step(rng, facts, schema),
        lambda: _join_step(rng, facts, schema),
        lambda: _filter_step(rng, facts, schema),
        lambda: _aggregate_step(rng, facts, schema),
        lambda: _having_step(rng, facts, schema),
        lambda: _order_step(rng, facts, schema),
    )
    steps = [s for s in (b() for b in builders) if s]
    if not steps:
        # last-resort but still grounded: name the shape of the query
        steps = [f"The query reads from {_table_list(facts.tables)} and returns "
                 f"{facts.n_select_items} column(s)."]
    # keep the trace to 3..6 steps by dropping the least informative middle steps
    if len(steps) > 6:
        steps = steps[:3] + steps[-2:]
    lines = [f"{i}. {s}" for i, s in enumerate(steps, start=1)]
    lines.append(f"Final SQL: {_clean_sql(gold_sql)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# paraphrase writer
# --------------------------------------------------------------------------- #
_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("show", ("list", "display", "return")),
    ("list", ("show", "display")),
    ("display", ("show", "list")),
    ("find", ("identify", "locate")),
    ("get", ("retrieve", "obtain")),
    ("give", ("provide", "return")),
    ("how many", ("the number of",)),
    ("average", ("mean",)),
    ("mean", ("average",)),
    ("total", ("overall",)),
    ("highest", ("greatest", "maximum")),
    ("lowest", ("smallest", "minimum")),
    ("more than", ("greater than",)),
    ("less than", ("lower than",)),
    ("each", ("every",)),
    ("together with", ("along with",)),
    ("that are", ("which are",)),
    ("that is", ("which is",)),
    ("who have", ("that have",)),
    ("ordered from highest to lowest", ("sorted in descending order",)),
    ("ordered from lowest to highest", ("sorted in ascending order",)),
    ("what is", ("what's",)),
    ("what are", ("what're",)),
)

_FILLERS: tuple[tuple[str, str], ...] = (
    (r"^\s*(?:please|hey|hi)\s*,?\s*", ""),
    (r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?", ""),
    (r"^\s*(?:i|we)\s+(?:want|need|would like)\s+to\s+know\s+", ""),
    (r"^\s*tell\s+me\s+", ""),
    (r"\s+in\s+total\s*", " "),
    (r"\s+for\s+me\s*", " "),
    (r"\ball\s+of\s+the\b", "all the"),
    (r"\s+please\s*", " "),
)

_NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)")
_QUOTED_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _numbers(text: str) -> list[str]:
    return sorted(_NUM_RE.findall(text or ""))


def _quoted(text: str) -> list[str]:
    out: list[str] = []
    for m in _QUOTED_RE.finditer(text or ""):
        out.append((m.group(1) if m.group(1) is not None else m.group(2)) or "")
    return sorted(out)


def _title_terms(text: str) -> list[str]:
    """Capitalised multi-letter words: usually entity values that must survive."""
    return sorted({w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text or "")
                   if w.lower() not in {"the", "a", "an", "show", "list", "what",
                                        "which", "who", "how", "find", "give", "get"}})


def paraphrase_preserves(original: str, rewritten: str) -> bool:
    """Guard rail: a rewrite may change style, never facts.

    Rejects rewrites that drop/add numbers, drop a capitalised entity value,
    change quoted literals, or drift in length by more than 2x in either
    direction. This is the difference between a paraphrase and a hallucination.
    """
    if not rewritten or not rewritten.strip():
        return False
    if normalize_sql(rewritten) == normalize_sql(original):
        return False
    # A rewrite must keep quoting balanced: an odd number of apostrophes means an
    # entity value lost its closing quote, which would change what the model has
    # to copy into the SQL literal.
    if original.count("'") % 2 == 0 and rewritten.count("'") % 2 != 0:
        return False
    if _numbers(original) != _numbers(rewritten):
        return False
    if _quoted(original) != _quoted(rewritten):
        return False
    missing = [t for t in _title_terms(original)
               if t.lower() not in rewritten.lower()]
    if missing:
        return False
    a, b = len(original.split()), len(rewritten.split())
    return a * 0.4 <= b <= a * 2.2


def _apply_synonyms(rng: random.Random, question: str) -> str:
    text = question
    # pick at most two substitutions so the sentence still reads naturally
    candidates = [(a, b) for a, opts in _SYNONYMS
                  if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text, re.IGNORECASE)
                  for b in opts]
    rng.shuffle(candidates)
    applied = 0
    for word, repl in candidates:
        if applied >= 2:
            break
        pattern = re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            continue
        # preserve the original capitalisation of the first letter
        new = repl
        if m.group(0)[:1].isupper():
            new = _capitalize(repl)
        text = text[:m.start()] + new + text[m.end():]
        applied += 1
    return text


_IMPERATIVE_RE = re.compile(
    r"^(show|list|display|give|return|find|get|identify|locate|retrieve|count|"
    r"provide|obtain|compute|calculate)\b", re.IGNORECASE)


def _apply_politeness(rng: random.Random, question: str) -> str:
    """Add a conversational frame without touching the information content."""
    body = question.strip().rstrip("?.")
    if _IMPERATIVE_RE.match(body):
        lead = _one_of(rng, ["Could you ", "Can you please ", "Would you "])
        rest = body[:1].lower() + body[1:]
        return f"{lead}{rest}?"
    lead = _one_of(rng, ["Can you tell me ", "I would like to know ",
                         "Could you tell me "])
    return f"{lead}{body[:1].lower() + body[1:]}?"


def _apply_count_form(rng: random.Random, question: str) -> str:
    """``how many X`` <-> ``count the X`` <-> ``the number of X``."""
    m = re.match(r"^how\s+many\s+(.*?)\s*([?.])?$", question, re.IGNORECASE | re.DOTALL)
    if m:
        rest = m.group(1).strip().rstrip("?.")
        return _one_of(rng, [f"Count the {rest}.", f"What is the number of {rest}?",
                             f"Return the number of {rest}."])
    m = re.match(r"^(?:what\s+is\s+|return\s+|give\s+|show\s+)?the\s+number\s+of\s+"
                 r"(.*?)\s*([?.])?$", question, re.IGNORECASE | re.DOTALL)
    if m:
        rest = m.group(1).strip().rstrip("?.")
        return _one_of(rng, [f"How many {rest} are there?",
                             f"How many {rest}?",
                             f"What is the count of {rest}?"])
    m = re.match(r"^(?:count|return the count of|give the count of)\s+(?:the\s+)?"
                 r"(.*?)\s*([?.])?$", question, re.IGNORECASE | re.DOTALL)
    if m:
        rest = m.group(1).strip().rstrip("?.")
        return _one_of(rng, [f"How many {rest} are there?",
                             f"What is the number of {rest}?"])
    return ""


_TAIL_PREP_RE = re.compile(
    r"\s+(in|with|for|from|on|at|within)\s+([^,;]{2,60}?)\s*([?.]?)\s*$",
    re.IGNORECASE,
)
_TAIL_BAN = re.compile(
    r"\b(ordered|sorted|grouped|descending|ascending|highest|lowest|limit|top|"
    r"more than|less than|greater than|fewer than)\b",
    re.IGNORECASE)
_FRONT_MODIFIER_RE = re.compile(
    r"^(show|list|display|give|return|find|get)\s+(?:me\s+)?(?:the\s+|all\s+)?"
    r"(.+?)\s+(who|that|which|whose)\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _apply_reorder(question: str) -> str:
    """Restructure word order without changing scope.

    Two shapes are handled: a short trailing prepositional modifier is hoisted to
    the front (``Show X in Y`` -> ``In Y, show X``), and a relative clause is
    turned into a wh-question with the same restrictor (``Show X that Y`` ->
    ``Which X are Y?``).
    """
    m = _FRONT_MODIFIER_RE.match(question.strip().rstrip("?."))
    if m:
        head, subject, rel, clause = (m.group(1).lower(), m.group(2).strip(),
                                      m.group(3).lower(), m.group(4).strip())
        if not _TAIL_BAN.search(clause) and len(clause.split()) <= 14:
            verb = "are" if subject.split()[-1:][0].endswith("s") else "is"
            wh = "Which" if rel in ("who", "that", "which") else "What"
            return f"{wh} {subject} {verb} the ones {rel} {clause}?"
    m = _TAIL_PREP_RE.search(question)
    if not m:
        return ""
    prep, tail = m.group(1), m.group(2).strip()
    if _TAIL_BAN.search(tail) or len(tail.split()) > 8:
        return ""
    head = question[:m.start()].strip()
    if not head or len(head.split()) < 3:
        return ""
    head = head.rstrip(",").strip()
    body = f"{_capitalize(prep)} {tail}, {head[:1].lower() + head[1:]}"
    # keep the sentence mood: an imperative stays imperative, a question stays one
    body = body.rstrip("?.")
    return body + "." if _IMPERATIVE_RE.match(head) else body + "?"


_NP_BREAK_RE = re.compile(r"\b(of|for|in|with|that|who|which|whose|from|on)\b",
                          re.IGNORECASE)


def _is_plural_subject(rest: str) -> bool:
    """Number agreement for the wh-question rewrite: look at the head noun.

    ``the names of passengers`` -> *names* (plural); ``the artist with the
    highest attendance`` -> *artist* (singular). The head noun is the last word of
    the noun phrase that precedes the first relational preposition or relative
    pronoun, which is precise enough for Spider-style questions.
    """
    np_text = _NP_BREAK_RE.split(rest, maxsplit=1)[0]
    words = [re.sub(r"[^A-Za-z']", "", w) for w in np_text.split()]
    words = [w for w in words if w and w.lower() not in
             {"the", "a", "an", "all", "each", "every", "its", "their", "his", "her"}]
    if not words:
        return False
    head = words[-1].lower()
    if head.endswith("s") and not head.endswith(("ss", "us", "is")):
        return True
    return False


def _apply_clause(question: str) -> str:
    m = re.match(r"^(show|list|display|give|return|find|get)\s+(?:me\s+)?(.*)$",
                 question, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    rest = m.group(2).strip().rstrip("?.").strip()
    if not rest:
        return ""
    verb = "are" if _is_plural_subject(rest) else "is"
    return f"What {verb} {rest}?"


def _apply_compression(rng: random.Random, question: str) -> str:
    text = question
    for pattern, repl in _FILLERS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([?.])", r"\1", text)
    if text and text[0].islower():
        text = _capitalize(text)
    return text


_STRATEGY_FUNCS: dict[str, Any] = {
    "synonym": _apply_synonyms,
    "politeness": _apply_politeness,
    "count_form": _apply_count_form,
    "reorder": lambda rng, q: _apply_reorder(q),
    "clause": lambda rng, q: _apply_clause(q),
    "compression": _apply_compression,
}


def write_paraphrase(rng: random.Random, *, question: str, strategy: str) -> str:
    """Rewrite ``question`` with ``strategy``; always returns a *valid* paraphrase.

    The requested strategy is tried first, then every other strategy, and only a
    rewrite that passes :func:`paraphrase_preserves` is returned. The final
    fallback is a purely mechanical politeness frame, which cannot alter content
    because it only prepends words -- so this function never emits a rewrite that
    changes the question's meaning.
    """
    original = _clean_sql(question).rstrip(".")
    rest = [s for s in _STRATEGY_FUNCS if s != strategy]
    # Rotate the fallback order with the seeded RNG: two slots on the same source
    # often need a fallback, and a fixed order would make them converge on the
    # identical sentence (which the dataset layer would then have to discard).
    rng.shuffle(rest)
    for name in [strategy] + rest:
        fn = _STRATEGY_FUNCS.get(name)
        if fn is None:
            continue
        candidate = _clean_sql(fn(rng, original))
        if candidate and paraphrase_preserves(original, candidate):
            return candidate
    # Guaranteed-safe last resort: a frame that only adds words.
    body = original.rstrip("?.")
    if _IMPERATIVE_RE.match(body):
        return f"Could you {body[:1].lower() + body[1:]}?"
    return f"Can you tell me {body[:1].lower() + body[1:]}?"


# --------------------------------------------------------------------------- #
# correction-note writer
# --------------------------------------------------------------------------- #
_OP_PHRASE = {
    ">=": "at least", ">": "strictly greater than", "<=": "at most",
    "<": "strictly less than", "=": "equal to", "<>": "different from",
}


@dataclass
class _Diff:
    category: str
    wrong: str
    right: str


def _first_difference(gold: SqlFacts, bad: SqlFacts, schema: DdlSchema) -> _Diff:
    """Find the most salient structural difference between gold and bad SQL."""
    if gold.tables != bad.tables:
        added = [t for t in bad.tables if t not in gold.tables]
        removed = [t for t in gold.tables if t not in bad.tables]
        wrong = added[0] if added else "the wrong table"
        right = removed[0] if removed else gold.tables[0] if gold.tables else "the correct table"
        return _Diff("table", wrong, right)
    if gold.aggregates != bad.aggregates:
        return _Diff("aggregate", bad.aggregates[0] if bad.aggregates else "no aggregate",
                     gold.aggregates[0] if gold.aggregates else "no aggregate")
    gold_lits = _literals(gold)
    bad_lits = _literals(bad)
    if gold_lits != bad_lits:
        return _Diff("literal", bad_lits[0] if bad_lits else "no value",
                     gold_lits[0] if gold_lits else "no value")
    if bool(gold.group_by) != bool(bad.group_by):
        return _Diff("group_by", "grouping" if bad.group_by else "no grouping",
                     "grouping" if gold.group_by else "no grouping")
    gold_ops = [(p.op, p.left) for p in gold.where]
    bad_ops = [(p.op, p.left) for p in bad.where]
    if gold_ops != bad_ops:
        gop = next((p for p in gold.where if p.op), None)
        bop = next((p for p in bad.where if p.op), None)
        if gop and bop and gop.op != bop.op:
            return _Diff("operator", f"{bop.left} {bop.op} {bop.right}",
                         f"{gop.left} {gop.op} {gop.right}")
        if bop and gop:
            return _Diff("filter", bop.left or bop.raw, gop.left or gop.raw)
    gold_cols = sorted({_col(r) for r in iter_refs(gold)})
    bad_cols = sorted({_col(r) for r in iter_refs(bad)})
    if gold_cols != bad_cols:
        extra = [c for c in bad_cols if c not in gold_cols]
        missing = [c for c in gold_cols if c not in bad_cols]
        return _Diff("column", extra[0] if extra else "an extra column",
                     missing[0] if missing else "the intended column")
    if [t.descending for t in gold.order_by] != [t.descending for t in bad.order_by]:
        return _Diff("order", "the opposite sort direction", "the requested sort direction")
    if gold.select_distinct != bad.select_distinct:
        return _Diff("distinct", "no DISTINCT" if not bad.select_distinct else "DISTINCT",
                     "DISTINCT" if gold.select_distinct else "no DISTINCT")
    return _Diff("shape", "a different query shape", "the intended query shape")


def _literals(facts: SqlFacts) -> list[str]:
    out: list[str] = []
    for pred in facts.where:
        out.extend(pred.literals)
    return out


def write_correction(rng: random.Random, *, schema: DdlSchema, question: str,
                     gold_sql: str, bad_sql: str, error_kind: str) -> str:
    """Explain *why* ``bad_sql`` is wrong, naming real identifiers."""
    gold = read_sql(gold_sql, schema)
    bad = read_sql(bad_sql, schema)
    diff = _first_difference(gold, bad, schema)
    if error_kind == "missing_join" and diff.category in ("table", "column", "shape"):
        return (f"The query reads only {_table_list(bad.tables) or 'one table'} and never "
                f"joins the related table, so it cannot see the rows needed to answer the "
                f"question; it must join {_table_list(gold.tables)} on their shared key "
                f"before filtering.")
    if error_kind == "missing_group_by" and diff.category in ("group_by", "shape"):
        group_cols = _cols(gold.group_refs)
        return (f"The query aggregates over the whole table without grouping, so it "
                f"returns one row instead of one row per {group_cols or 'group'}; a "
                f"GROUP BY {group_cols or 'the grouping column'} is required.")
    if error_kind == "missing_distinct" and diff.category in ("distinct", "shape"):
        return ("The query omits DISTINCT, so duplicate answers are repeated in the "
                "output; the question asks for the distinct values only.")
    if diff.category == "table":
        return (f"The query reads from {diff.wrong} instead of {diff.right}; that table "
                f"holds the wrong entity, so the selected rows answer a different "
                f"question. It must use {diff.right}.")
    if diff.category == "aggregate":
        return (f"The query computes {diff.wrong.upper()} where {diff.right.upper()} is "
                f"required, so the returned number measures the wrong thing even though "
                f"the rows are filtered correctly.")
    if diff.category == "literal":
        return (f"The query compares against {diff.wrong} instead of {diff.right}; the "
                f"filter therefore keeps a different set of rows than the question "
                f"describes.")
    if diff.category == "operator":
        return (f"The predicate is {diff.wrong}, which {'excludes' if '<' in diff.right else 'includes'} "
                f"the boundary rows the question asks about; it should be {diff.right}.")
    if diff.category == "column":
        return (f"The query returns {diff.wrong} instead of {diff.right}, so the output "
                f"column carries the wrong attribute of the same row.")
    if diff.category == "order":
        return ("The query sorts in the opposite direction to the one the question "
                "asks for, so the first row is the worst match rather than the best.")
    if diff.category == "group_by":
        return ("The query groups the rows when the question asks for a single "
                "aggregate over all of them, so it returns many rows instead of one.")
    return (f"The query is structurally different from the one the question requires: "
            f"it computes {diff.wrong} rather than {diff.right}.")


# --------------------------------------------------------------------------- #
# judge writer
# --------------------------------------------------------------------------- #
@dataclass
class JudgeOutcome:
    score: float
    verdict: str
    rationale: str

    def as_json(self) -> str:
        return json.dumps({"score": round(self.score, 2), "verdict": self.verdict,
                           "rationale": self.rationale}, ensure_ascii=False)


#: score thresholds for the three-way verdict. ``verdict_for_score`` (imported from
#: the prompt contract module) is the *only* source of truth for ``verdict``: every
#: judge path derives the label from the score, so the two fields can never
#: disagree in an emitted JSON record.
_ = (CONSISTENT_AT, INCONSISTENT_BELOW)      # re-exported for callers/tests


def _verdict(score: float) -> str:
    return verdict_for_score(score)


def _schema_error_rationale(analysis: Any) -> str:
    err = analysis.errors[0] if analysis.errors else None
    if err is None:
        return "the query could not be validated against the schema"
    if err.code == "unknown_column":
        return f"the query references a column that does not exist in the schema ({err.detail})"
    if err.code == "unknown_table":
        return f"the query references a table that does not exist in the schema ({err.detail})"
    if err.code == "unknown_table_qualifier":
        return f"the query uses a table qualifier that is not in scope ({err.detail})"
    if err.code == "no_from_clause":
        return "the query has no FROM clause, so it does not read the schema at all"
    if err.code == "destructive_statement":
        return "the query is not a read-only SELECT statement"
    return f"the query fails schema validation ({err.code}: {err.detail})"


def _structural_score(question: str, gold: SqlFacts, cand: SqlFacts,
                      analysis: Any) -> JudgeOutcome:
    """Compare a candidate against a reference query, clause by clause."""
    if not analysis.ok:
        return JudgeOutcome(0.08, "inconsistent", _schema_error_rationale(analysis))

    score = 1.0
    notes: list[str] = []

    if gold.tables != cand.tables:
        score -= 0.55
        missing = [t for t in gold.tables if t not in cand.tables]
        extra = [t for t in cand.tables if t not in gold.tables]
        bits = []
        if missing:
            bits.append(f"it never reads {_table_list(missing)}")
        if extra:
            bits.append(f"it reads unrelated {_table_list(extra)}")
        notes.append("the table scope is wrong: " + " and ".join(bits))
    if bool(gold.joins) != bool(cand.joins):
        score -= 0.18
        notes.append("the join structure differs from the required one")
    if gold.aggregates != cand.aggregates:
        score -= 0.22
        notes.append(f"it computes {cand.aggregates or 'no aggregate'} where "
                     f"{gold.aggregates or 'no aggregate'} is required")
    g_lits, c_lits = _literals(gold), _literals(cand)
    if g_lits != c_lits:
        score -= 0.20
        notes.append(f"the literal filter differs ({c_lits or 'none'} vs "
                     f"{g_lits or 'none'})")
    g_ops = [(p.op, normalize_sql(p.left)) for p in gold.where]
    c_ops = [(p.op, normalize_sql(p.left)) for p in cand.where]
    if g_ops != c_ops:
        score -= 0.15
        notes.append("the filter predicates differ")
    if bool(gold.group_by) != bool(cand.group_by):
        score -= 0.15
        notes.append("grouping is missing or spurious")
    elif gold.group_refs and cand.group_refs:
        if sorted(r.qualified for r in gold.group_refs) != sorted(
                r.qualified for r in cand.group_refs):
            score -= 0.10
            notes.append("the grouping key differs")
    if bool(gold.order_by) != bool(cand.order_by):
        score -= 0.10
        notes.append("ordering is missing or unrequested")
    elif [t.descending for t in gold.order_by] != [t.descending for t in cand.order_by]:
        score -= 0.12
        notes.append("the sort direction is reversed")
    if gold.limit != cand.limit:
        score -= 0.08
        notes.append(f"the row limit differs ({cand.limit} vs {gold.limit})")
    if gold.select_distinct != cand.select_distinct:
        score -= 0.06
        notes.append("DISTINCT is missing or unrequested")
    if normalize_sql(gold.sql) == normalize_sql(cand.sql):
        return JudgeOutcome(1.0, "consistent",
                            "Every clause matches the reference: the same tables, the "
                            "same filters, the same computation and the same ordering.")
    score = max(0.05, min(0.98, score))
    if score >= 0.75:
        rationale = ("The query is equivalent to the reference in substance; the only "
                     "difference is " + (notes[0] if notes else "cosmetic"))
    else:
        rationale = _capitalize("; ".join(notes[:2]) if notes else
                                "the query differs from the reference")
        rationale += "."
    return JudgeOutcome(round(score, 4), _verdict(score), rationale)


def _intent_score(question: str, cand: SqlFacts, analysis: Any) -> JudgeOutcome:
    """No reference available: grade the candidate against the question itself.

    Calibration matters more here than in the reference path, because there is no
    gold query to anchor against. The rubric is therefore built so that a query
    that is *schema-valid and mentions the entities the question names* lands in
    the ``consistent`` band, while a query that answers a different question lands
    in ``inconsistent`` -- rather than everything clustering at "uncertain", which
    would make the judge look uninformative in the report.

    Anchors: schema-invalid -> ~0.1; valid but contradicts the asked-for
    aggregation -> ~0.3; valid and aligned -> ~0.8.
    """
    if not analysis.ok:
        return JudgeOutcome(0.10, "inconsistent", _schema_error_rationale(analysis))
    intent = read_intent(question)
    # Base: a schema-valid SELECT that reads real tables is already plausible.
    score = 0.62
    notes: list[str] = []
    penalties: list[str] = []

    # -- aggregation intent (the strongest signal) -----------------------
    wants = ("count" if intent.asks_count else
             "avg" if intent.asks_average else
             "sum" if intent.asks_sum else "")
    if wants:
        if wants in cand.aggregates:
            score += 0.16
        else:
            score -= 0.32
            penalties.append(f"the question asks for {_INTENT_WORD[wants]} but the "
                             f"query computes {cand.aggregates or 'no aggregate'}")
    # -- ordering / limit intent ----------------------------------------
    if intent.asks_extreme == "max":
        if any(t.descending for t in cand.order_by):
            score += 0.08
        elif cand.order_by:
            score -= 0.12
            penalties.append("the query sorts in the opposite direction to the "
                             "extreme the question asks for")
    elif intent.asks_extreme == "min":
        if any(not t.descending for t in cand.order_by):
            score += 0.08
        elif cand.order_by:
            score -= 0.12
            penalties.append("the query sorts in the opposite direction to the "
                             "extreme the question asks for")
    if intent.asks_top:
        if cand.limit is not None:
            score += 0.06
        else:
            score -= 0.10
            penalties.append("the question asks for a top/bottom row but the query "
                             "does not limit the result")
    # -- grouping intent -------------------------------------------------
    if intent.asks_grouped:
        if cand.group_by:
            score += 0.10
        else:
            score -= 0.20
            penalties.append("the question asks for a value per group but the query "
                             "does not group the rows")
    elif cand.group_by and not cand.has_aggregate:
        score -= 0.06
        penalties.append("the query groups rows the question does not ask to group")
    # -- schema linking: does the query use the entities the question names?
    q_tokens = {w.lower() for w in re.findall(r"[A-Za-z_]{3,}", question)}
    refs = list(iter_refs(cand))
    if refs:
        hits = sum(1 for r in refs
                   if any(part in q_tokens
                          for part in re.split(r"[_\s]+", r.column.lower())))
        ratio = hits / len(refs)
        score += 0.14 * ratio
        if ratio < 0.34:
            score -= 0.14
            penalties.append("several columns used by the query are not mentioned "
                             "by the question")
    if penalties:
        notes = penalties
    else:
        notes = ["the query's tables, filters and aggregation line up with the "
                 "question"]
    score = max(0.05, min(0.97, score))
    verdict = _verdict(score)
    if verdict == "inconsistent":
        rationale = _capitalize("; ".join(notes[:2])) + "."
    elif verdict == "uncertain":
        rationale = ("Mostly aligned with the question, but "
                     + "; ".join(notes[:2]) + "." if penalties
                     else "Broadly aligned with the question; the mapping is not "
                          "unambiguous.")
    else:
        rationale = _capitalize(notes[0]) + "."
    return JudgeOutcome(round(score, 4), verdict, rationale)


_INTENT_WORD = {"count": "a count", "avg": "an average", "sum": "a total"}


def write_judge(rng: random.Random, *, schema: DdlSchema, question: str,
                candidate_sql: str, gold_sql: str = "") -> str:
    """Emit the calibrated JSON verdict the validator consumes."""
    db = schema.to_database()
    analysis = analyze_sql(candidate_sql, db)
    cand = read_sql(candidate_sql, schema)
    if gold_sql.strip():
        gold = read_sql(gold_sql, schema)
        outcome = _structural_score(question, gold, cand, analysis)
    else:
        outcome = _intent_score(question, cand, analysis)
    return outcome.as_json()
