"""SQL utilities: extraction, tokenizing, schema-level validation, normalization.

Deliberately dependency-free (no sqlglot / sqlparse). Two levels of checking:

1. **Static** -- tokenize the SQL and verify that every table and column
   reference resolves against the Spider schema. This is what catches
   "hallucinated column" errors from a generator model without touching a DB.
2. **Dynamic** -- ``sqlite3`` ``EXPLAIN``/``execute`` against the fixture
   database, which is the real syntax + semantic check (see ``dbexec.py``).

Normalization follows the Spider exact-match convention closely enough that
numbers are comparable across systems, and it is documented so a reader can see
exactly what was (and was not) normalized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schema import Database, SQL_KEYWORDS, normalize_ident

# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL statement out of a model response.

    Handles ```sql fences, bare fences, a leading ``SQL:`` label and prose
    wrappers. Returns ``""`` when nothing SQL-like is found.
    """
    if not text:
        return ""
    t = text.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*(?:sql|SQL|Sql)\s*[:\-]\s*", "", t)
    # drop a leading explanation sentence: keep from the first SELECT/WITH
    m = re.search(r"\b(select|with)\b", t, re.IGNORECASE)
    if m and m.start() > 0:
        t = t[m.start():]
    # cut at the first semicolon that is not inside a string literal
    out, in_str, quote = [], False, ""
    for ch in t:
        if in_str:
            out.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str, quote = True, ch
            out.append(ch)
            continue
        if ch == ";":
            break
        out.append(ch)
    sql = "".join(out).strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


# --------------------------------------------------------------------------- #
# tokenizing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tok:
    kind: str      # ident | keyword | string | number | punct | op
    value: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.kind}:{self.value}"


_STR_RE = re.compile(r"'(?:[^']|'')*'")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_OPS = ("<=", ">=", "<>", "!=", "||", "=", "<", ">", "+", "-", "*", "/", "%")
_PUNCT = "(),."


def tokenize(sql: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "'":
            m = _STR_RE.match(sql, i)
            if m:
                toks.append(Tok("string", m.group(0)))
                i = m.end()
                continue
            toks.append(Tok("string", sql[i:]))  # unterminated; validator flags it
            break
        if ch in "\"`[":
            close = {"\"": "\"", "`": "`", "[": "]"}[ch]
            j = sql.find(close, i + 1)
            if j < 0:
                toks.append(Tok("ident", sql[i + 1:]))
                break
            toks.append(Tok("ident", sql[i + 1:j]))
            i = j + 1
            continue
        if ch.isdigit():
            m = _NUM_RE.match(sql, i)
            assert m is not None
            toks.append(Tok("number", m.group(0)))
            i = m.end()
            continue
        m = _IDENT_RE.match(sql, i)
        if m:
            word = m.group(0)
            kind = "keyword" if word.lower() in SQL_KEYWORDS else "ident"
            toks.append(Tok(kind, word))
            i = m.end()
            continue
        for op in _OPS:
            if sql.startswith(op, i):
                toks.append(Tok("op", op))
                i += len(op)
                break
        else:
            if ch in _PUNCT:
                toks.append(Tok("punct", ch))
            else:
                toks.append(Tok("punct", ch))
            i += 1
    return toks


# --------------------------------------------------------------------------- #
# static analysis / validation
# --------------------------------------------------------------------------- #
@dataclass
class SqlIssue:
    code: str
    detail: str
    severity: str = "error"     # error | warning

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.severity}:{self.code}] {self.detail}"


@dataclass
class SqlAnalysis:
    sql: str
    tables: list[str] = field(default_factory=list)
    columns: list[tuple[str | None, str]] = field(default_factory=list)
    issues: list[SqlIssue] = field(default_factory=list)
    has_aggregate: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_join: bool = False
    has_subquery: bool = False
    has_set_op: bool = False
    num_placeholders: int = 0

    @property
    def errors(self) -> list[SqlIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[SqlIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


# clause keywords after which an identifier may be an alias rather than a column
_ALIAS_STOP = {"from", "where", "group", "order", "having", "limit", "on", "join",
               "union", "except", "intersect", "select", "as", "and", "or", "by"}


def analyze_sql(sql: str, db: Database | None = None) -> SqlAnalysis:
    """Static analysis + optional schema validation."""
    an = SqlAnalysis(sql=sql)
    if not sql.strip():
        an.issues.append(SqlIssue("empty_sql", "empty statement"))
        return an
    toks = tokenize(sql)
    if not toks:
        an.issues.append(SqlIssue("empty_sql", "no tokens"))
        return an

    lower = [t.value.lower() if t.kind != "string" else t.value for t in toks]
    an.has_aggregate = any(w in {"count", "sum", "avg", "min", "max"} for w in lower)
    an.has_group_by = _has_seq(lower, ["group", "by"])
    an.has_order_by = _has_seq(lower, ["order", "by"])
    an.has_limit = "limit" in lower
    an.has_join = "join" in lower
    an.has_set_op = any(w in {"union", "except", "intersect"} for w in lower)
    an.has_subquery = lower.count("select") > 1
    an.num_placeholders = sum(1 for t in toks if t.kind == "punct" and t.value == "?")

    # ---- string literal balance ---------------------------------------
    if sql.count("'") % 2 != 0:
        an.issues.append(SqlIssue("unterminated_string", "odd number of single quotes"))
    # ---- paren balance ------------------------------------------------
    depth = 0
    for t in toks:
        if t.value == "(" and t.kind == "punct":
            depth += 1
        elif t.value == ")" and t.kind == "punct":
            depth -= 1
            if depth < 0:
                an.issues.append(SqlIssue("unbalanced_paren", "closing paren without opener"))
                depth = 0
    if depth > 0:
        an.issues.append(SqlIssue("unbalanced_paren", f"{depth} unclosed paren(s)"))
    # ---- must start with select / with ---------------------------------
    first = next((t for t in toks if t.kind != "punct" or t.value == "("), None)
    if first is None or first.value.lower() not in {"select", "with"}:
        an.issues.append(SqlIssue("not_select", f"statement starts with {first.value!r}, expected SELECT/WITH"))
    # ---- no DML/DDL ----------------------------------------------------
    banned = {"insert", "update", "delete", "drop", "alter", "create", "attach",
              "detach", "pragma", "vacuum", "replace"}
    for t in toks:
        if t.kind == "ident" and t.value.lower() in banned:
            an.issues.append(SqlIssue("destructive_statement",
                                      f"forbidden keyword {t.value.upper()}", "error"))
            break
    # ---- aggregate without GROUP BY is only legal if no bare column ----
    if an.has_aggregate and not an.has_group_by:
        # bare columns alongside an aggregate is legal in SQLite but almost
        # always a modelling error for text2sql; flag as warning only.
        if _bare_column_with_aggregate(toks):
            an.issues.append(SqlIssue(
                "aggregate_without_group_by",
                "SELECT mixes aggregate and bare column without GROUP BY", "warning"))

    # ---- schema linking -------------------------------------------------
    if db is not None:
        _link_schema(an, toks, db)
    return an


def _has_seq(words: Sequence[str], seq: Sequence[str]) -> bool:
    n = len(seq)
    for i in range(len(words) - n + 1):
        if list(words[i:i + n]) == list(seq):
            return True
    return False


def _bare_column_with_aggregate(toks: list[Tok]) -> bool:
    # find the SELECT ... FROM span at depth 0
    depth = 0
    sel_at = None
    for i, t in enumerate(toks):
        if t.value == "(" and t.kind == "punct":
            depth += 1
        elif t.value == ")" and t.kind == "punct":
            depth -= 1
        elif depth == 0 and t.kind == "keyword" and t.value.lower() == "select":
            sel_at = i
            break
    if sel_at is None:
        return False
    depth = 0
    has_agg = False
    has_bare = False
    for t in toks[sel_at + 1:]:
        if t.value == "(" and t.kind == "punct":
            depth += 1
        elif t.value == ")" and t.kind == "punct":
            depth -= 1
        elif depth == 0 and t.kind == "keyword" and t.value.lower() in {"from", "where"}:
            break
        if depth == 0 and t.kind == "ident":
            if t.value.lower() in {"count", "sum", "avg", "min", "max"}:
                has_agg = True
            elif t.value.lower() not in SQL_KEYWORDS:
                has_bare = True
    return has_agg and has_bare


def _link_schema(an: SqlAnalysis, toks: list[Tok], db: Database) -> None:
    """Resolve table and column references against the schema."""
    table_norms = {t.norm: t for t in db.tables}
    col_by_norm: dict[str, list[tuple[str, Any]]] = {}
    for t in db.tables:
        for c in t.columns:
            col_by_norm.setdefault(c.norm, []).append((t.norm, c))

    # ---- tables referenced -------------------------------------------
    refs: list[str] = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.kind == "keyword" and t.value.lower() in {"from", "join"}:
            j = i + 1
            while j < n:
                tj = toks[j]
                if tj.kind == "punct" and tj.value == "(":
                    break
                if tj.kind in ("ident",):
                    refs.append(tj.value)
                elif tj.kind == "keyword" and tj.value.lower() in {"select"}:
                    break
                elif tj.kind == "punct" and tj.value == ",":
                    pass
                elif tj.kind == "keyword" and tj.value.lower() in {"as", "on", "where",
                                                                   "group", "order", "limit",
                                                                   "join", "inner", "left",
                                                                   "right", "outer", "having"}:
                    break
                j += 1
            i = j
            continue
        i += 1

    seen: set[str] = set()
    for r in refs:
        rn = normalize_ident(r)
        if rn in table_norms:
            if rn not in seen:
                seen.add(rn)
                an.tables.append(table_norms[rn].name)
        elif rn in col_by_norm:
            # a column in FROM position is a modelling error, but the real
            # message is "table not found"; keep the specific code.
            an.issues.append(SqlIssue("column_used_as_table", f"{r!r} is a column, not a table"))
        else:
            an.issues.append(SqlIssue("unknown_table", f"table {r!r} not in schema"))

    if not refs:
        an.issues.append(SqlIssue("no_from_clause", "no table reference found after FROM/JOIN"))
        return

    # ---- columns ------------------------------------------------------
    # aliases: `FROM t AS a` / `JOIN t a` / `FROM t a`
    aliases: dict[str, str] = {}
    for i, t in enumerate(toks):
        if t.kind == "keyword" and t.value.lower() in {"from", "join"} and i + 2 < n:
            a, b = toks[i + 1], toks[i + 2]
            if a.kind == "ident" and b.kind == "ident" and b.value.lower() not in SQL_KEYWORDS:
                aliases[b.value.lower()] = a.value
            elif a.kind == "ident" and b.kind == "keyword" and b.value.lower() == "as" and i + 3 < n:
                c = toks[i + 3]
                if c.kind == "ident":
                    aliases[c.value.lower()] = a.value

    for i, t in enumerate(toks):
        if t.kind != "ident":
            continue
        prev = toks[i - 1] if i > 0 else None
        if prev is not None and prev.kind == "punct" and prev.value == ".":
            continue  # handled with the qualifier
        nxt = toks[i + 1] if i + 1 < n else None
        qualified = nxt is not None and nxt.kind == "punct" and nxt.value == "."
        if qualified and i + 2 < n and toks[i + 2].kind == "ident":
            tbl_raw, col_raw = t.value, toks[i + 2].value
            tbl_norm = normalize_ident(aliases.get(tbl_raw.lower(), tbl_raw))
            col_norm = normalize_ident(col_raw)
            an.columns.append((tbl_raw, col_raw))
            if tbl_norm not in table_norms:
                an.issues.append(SqlIssue("unknown_table_qualifier",
                                          f"qualifier {tbl_raw!r} not a table/alias"))
            else:
                tab = table_norms[tbl_norm]
                if tab.column(col_raw) is None:
                    an.issues.append(SqlIssue(
                        "unknown_column",
                        f"column {col_raw!r} not in table {tab.name!r}"))
            continue
        v = t.value.lower()
        if v in SQL_KEYWORDS:
            continue
        if v in aliases or v in table_norms:
            continue
        # skip alias definitions and function names
        if nxt is not None and nxt.kind == "punct" and nxt.value == "(":
            continue  # function call
        if prev is not None and prev.kind == "keyword" and prev.value.lower() == "as":
            continue  # alias definition
        if v in col_by_norm:
            an.columns.append((None, t.value))
        else:
            # could be a bare alias introduced without AS: `... ) total`
            if prev is not None and prev.kind == "punct" and prev.value == ")":
                continue
            if prev is not None and prev.kind == "keyword" and prev.value.lower() in {
                    "by", "distinct"}:
                an.issues.append(SqlIssue("unknown_column", f"column {t.value!r} not in schema"))
                continue
            an.issues.append(SqlIssue("unknown_column", f"column {t.value!r} not in schema"))


# --------------------------------------------------------------------------- #
# normalization for exact match
# --------------------------------------------------------------------------- #
_ALIAS_AS_RE = re.compile(r"\s+as\s+[a-z_][a-z0-9_]*", re.IGNORECASE)


def normalize_sql(sql: str, *, strip_aliases: bool = True) -> str:
    """Spider-style exact-match normalization.

    Applied in this order: lowercase, collapse whitespace, drop ``AS alias``,
    drop surrounding backticks/double quotes, remove trailing semicolon.
    Literal values and table/column names are preserved (case-folded only).
    """
    if not sql:
        return ""
    s = sql.strip().rstrip(";").strip()
    if strip_aliases:
        s = _ALIAS_AS_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"[\"`]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),])\s*", r"\1", s)
    s = re.sub(r"\s*([<>=!]+)\s*", r"\1", s)
    return s.strip()


def sql_skeleton(sql: str) -> str:
    """Abstract literals out of a SQL string -> structural signature.

    Used for the "same query shape" check in the semantic validator and for
    deduplicating paraphrases that collapse to the identical query.
    """
    toks = tokenize(sql)
    out: list[str] = []
    for t in toks:
        if t.kind == "string":
            out.append("'<str>'")
        elif t.kind == "number":
            out.append("<num>")
        else:
            out.append(t.value.lower())
    return " ".join(out)


def sql_literals(sql: str) -> list[str]:
    return [t.value for t in tokenize(sql) if t.kind in ("string", "number")]


def referenced_entities(sql: str) -> set[str]:
    """Lowercased identifiers that are *not* keywords -- for overlap metrics."""
    return {t.value.lower() for t in tokenize(sql)
            if t.kind == "ident" and t.value.lower() not in SQL_KEYWORDS}


def has_limit(sql: str) -> bool:
    return "limit" in (t.value.lower() for t in tokenize(sql))


# --------------------------------------------------------------------------- #
# difficulty scoring (drives hard-example resampling)
# --------------------------------------------------------------------------- #
def difficulty_score(sql: str, db: Database | None = None) -> float:
    """Heuristic 0..1 difficulty used to pick hard examples for resampling.

    Combines structural complexity (joins, subqueries, set ops, aggregation +
    group by, HAVING, nesting depth), schema size, and the fraction of literal
    values that must be copied exactly (a known text2sql failure mode).
    Deliberately interpretable: every term is visible in the score.
    """
    an = analyze_sql(sql, db)
    toks = tokenize(sql)
    score = 0.0
    score += 0.16 if an.has_join else 0.0
    score += 0.10 * min(len(re.findall(r"\bjoin\b", sql, re.IGNORECASE)), 3) / 3.0 * 3
    score += 0.18 if an.has_subquery else 0.0
    score += 0.14 if an.has_set_op else 0.0
    score += 0.10 if (an.has_aggregate and an.has_group_by) else 0.0
    score += 0.08 if re.search(r"\bhaving\b", sql, re.IGNORECASE) else 0.0
    score += 0.08 if re.search(r"\border\s+by\b", sql, re.IGNORECASE) and an.has_limit else 0.0
    depth = _max_depth(toks)
    score += min(0.12, 0.03 * max(0, depth - 1))
    if db is not None:
        score += min(0.10, 0.02 * max(0, len(db.tables) - 3))
    lits = [t for t in toks if t.kind == "string"]
    score += min(0.10, 0.03 * len(lits))
    score += 0.05 if len(sql) > 160 else 0.0
    return round(min(1.0, score), 4)


def _max_depth(toks: list[Tok]) -> int:
    d = m = 0
    for t in toks:
        if t.value == "(" and t.kind == "punct":
            d += 1
            m = max(m, d)
        elif t.value == ")" and t.kind == "punct":
            d -= 1
    return m


# --------------------------------------------------------------------------- #
# error injection (hard-negative construction, offline deterministic path)
# --------------------------------------------------------------------------- #
ERROR_KINDS = [
    "wrong_table",        # swap in a table that exists but is irrelevant
    "wrong_column",       # replace a column with a sibling column from another table
    "wrong_aggregate",    # COUNT/SUM/AVG swapped
    "wrong_operator",     # > vs >=, = vs <>
    "missing_group_by",   # aggregate with bare column
    "missing_join",       # FROM only the first table
    "wrong_literal",      # mutate a literal value
    "missing_distinct",
    "wrong_order",        # ASC/DESC or missing ORDER BY direction
]


def inject_error(sql: str, db: Database, kind: str) -> str:
    """Deterministically corrupt a gold SQL in a schema-plausible way.

    Returns the corrupted SQL. May return the input unchanged when the
    requested corruption does not apply to this query; callers must verify
    with ``analyze_sql`` / execution that the result actually differs.
    """
    if kind == "wrong_aggregate":
        for a, b in (("count", "sum"), ("sum", "count"), ("avg", "sum"),
                     ("min", "max"), ("max", "min")):
            if re.search(rf"\b{a}\s*\(", sql, re.IGNORECASE):
                return re.sub(rf"\b{a}\s*\(", f"{b.upper()}(", sql, count=1, flags=re.IGNORECASE)
        return sql
    if kind == "wrong_operator":
        # Work on token boundaries so that replacing "=" cannot corrupt ">=".
        # Flip the first comparison operator into a genuinely different one that
        # usually changes the result set (>= <-> >, = <-> <>, > <-> <).
        swaps = {">=": ">", "<=": "<", ">": ">=", "<": "<=", "=": "<>", "<>": "="}
        for t in tokenize(sql):
            if t.kind == "op" and t.value in swaps:
                new = swaps[t.value]
                # replace only the first textual occurrence of this operator
                pattern = re.escape(t.value)
                return re.sub(rf"(?<![<>=!]){pattern}(?![<>=])", new, sql, count=1)
        return sql
    if kind == "wrong_literal":
        m = re.search(r"'([^']*)'", sql)
        if m:
            lit = m.group(1)
            alt = lit + "_x" if len(lit) < 3 else lit[:-1] + ("z" if lit[-1] != "z" else "y")
            return sql[:m.start(1)] + alt + sql[m.end(1):]
        m = re.search(r"\b(\d+)\b", sql)
        if m:
            return sql[:m.start(1)] + str(int(m.group(1)) + 1) + sql[m.end(1):]
        return sql
    if kind == "missing_group_by":
        return re.sub(r"\s+group\s+by\s+[^)]*?(?=(?:\s+having|\s+order|\s+limit|$))",
                      " ", sql, count=1, flags=re.IGNORECASE).strip()
    if kind == "missing_distinct":
        return re.sub(r"\bselect\s+distinct\b", "SELECT", sql, count=1, flags=re.IGNORECASE)
    if kind == "wrong_order":
        if re.search(r"\basc\b", sql, re.IGNORECASE):
            return re.sub(r"\basc\b", "DESC", sql, count=1, flags=re.IGNORECASE)
        if re.search(r"\bdesc\b", sql, re.IGNORECASE):
            return re.sub(r"\bdesc\b", "ASC", sql, count=1, flags=re.IGNORECASE)
        return re.sub(r"\border\s+by\s+", "ORDER BY 0, ", sql, count=1, flags=re.IGNORECASE)
    if kind == "wrong_table":
        m = re.search(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE)
        if not m:
            return sql
        used = m.group(1).lower()
        cands = [t.name for t in db.tables if t.norm != normalize_ident(used)]
        if not cands:
            return sql
        return sql[:m.start(1)] + cands[0] + sql[m.end(1):]
    if kind == "wrong_column":
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", sql)
        # replace the first selected identifier that is a real column
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", sql):
            word = m.group(0)
            if word.lower() in SQL_KEYWORDS:
                continue
            for t in db.tables:
                if t.column(word) is not None and t.columns:
                    sib = next((c for c in t.columns if c.norm != normalize_ident(word)), None)
                    if sib is not None:
                        return sql[:m.start()] + sib.name + sql[m.end():]
        return sql
    if kind == "missing_join":
        # keep only the first table, drop join clauses
        sql2 = re.sub(r"\s+(?:inner\s+|left\s+|right\s+|full\s+)?join\s+[^)]*?(?=(?:\s+where|\s+group|\s+order|\s+limit|$))",
                      " ", sql, count=1, flags=re.IGNORECASE)
        return sql2.strip()
    return sql
