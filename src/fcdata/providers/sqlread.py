"""Read SQL and DDL *back* into structured facts.

Why this module exists
----------------------
The offline teacher (:mod:`fcdata.providers.mock`) only ever sees a **prompt** --
it has no access to ``SpiderData``, no ``Database`` object, no execution engine.
Yet its whole job is to explain a gold query and to judge a candidate query
*grounded in the real schema*. So it must be able to recover the schema from the
DDL block that the prompt embeds, and recover the clause structure of the SQL
text. That recovery is this module.

Everything here is deliberately dependency-free and string/token based: the
project bans ``sqlglot``/``sqlparse`` (see CONTRACTS.md) so that the pipeline
installs with stdlib + numpy/torch only. The parser is therefore *not* a full SQL
grammar -- it is a pragmatic clause reader that is correct for the SELECT-shaped
statements this project generates (single statement, optional set operations,
optional subqueries), and it degrades to "no facts" instead of raising.

The recovered structures are also reused by the mock judge to run
:func:`fcdata.sqlutils.analyze_sql` against a schema reconstructed from the
prompt, which is what makes the judge *calibrated* rather than a coin flip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..schema import Database, ForeignKey, Table, Column, normalize_ident
from ..sqlutils import SQL_KEYWORDS, tokenize

# --------------------------------------------------------------------------- #
# DDL -> schema
# --------------------------------------------------------------------------- #
_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[`\"\[]?(?P<name>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\((?P<body>.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_FK_COMMENT_RE = re.compile(
    r"--\s*FOREIGN\s+KEY\s*\(\s*[`\"\[]?(?P<st>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\.\s*"
    r"[`\"\[]?(?P<sc>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\)\s*REFERENCES\s+"
    r"[`\"\[]?(?P<dt>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\(\s*"
    r"[`\"\[]?(?P<dc>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\)",
    re.IGNORECASE,
)
_INLINE_FK_RE = re.compile(
    r"(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(\s*[`\"\[]?(?P<sc>[A-Za-z_][A-Za-z0-9_$]*)"
    r"[`\"\]]?\s*\)\s*REFERENCES\s+[`\"\[]?(?P<dt>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*"
    r"\(\s*[`\"\[]?(?P<dc>[A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DdlColumn:
    name: str
    type: str = "text"
    primary_key: bool = False


@dataclass
class DdlTable:
    name: str
    columns: list[DdlColumn] = field(default_factory=list)

    def column(self, name: str) -> DdlColumn | None:
        n = normalize_ident(name)
        for c in self.columns:
            if normalize_ident(c.name) == n:
                return c
        return None

    @property
    def norm(self) -> str:
        return normalize_ident(self.name)


@dataclass
class DdlSchema:
    """Schema recovered from a prompt's ``CREATE TABLE`` block."""

    tables: list[DdlTable] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------
    def table(self, name: str) -> DdlTable | None:
        n = normalize_ident(name)
        for t in self.tables:
            if t.norm == n:
                return t
        return None

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def column_type(self, table: str, column: str) -> str:
        t = self.table(table)
        c = t.column(column) if t else None
        return c.type if c else "text"

    def resolve_column(self, name: str) -> list[tuple[str, str]]:
        """All ``(table, column)`` pairs whose column name matches ``name``."""
        n = normalize_ident(name)
        return [(t.name, c.name) for t in self.tables for c in t.columns
                if normalize_ident(c.name) == n]

    def fk_between(self, left: str, right: str) -> tuple[str, str, str, str] | None:
        """The FK pair linking two tables, in either direction."""
        a, b = normalize_ident(left), normalize_ident(right)
        for st, sc, dt, dc in self.foreign_keys:
            if normalize_ident(st) == a and normalize_ident(dt) == b:
                return (st, sc, dt, dc)
            if normalize_ident(st) == b and normalize_ident(dt) == a:
                return (dt, dc, st, sc)
        return None

    def to_database(self, db_id: str = "prompt_db") -> Database:
        """Materialize a :class:`fcdata.schema.Database` so ``analyze_sql`` works.

        This is the bridge that lets the mock judge reuse the *real* validator
        (schema linking, unknown-column detection) instead of re-implementing it.
        """
        db = Database(db_id=db_id)
        index_of: dict[str, int] = {}
        for i, t in enumerate(self.tables):
            cols = [Column(table_index=i, name=c.name, type=c.type,
                           is_primary_key=c.primary_key) for c in t.columns]
            db.tables.append(Table(index=i, name=t.name, columns=cols))
            index_of[t.norm] = i
        for st, sc, dt, dc in self.foreign_keys:
            si, di = index_of.get(normalize_ident(st)), index_of.get(normalize_ident(dt))
            if si is None or di is None:
                continue
            src = next((k for k, c in enumerate(db.tables[si].columns)
                        if c.norm == normalize_ident(sc)), None)
            dst = next((k for k, c in enumerate(db.tables[di].columns)
                        if c.norm == normalize_ident(dc)), None)
            if src is None or dst is None:
                continue
            db.foreign_keys.append(ForeignKey(si, src, di, dst))
        return db

    @property
    def is_empty(self) -> bool:
        return not self.tables


def _split_paren_aware(body: str) -> list[str]:
    """Split a CREATE TABLE body on top-level commas (keeps ``VARCHAR(20)``)."""
    out: list[str] = []
    depth = 0
    in_str = False
    start = 0
    for i, ch in enumerate(body):
        if in_str:
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(body[start:i])
            start = i + 1
    out.append(body[start:])
    return [p.strip() for p in out if p.strip()]


def parse_ddl(ddl: str) -> DdlSchema:
    """Parse a ``CREATE TABLE`` block into a :class:`DdlSchema`.

    Tolerant by design: unknown constraint lines are skipped rather than raising,
    because the DDL in a prompt may come from any of three renderers (Spider's
    ``schema.sql``, ``Database.render_ddl``, or a hand-written fixture).
    """
    schema = DdlSchema()
    if not ddl:
        return schema
    for m in _CREATE_RE.finditer(ddl):
        table = DdlTable(name=m.group("name"))
        for part in _split_paren_aware(m.group("body")):
            if _INLINE_FK_RE.search(part):
                continue
            if re.match(r"^(?:CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK)\b",
                        part, re.IGNORECASE):
                continue
            tokens = part.split()
            if not tokens:
                continue
            col_name = tokens[0].strip('`"[]')
            if not re.match(r"^[A-Za-z_]", col_name):
                continue
            col_type = tokens[1] if len(tokens) > 1 else "text"
            table.columns.append(DdlColumn(
                name=col_name,
                type=_normalize_ddl_type(col_type),
                primary_key=bool(re.search(r"\bPRIMARY\s+KEY\b", part, re.IGNORECASE)),
            ))
        if table.columns:
            schema.tables.append(table)
    for m in _FK_COMMENT_RE.finditer(ddl):
        schema.foreign_keys.append((m.group("st"), m.group("sc"),
                                    m.group("dt"), m.group("dc")))
    # inline FOREIGN KEY (col) REFERENCES t(c) -- the table name is the last one seen
    for m in _INLINE_FK_RE.finditer(ddl):
        src_table = _table_containing(schema, m.start(), ddl)
        if src_table is not None:
            schema.foreign_keys.append((src_table, m.group("sc"),
                                        m.group("dt"), m.group("dc")))
    return schema


def _table_containing(schema: DdlSchema, offset: int, ddl: str) -> str | None:
    """Which CREATE TABLE block contains ``offset`` (for inline FK constraints)."""
    best: str | None = None
    for m in _CREATE_RE.finditer(ddl):
        if m.start() <= offset <= m.end():
            best = m.group("name")
    return best


_DDL_TYPE_MAP = {
    "int": "number", "integer": "number", "real": "number", "float": "number",
    "double": "number", "decimal": "number", "numeric": "number", "number": "number",
    "date": "time", "datetime": "time", "time": "time", "timestamp": "time", "year": "time",
    "bool": "boolean", "boolean": "boolean",
}


def _normalize_ddl_type(raw: str) -> str:
    t = raw.strip().lower().rstrip(",")
    t = re.sub(r"\(.*\)", "", t).strip()
    if t in _DDL_TYPE_MAP:
        return _DDL_TYPE_MAP[t]
    for key, val in _DDL_TYPE_MAP.items():
        if t.startswith(key):
            return val
    if any(k in t for k in ("char", "text", "varchar", "string", "clob", "blob")):
        return "text"
    return "text"


# --------------------------------------------------------------------------- #
# SQL text -> clause structure
# --------------------------------------------------------------------------- #
def _depth_and_string_map(sql: str) -> tuple[list[int], list[bool]]:
    """Per-character paren depth and "inside a string literal" flag.

    Needed because every clause split must ignore separators that appear inside
    a literal (``WHERE name = 'a, b'``) or inside a subquery.
    """
    depth = [0] * len(sql)
    in_str = [False] * len(sql)
    d = 0
    quoted = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if quoted:
            in_str[i] = True
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":   # escaped quote ''
                    in_str[i + 1] = True
                    i += 2
                    continue
                quoted = False
            i += 1
            continue
        if ch == "'":
            quoted = True
            in_str[i] = True
            i += 1
            continue
        if ch == "(":
            depth[i] = d
            d += 1
        elif ch == ")":
            d = max(0, d - 1)
            depth[i] = d
        else:
            depth[i] = d
        i += 1
    return depth, in_str


_CLAUSE_ORDER = ("select", "from", "where", "group by", "having", "order by", "limit")
_SET_OPS = ("union all", "union", "except", "intersect")


def _find_keyword(sql: str, words: Sequence[str], start: int = 0) -> int:
    """First top-level, non-string occurrence of any of ``words`` at/after ``start``."""
    depth, in_str = _depth_and_string_map(sql)
    pats = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
    best = -1
    for pat in pats:
        for m in pat.finditer(sql, start):
            i = m.start()
            if i < len(depth) and depth[i] == 0 and not in_str[i]:
                if best < 0 or i < best:
                    best = i
                break
    return best


def _split_top_level(sql: str, seps: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``sql`` on top-level occurrences of any separator; return parts + seps.

    Word separators (``and``, ``union all``) additionally require non-word
    characters on both sides, so ``brand`` is not split on ``and``. Punctuation
    separators (``,``) have no such requirement -- otherwise ``T1.name, COUNT(*)``
    would never split, because the character before the comma is a letter.
    """
    depth, in_str = _depth_and_string_map(sql)
    ordered = sorted(seps, key=len, reverse=True)
    parts: list[str] = []
    found: list[str] = []
    start = 0
    i = 0
    n = len(sql)
    while i < n:
        if depth[i] == 0 and not in_str[i]:
            for sep in ordered:
                if sql[i:i + len(sep)].lower() != sep:
                    continue
                if sep[:1].isalnum() or sep[-1:].isalnum():
                    before = sql[i - 1] if i > 0 else " "
                    after = sql[i + len(sep)] if i + len(sep) < n else " "
                    if (before.isalnum() or before == "_"
                            or after.isalnum() or after == "_"):
                        continue
                parts.append(sql[start:i])
                found.append(sep)
                i += len(sep)
                start = i
                break
            else:
                i += 1
            continue
        i += 1
    parts.append(sql[start:])
    return parts, found


def split_set_ops(sql: str) -> tuple[list[str], list[str]]:
    """Split a top-level set operation (``UNION``/``EXCEPT``/``INTERSECT``)."""
    return _split_top_level(sql, _SET_OPS)


def clause_spans(sql: str) -> dict[str, str]:
    """Return the raw text of each top-level clause of one SELECT block."""
    spans: dict[str, str] = {}
    positions: list[tuple[int, str]] = []
    for kw in _CLAUSE_ORDER:
        pos = _find_keyword(sql, [kw])
        if pos >= 0:
            positions.append((pos, kw))
    positions.sort()
    for idx, (pos, kw) in enumerate(positions):
        kw_len = len(kw)
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(sql)
        spans[kw] = sql[pos + kw_len:end].strip()
    return spans


# --------------------------------------------------------------------------- #
# resolved references
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ref:
    """A column reference resolved against the prompt schema."""

    raw: str
    table: str
    column: str
    type: str = "text"

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class SelectItem:
    raw: str
    expr: str
    alias: str = ""
    aggregate: str = ""
    distinct: bool = False
    refs: list[Ref] = field(default_factory=list)
    star: bool = False

    @property
    def is_bare_column(self) -> bool:
        return not self.aggregate and not self.star and len(self.refs) == 1


@dataclass
class JoinClause:
    kind: str
    table: str
    alias: str = ""
    on: str = ""
    pairs: list[tuple[Ref, Ref]] = field(default_factory=list)


@dataclass
class Predicate:
    raw: str
    connector: str = ""      # "" for the first predicate, then "and"/"or"
    op: str = ""             # = <> < <= > >= like in between isnull notnull exists
    left: str = ""
    right: str = ""
    refs: list[Ref] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)
    has_subquery: bool = False


@dataclass
class OrderTerm:
    expr: str
    descending: bool = False
    refs: list[Ref] = field(default_factory=list)
    aggregate: str = ""


@dataclass
class SqlFacts:
    sql: str
    select_distinct: bool = False
    select_items: list[SelectItem] = field(default_factory=list)
    from_tables: list[tuple[str, str]] = field(default_factory=list)   # (table, alias)
    joins: list[JoinClause] = field(default_factory=list)
    where: list[Predicate] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    group_refs: list[Ref] = field(default_factory=list)
    having: str = ""
    order_by: list[OrderTerm] = field(default_factory=list)
    limit: int | None = None
    set_ops: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)

    # -- convenience -----------------------------------------------------
    @property
    def has_aggregate(self) -> bool:
        return any(i.aggregate for i in self.select_items) or bool(self.having)

    @property
    def aggregates(self) -> list[str]:
        return [i.aggregate for i in self.select_items if i.aggregate]

    @property
    def tables(self) -> list[str]:
        seen: list[str] = []
        for t, _ in self.from_tables:
            if t not in seen:
                seen.append(t)
        for j in self.joins:
            if j.table not in seen:
                seen.append(j.table)
        return seen

    @property
    def n_select_items(self) -> int:
        return len(self.select_items)


_AGG_RE = re.compile(r"^\s*(count|sum|avg|min|max)\s*\(", re.IGNORECASE)
_QUALIFIED_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_$]*)\b")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_STR_LIT_RE = re.compile(r"'(?:[^']|'')*'")


def _strip_alias(expr: str) -> tuple[str, str]:
    """Split ``expr AS alias`` / ``expr alias`` into (expr, alias)."""
    m = re.match(r"^(.*?)\s+AS\s+([A-Za-z_][A-Za-z0-9_$]*)\s*$", expr,
                 re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2)
    m = re.match(r"^(.*[)\w'\"`\]])\s+([A-Za-z_][A-Za-z0-9_$]*)\s*$", expr, re.DOTALL)
    if m and not re.search(r"\b(and|or|not|like|in|between|is)\s*$", m.group(1),
                           re.IGNORECASE):
        return m.group(1).strip(), m.group(2)
    return expr.strip(), ""


def _alias_map(from_tables: Sequence[tuple[str, str]],
               joins: Sequence[JoinClause]) -> dict[str, str]:
    """alias (lowercased) -> real table name, including the table name itself."""
    out: dict[str, str] = {}
    for table, alias in list(from_tables) + [(j.table, j.alias) for j in joins]:
        out[normalize_ident(table)] = table
        if alias:
            out[normalize_ident(alias)] = table
    return out


def resolve_refs(fragment: str, schema: DdlSchema,
                 aliases: dict[str, str]) -> list[Ref]:
    """Resolve every column reference in a SQL fragment against the schema."""
    refs: list[Ref] = []
    seen: set[tuple[str, str]] = set()

    def _add(table: str, column: str) -> None:
        key = (normalize_ident(table), normalize_ident(column))
        if key in seen:
            return
        seen.add(key)
        refs.append(Ref(raw=f"{table}.{column}", table=table, column=column,
                        type=schema.column_type(table, column)))

    for m in _QUALIFIED_RE.finditer(fragment):
        qualifier, column = m.group(1), m.group(2)
        table = aliases.get(normalize_ident(qualifier))
        if table is None:
            table = qualifier
        _add(table, column)
    # bare identifiers that are real columns and not keywords / not qualified
    without_qualified = _QUALIFIED_RE.sub(" ", _STR_LIT_RE.sub(" ", fragment))
    for m in _IDENT_RE.finditer(without_qualified):
        word = m.group(0)
        if word.lower() in SQL_KEYWORDS:
            continue
        tail = without_qualified[m.end():]
        if re.match(r"\s*\(", tail):        # function call
            continue
        if schema.table(word) is not None:  # table name
            continue
        hits = schema.resolve_column(word)
        if len(hits) == 1:
            _add(hits[0][0], hits[0][1])
        elif len(hits) > 1:
            _add(hits[0][0], hits[0][1])
    return refs


def _literals(fragment: str) -> list[str]:
    lits = [m.group(0) for m in _STR_LIT_RE.finditer(fragment)]
    lits += [m.group(0) for m in re.finditer(r"(?<![\w.'\"])-?\d+(?:\.\d+)?(?![\w.])",
                                             _STR_LIT_RE.sub(" ", fragment))]
    return lits


def parse_select_items(select_clause: str, schema: DdlSchema,
                       aliases: dict[str, str]) -> list[SelectItem]:
    parts, _ = _split_top_level(select_clause, [","])
    items: list[SelectItem] = []
    for part in parts:
        raw = part.strip()
        if not raw:
            continue
        expr, alias = _strip_alias(raw)
        agg_m = _AGG_RE.match(expr)
        aggregate = agg_m.group(1).lower() if agg_m else ""
        items.append(SelectItem(
            raw=raw, expr=expr, alias=alias, aggregate=aggregate,
            distinct=bool(re.match(r"^DISTINCT\b", expr, re.IGNORECASE)),
            refs=resolve_refs(expr, schema, aliases),
            star="*" in expr and not aggregate,
        ))
    return items


_JOIN_RE = re.compile(
    r"(?P<kind>\b(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+(?:OUTER\s+)?)?\bJOIN\b",
    re.IGNORECASE,
)


def parse_from(from_clause: str, schema: DdlSchema) -> tuple[
        list[tuple[str, str]], list[JoinClause], dict[str, str]]:
    """Parse ``FROM`` into base tables + joins + alias map."""
    from_tables: list[tuple[str, str]] = []
    joins: list[JoinClause] = []
    # split the clause at top-level JOIN keywords
    depth, in_str = _depth_and_string_map(from_clause)
    cuts: list[tuple[int, int, str]] = []
    for m in _JOIN_RE.finditer(from_clause):
        i = m.start()
        if i < len(depth) and depth[i] == 0 and not in_str[i]:
            kind = (m.group("kind") or "inner").strip().lower()
            kind = kind.replace(" outer", "").replace("join", "").strip() or "inner"
            cuts.append((i, m.end(), kind))
    base_end = cuts[0][0] if cuts else len(from_clause)
    base = from_clause[:base_end].strip()
    for chunk in _split_top_level(base, [","])[0]:
        table, alias = _parse_table_ref(chunk, schema)
        if table:
            from_tables.append((table, alias))
    for idx, (start, end, kind) in enumerate(cuts):
        stop = cuts[idx + 1][0] if idx + 1 < len(cuts) else len(from_clause)
        body = from_clause[end:stop].strip()
        on_pos = _find_keyword(body, ["on"])
        ref_part = body[:on_pos].strip() if on_pos >= 0 else body
        on = body[on_pos + 2:].strip() if on_pos >= 0 else ""
        table, alias = _parse_table_ref(ref_part, schema)
        if not table:
            continue
        aliases = _alias_map(from_tables, joins + [JoinClause(kind, table, alias)])
        pairs: list[tuple[Ref, Ref]] = []
        if on:
            refs = resolve_refs(on, schema, aliases)
            for a, b in zip(refs[0::2], refs[1::2]):
                pairs.append((a, b))
        joins.append(JoinClause(kind=kind, table=table, alias=alias, on=on, pairs=pairs))
    return from_tables, joins, _alias_map(from_tables, joins)


def _parse_table_ref(chunk: str, schema: DdlSchema) -> tuple[str, str]:
    """Parse ``table [AS] alias`` (also tolerates a trailing subquery alias)."""
    text = chunk.strip().rstrip(",").strip()
    if not text:
        return "", ""
    if text.startswith("("):
        m = re.search(r"\)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_$]*)\s*$", text,
                      re.IGNORECASE)
        return "", m.group(1) if m else ""
    m = re.match(r"[`\"\[]?([A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*"
                 r"(?:AS\s+)?([A-Za-z_][A-Za-z0-9_$]*)?", text, re.IGNORECASE)
    if not m:
        return "", ""
    table = m.group(1)
    alias = m.group(2) or ""
    if alias.lower() in SQL_KEYWORDS:
        alias = ""
    if schema.table(table) is None:
        # keep the raw name anyway: the judge needs to *report* unknown tables
        return table, alias
    return schema.table(table).name if schema.table(table) else table, alias


_OP_PATTERNS: tuple[tuple[str, str], ...] = (
    ("not like", "not like"), ("like", "like"),
    ("not in", "not in"), ("in", "in"),
    ("not between", "not between"), ("between", "between"),
    ("is not null", "notnull"), ("is null", "isnull"),
    (">=", ">="), ("<=", "<="), ("<>", "<>"), ("!=", "<>"),
    (">", ">"), ("<", "<"), ("=", "="),
)

#: operator patterns that are words (or word sequences) rather than symbols
_WORD_OP_RE = re.compile(r"^[a-z]+(?: [a-z]+)*$")


def parse_where(where_clause: str, schema: DdlSchema,
                aliases: dict[str, str]) -> list[Predicate]:
    """Split a WHERE clause into AND/OR-joined predicates with operators."""
    if not where_clause.strip():
        return []
    chunks, connectors = _split_top_level(where_clause, ["and", "or"])
    preds: list[Predicate] = []
    for idx, chunk in enumerate(chunks):
        raw = chunk.strip()
        if not raw:
            continue
        connector = connectors[idx - 1].lower() if idx > 0 and idx - 1 < len(connectors) else ""
        low = raw.lower()
        op = ""
        left, right = raw, ""
        for pat, name in _OP_PATTERNS:
            if _WORD_OP_RE.match(pat):
                pos = _find_keyword(raw, [pat])
            else:
                pos = _find_op(raw, pat)
            if pos >= 0:
                op = name
                left = raw[:pos].strip()
                right = raw[pos + len(pat):].strip()
                break
        preds.append(Predicate(
            raw=raw, connector=connector, op=op, left=left, right=right,
            refs=resolve_refs(raw, schema, aliases),
            literals=_literals(raw),
            has_subquery=bool(re.search(r"\bselect\b", low)),
        ))
    return preds


def _find_op(sql: str, op: str) -> int:
    depth, in_str = _depth_and_string_map(sql)
    i = sql.find(op)
    while i >= 0:
        if i < len(depth) and depth[i] == 0 and not in_str[i]:
            return i
        i = sql.find(op, i + 1)
    return -1


def parse_order(order_clause: str, schema: DdlSchema,
                aliases: dict[str, str]) -> list[OrderTerm]:
    terms: list[OrderTerm] = []
    for chunk in _split_top_level(order_clause, [","])[0]:
        raw = chunk.strip()
        if not raw:
            continue
        descending = bool(re.search(r"\bdesc\b", raw, re.IGNORECASE))
        expr = re.sub(r"\s+(?:asc|desc)\s*$", "", raw, flags=re.IGNORECASE).strip()
        agg_m = _AGG_RE.match(expr)
        terms.append(OrderTerm(expr=expr, descending=descending,
                               refs=resolve_refs(expr, schema, aliases),
                               aggregate=agg_m.group(1).lower() if agg_m else ""))
    return terms


def parse_limit(limit_clause: str) -> int | None:
    m = re.match(r"\s*(\d+)", limit_clause or "")
    return int(m.group(1)) if m else None


def _collect_subqueries(sql: str) -> list[str]:
    """Text of each parenthesised sub-SELECT, innermost last (good enough to explain)."""
    out: list[str] = []
    for m in re.finditer(r"\(([^()]*?\bselect\b[^()]*?)\)", sql, re.IGNORECASE | re.DOTALL):
        out.append(m.group(1).strip())
    return out


def read_sql(sql: str, schema: DdlSchema) -> SqlFacts:
    """Parse one SQL statement into :class:`SqlFacts` (never raises)."""
    facts = SqlFacts(sql=sql.strip())
    if not facts.sql:
        return facts
    parts, ops = split_set_ops(facts.sql)
    facts.set_ops = [o.lower() for o in ops]
    # facts are read from the first block; set operations are surfaced as a flag
    block = parts[0] if parts else facts.sql
    facts.subqueries = _collect_subqueries(block)
    spans = clause_spans(block)

    select_clause = spans.get("select", "")
    facts.select_distinct = bool(re.match(r"^\s*distinct\b", select_clause, re.IGNORECASE))
    if facts.select_distinct:
        select_clause = re.sub(r"^\s*distinct\b", "", select_clause, count=1,
                               flags=re.IGNORECASE)
    from_tables, joins, aliases = parse_from(spans.get("from", ""), schema)
    facts.from_tables = from_tables
    facts.joins = joins
    facts.select_items = parse_select_items(select_clause, schema, aliases)
    facts.where = parse_where(spans.get("where", ""), schema, aliases)
    group_raw = spans.get("group by", "")
    facts.group_by = [g.strip() for g in _split_top_level(group_raw, [","])[0] if g.strip()]
    facts.group_refs = resolve_refs(group_raw, schema, aliases)
    facts.having = spans.get("having", "")
    facts.order_by = parse_order(spans.get("order by", ""), schema, aliases)
    facts.limit = parse_limit(spans.get("limit", ""))
    return facts


# --------------------------------------------------------------------------- #
# small question-intent reader (used to keep explanations grounded in the ask)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuestionIntent:
    asks_count: bool = False
    asks_average: bool = False
    asks_sum: bool = False
    asks_extreme: str = ""          # "max" | "min" | ""
    asks_grouped: bool = False      # "each" / "per" / "for every"
    asks_ordered: bool = False
    asks_top: bool = False          # "top N" / "highest" / "most"
    asks_set_membership: bool = False   # "not in" / "that have no"
    asks_join: bool = False         # mentions a relationship word


_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("how many", "count"), ("number of", "count"), ("count", "count"),
    ("average", "average"), ("mean", "average"),
    ("total", "sum"), ("sum", "sum"),
    ("highest", "max"), ("largest", "max"), ("most", "max"), ("maximum", "max"),
    ("lowest", "min"), ("smallest", "min"), ("least", "min"), ("minimum", "min"),
)


def read_intent(question: str) -> QuestionIntent:
    q = (question or "").lower()
    flags = {name: (name in q) for _, name in _INTENT_PATTERNS}
    # explicit patterns beat naive substring checks
    asks_count = any(p in q for p in ("how many", "number of", "count"))
    asks_average = any(p in q for p in ("average", "mean"))
    asks_sum = any(p in q for p in ("total", "sum of", "summed"))
    extreme = ""
    for word, kind in (("highest", "max"), ("largest", "max"), ("most", "max"),
                       ("maximum", "max"), ("lowest", "min"), ("smallest", "min"),
                       ("least", "min"), ("minimum", "min")):
        if word in q:
            extreme = kind
            break
    return QuestionIntent(
        asks_count=asks_count,
        asks_average=asks_average,
        asks_sum=asks_sum,
        asks_extreme=extreme,
        asks_grouped=bool(re.search(r"\b(each|every|per)\b", q)),
        asks_ordered=bool(re.search(r"\b(order|sort|rank|from (?:high|low)|descending|ascending)\b", q)),
        asks_top=bool(re.search(r"\b(top|first|last)\b", q)),
        asks_set_membership=bool(re.search(
            r"\b(not in|no |never|without|that have no|who have no|excluded)\b", q)),
        asks_join=bool(re.search(
            r"\b(of|with|who|whose|that|for each|belong|associated|together)\b", q)),
    )


def iter_refs(facts: SqlFacts) -> Iterator[Ref]:
    """Every resolved column reference in a parsed statement."""
    seen: set[tuple[str, str]] = set()
    pools: list[Iterable[Ref]] = [
        [r for item in facts.select_items for r in item.refs],
        [r for p in facts.where for r in p.refs],
        facts.group_refs,
        [r for t in facts.order_by for r in t.refs],
        [r for j in facts.joins for pair in j.pairs for r in pair],
    ]
    for pool in pools:
        for ref in pool:
            key = (normalize_ident(ref.table), normalize_ident(ref.column))
            if key not in seen:
                seen.add(key)
                yield ref
