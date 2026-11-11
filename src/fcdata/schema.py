"""Spider schema model + DDL rendering + schema-linking helpers.

Spider ships ``tables.json`` with one entry per database::

    {"db_id": "department_management",
     "table_names_original": ["department", "head", "management", ...],
     "column_names_original": [[-1, "*"], [0, "Department_ID"], ...],
     "column_types": ["text", "number", ...],
     "primary_keys": [1, 5, ...],
     "foreign_keys": [[2, 3], ...]}

This module turns that into a typed model that the SQL validator, the prompt
builder and the fixture generator all share.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit", "join",
    "inner", "left", "right", "outer", "full", "on", "as", "and", "or", "not",
    "in", "is", "null", "like", "between", "distinct", "count", "sum", "avg",
    "min", "max", "asc", "desc", "union", "except", "intersect", "case", "when",
    "then", "else", "end", "exists", "all", "any", "with", "cast", "real",
}


@dataclass(frozen=True)
class Column:
    table_index: int
    name: str                 # original name (what appears in SQL)
    type: str = "text"        # text | number | time | boolean | other
    is_primary_key: bool = False

    @property
    def norm(self) -> str:
        return normalize_ident(self.name)


@dataclass(frozen=True)
class ForeignKey:
    src_table: int
    src_col: int
    dst_table: int
    dst_col: int


@dataclass
class Table:
    index: int
    name: str
    columns: list[Column] = field(default_factory=list)

    @property
    def norm(self) -> str:
        return normalize_ident(self.name)

    def column(self, name: str) -> Column | None:
        n = normalize_ident(name)
        for c in self.columns:
            if c.norm == n:
                return c
        return None


@dataclass
class Database:
    db_id: str
    tables: list[Table] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    # optional: real DDL / sample rows attached by the loader
    ddl: str = ""
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # -- lookups ---------------------------------------------------------
    def table(self, name: str) -> Table | None:
        n = normalize_ident(name)
        for t in self.tables:
            if t.norm == n:
                return t
        return None

    def column(self, table: str, column: str) -> Column | None:
        t = self.table(table)
        return t.column(column) if t else None

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def all_columns(self) -> Iterator[tuple[Table, Column]]:
        for t in self.tables:
            for c in t.columns:
                yield t, c

    # -- rendering -------------------------------------------------------
    def render_ddl(self, max_tables: int = 0) -> str:
        """Render a compact ``CREATE TABLE`` block used in prompts."""
        if self.ddl:
            return self.ddl
        tables = self.tables[:max_tables] if max_tables else self.tables
        lines: list[str] = []
        for t in tables:
            cols = ", ".join(
                f"{c.name} {c.type.upper()}" + (" PRIMARY KEY" if c.is_primary_key else "")
                for c in t.columns
            )
            lines.append(f"CREATE TABLE {t.name} ({cols});")
        for fk in self.foreign_keys:
            try:
                st = self.tables[fk.src_table]
                dt = self.tables[fk.dst_table]
                lines.append(
                    f"-- FOREIGN KEY ({st.name}.{st.columns[fk.src_col].name}) "
                    f"REFERENCES {dt.name}({dt.columns[fk.dst_col].name})"
                )
            except (IndexError, KeyError):
                continue
        return "\n".join(lines)

    def to_spider_json(self) -> dict[str, Any]:
        column_names: list[list[Any]] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        table_offsets: dict[int, int] = {}
        for t in self.tables:
            table_offsets[t.index] = len(column_names)
            for c in t.columns:
                column_names.append([t.index, c.name])
                column_types.append(c.type)
        return {
            "db_id": self.db_id,
            "table_names_original": [t.name for t in self.tables],
            "table_names": [t.name.replace("_", " ") for t in self.tables],
            "column_names_original": column_names,
            "column_names": [[i, n.replace("_", " ")] for i, n in column_names],
            "column_types": column_types,
            "primary_keys": [i for i, (_, c) in enumerate(
                [(None, col) for t in self.tables for col in t.columns], start=1)
                if c.is_primary_key],
            "foreign_keys": [
                [table_offsets[fk.src_table] + fk.src_col,
                 table_offsets[fk.dst_table] + fk.dst_col]
                for fk in self.foreign_keys
            ],
        }

    @classmethod
    def from_spider_json(cls, obj: dict[str, Any]) -> "Database":
        db_id = obj["db_id"]
        raw_tables: list[str] = obj["table_names_original"]
        tables = [Table(index=i, name=name) for i, name in enumerate(raw_tables)]
        pks = set(obj.get("primary_keys") or [])
        col_types = obj.get("column_types") or []
        # column_names_original[i] == [table_idx, col_name]; index 0 is the "*" placeholder
        col_index_to_table_col: dict[int, tuple[int, int]] = {}
        for ci, (ti, cname) in enumerate(obj["column_names_original"]):
            if ti < 0:
                continue
            ctype = col_types[ci] if ci < len(col_types) else "text"
            col = Column(table_index=ti, name=cname, type=normalize_type(ctype),
                         is_primary_key=ci in pks)
            tables[ti].columns.append(col)
            col_index_to_table_col[ci] = (ti, len(tables[ti].columns) - 1)
        fks: list[ForeignKey] = []
        for src, dst in obj.get("foreign_keys") or []:
            if src not in col_index_to_table_col or dst not in col_index_to_table_col:
                continue
            st, sc = col_index_to_table_col[src]
            dt, dc = col_index_to_table_col[dst]
            fks.append(ForeignKey(st, sc, dt, dc))
        return cls(db_id=db_id, tables=tables, foreign_keys=fks)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def normalize_ident(name: str) -> str:
    """Lowercase, strip quotes/backticks/brackets and separators."""
    n = name.strip().strip('`"[]')
    return re.sub(r"[\s\-]+", "_", n).lower()


def normalize_type(t: str) -> str:
    t = (t or "").lower()
    if any(k in t for k in ("int", "real", "float", "double", "number", "decimal", "numeric")):
        return "number"
    if any(k in t for k in ("date", "time", "year")):
        return "time"
    if any(k in t for k in ("bool",)):
        return "boolean"
    return "text"


def build_databases(tables_json: Sequence[dict[str, Any]],
                    db_dir: Path | None = None) -> dict[str, Database]:
    dbs: dict[str, Database] = {}
    for obj in tables_json:
        db = Database.from_spider_json(obj)
        if db_dir is not None:
            sqlite_path = db_dir / f"{db.db_id}.sqlite"
            if sqlite_path.exists():
                db.rows = load_rows(sqlite_path)
        dbs[db.db_id] = db
    return dbs


def load_rows(sqlite_path: Path, max_rows: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Read a few rows per table so the fixture writer can rebuild the DB offline."""
    import sqlite3

    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return rows
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = [r[0] for r in cur.fetchall()]
        for name in names:
            try:
                cur.execute(f'SELECT * FROM "{name}" LIMIT {int(max_rows)}')
                cols = [d[0] for d in cur.description]
                rows[name] = [dict(zip(cols, r)) for r in cur.fetchall()]
            except sqlite3.Error:
                continue
    finally:
        conn.close()
    return rows


def schema_tokens_hint(db: Database) -> int:
    """Rough token count of the DDL block, useful for prompt budgeting."""
    return max(1, len(db.render_ddl()) // 4)
