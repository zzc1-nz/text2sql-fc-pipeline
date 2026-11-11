"""Real SQLite execution: build DBs, run queries with a timeout, compare results.

This is the ground truth of the whole project. "Execution accuracy" is only
meaningful if it is computed by actually running SQL against a real database,
so this module never fakes a result:

* databases are real ``sqlite3`` files built from the schema + rows,
* generated SQL is really executed (with an interrupt-based timeout),
* result sets are compared as bags of rows (Spider convention: order matters
  only when the gold query has ``ORDER BY``).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import Database, Table


# --------------------------------------------------------------------------- #
# database construction
# --------------------------------------------------------------------------- #
_SQL_TYPE = {"number": "REAL", "time": "TEXT", "boolean": "INTEGER", "text": "TEXT"}


def ddl_for_db(db: Database) -> str:
    """Create-table DDL for the *executable* sqlite fixture."""
    stmts: list[str] = []
    for t in db.tables:
        cols = ", ".join(f'"{c.name}" {_SQL_TYPE.get(c.type, "TEXT")}' for c in t.columns)
        stmts.append(f'CREATE TABLE "{t.name}" ({cols});')
    return "\n".join(stmts)


def _fk_clauses(db: Database) -> dict[str, list[str]]:
    """FOREIGN KEY table constraints, keyed by child table name.

    They are declared in the DDL (not enforced at runtime) so that the schema is
    faithful and so ``PRAGMA foreign_key_list`` can recover the relations when a
    database is introspected back into a ``Database`` model.
    """
    out: dict[str, list[str]] = {}
    for fk in db.foreign_keys:
        try:
            child = db.tables[fk.src_table]
            parent = db.tables[fk.dst_table]
            child_col = child.columns[fk.src_col].name
            parent_col = parent.columns[fk.dst_col].name
        except (IndexError, KeyError):
            continue
        out.setdefault(child.name, []).append(
            f'FOREIGN KEY ("{child_col}") REFERENCES "{parent.name}" ("{parent_col}")')
    return out


def build_sqlite(db: Database, path: str | os.PathLike[str], overwrite: bool = True) -> Path:
    """Materialize ``db`` (schema + ``db.rows``) into a sqlite file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if not overwrite:
            return p
        p.unlink()
    fks = _fk_clauses(db)
    conn = sqlite3.connect(str(p))
    try:
        cur = conn.cursor()
        for t in db.tables:
            cols = ", ".join(f'"{c.name}" {_SQL_TYPE.get(c.type, "TEXT")}' for c in t.columns)
            constraints = "".join(", " + c for c in fks.get(t.name, []))
            cur.execute(f'CREATE TABLE "{t.name}" ({cols}{constraints})')
            rows = db.rows.get(t.name) or db.rows.get(t.name.lower()) or []
            if rows:
                names = [c.name for c in t.columns]
                ph = ", ".join("?" for _ in names)
                collist = ", ".join(f'"{n}"' for n in names)
                payload = [tuple(_coerce_value(r.get(n), c.type) for n, c in zip(names, t.columns))
                           for r in rows]
                cur.executemany(
                    f'INSERT INTO "{t.name}" ({collist}) VALUES ({ph})', payload)
        # indices on the first column of each table keep the fixtures snappy
        for t in db.tables:
            if t.columns:
                try:
                    cur.execute(f'CREATE INDEX "ix_{t.name}" ON "{t.name}" ("{t.columns[0].name}")')
                except sqlite3.Error:
                    pass
        conn.commit()
    finally:
        conn.close()
    return p


def _coerce_value(v: Any, col_type: str) -> Any:
    if v is None:
        return None
    if col_type == "number":
        try:
            f = float(v)
            return int(f) if f.is_integer() and abs(f) < 1e15 else f
        except (TypeError, ValueError):
            return None
    return v


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_ms: float = 0.0
    timed_out: bool = False

    @property
    def n_rows(self) -> int:
        return len(self.rows)


_READ_ONLY_PREFIX = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def is_safe_to_execute(sql: str) -> bool:
    """Only allow read-only statements; reject anything with a write keyword."""
    if not _READ_ONLY_PREFIX.match(sql or ""):
        return False
    # strip string literals before scanning for banned words
    stripped = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return not re.search(
        r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex)\b",
        stripped, re.IGNORECASE)


def execute_sql(db_path: str | os.PathLike[str], sql: str,
                timeout_s: float = 5.0, max_rows: int = 10_000) -> ExecResult:
    """Execute ``sql`` read-only with a wall-clock timeout.

    The timeout uses ``sqlite3.Connection.interrupt()`` from a watchdog thread,
    which is the only way to cancel a long-running sqlite query in-process.
    """
    import time

    t0 = time.perf_counter()
    if not sql or not sql.strip():
        return ExecResult(ok=False, error="empty sql")
    if not is_safe_to_execute(sql):
        return ExecResult(ok=False, error="unsafe or non-SELECT statement rejected")
    p = Path(db_path)
    if not p.exists():
        return ExecResult(ok=False, error=f"database not found: {p}")

    timed_out = False
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=timeout_s)
        fired = threading.Event()

        def _watch() -> None:
            nonlocal timed_out
            if not fired.wait(timeout_s) and conn is not None:
                timed_out = True
                try:
                    conn.interrupt()
                except sqlite3.Error:
                    pass

        watchdog = threading.Thread(target=_watch, daemon=True)
        watchdog.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            cols = [d[0] for d in (cur.description or [])]
        finally:
            fired.set()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if timed_out:
            return ExecResult(ok=False, error="timeout", elapsed_ms=elapsed, timed_out=True)
        return ExecResult(ok=True, rows=[tuple(r) for r in rows], columns=cols,
                          elapsed_ms=elapsed)
    except sqlite3.Error as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return ExecResult(ok=False, error=str(exc), elapsed_ms=elapsed,
                          timed_out=timed_out)
    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------------------------------- #
# introspection: sqlite file -> Spider-style Database
# --------------------------------------------------------------------------- #
def introspect_sqlite(path: str | os.PathLike[str],
                      db_id: str | None = None,
                      with_rows: bool = True,
                      max_rows: int = 60) -> Database:
    """Read a real sqlite file and reconstruct the Spider schema model.

    Used by the fixture generator: we author databases as plain ``CREATE TABLE``
    + ``INSERT`` scripts (so the data is genuinely relational), then derive
    ``tables.json`` from the file rather than hand-maintaining two copies of the
    schema. That guarantees the schema in prompts is the schema that executes.
    """
    from .schema import Column, Database, ForeignKey, Table, load_rows, normalize_type

    p = Path(path)
    db_id = db_id or p.stem
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY rowid")
        table_names = [r[0] for r in cur.fetchall()]
        tables: list[Table] = []
        for ti, tname in enumerate(table_names):
            cur.execute(f'PRAGMA table_info("{tname}")')
            info = cur.fetchall()  # cid, name, type, notnull, dflt_value, pk
            cols = [
                Column(table_index=ti, name=row[1],
                       type=normalize_type(row[2] or "text"),
                       is_primary_key=bool(row[5]))
                for row in info
            ]
            tables.append(Table(index=ti, name=tname, columns=cols))
        fks: list[ForeignKey] = []
        for ti, tname in enumerate(table_names):
            cur.execute(f'PRAGMA foreign_key_list("{tname}")')
            for row in cur.fetchall():  # id, seq, table, from, to, ...
                ref_table, from_col, to_col = row[2], row[3], row[4]
                dst_idx = next((i for i, t in enumerate(tables) if t.name == ref_table), None)
                if dst_idx is None:
                    continue
                src_col = next((i for i, c in enumerate(tables[ti].columns)
                                if c.name == from_col), None)
                dst_col = next((i for i, c in enumerate(tables[dst_idx].columns)
                                if c.name == (to_col or "id")), None)
                if src_col is None or dst_col is None:
                    continue
                fks.append(ForeignKey(ti, src_col, dst_idx, dst_col))
    finally:
        conn.close()

    db = Database(db_id=db_id, tables=tables, foreign_keys=fks)
    if with_rows:
        db.rows = load_rows(p, max_rows=max_rows)
    return db


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def _norm_cell(v: Any) -> Any:
    """Normalize a cell so that 1 == 1.0 == "1" compares equal."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip().lower()
    s = str(v).strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else round(f, 6)
    except ValueError:
        return s.lower()


def _bag(rows: Iterable[Sequence[Any]], order_sensitive: bool) -> list[tuple[Any, ...]]:
    norm = [tuple(_norm_cell(c) for c in r) for r in rows]
    if order_sensitive:
        return norm
    return sorted(norm, key=lambda r: tuple("" if c is None else str(c) for c in r))


@dataclass
class MatchResult:
    exec_match: bool
    reason: str = ""
    gold_ok: bool = True
    gold_error: str = ""
    pred_rows: int = 0
    gold_rows: int = 0


def result_match(pred: ExecResult, gold: ExecResult,
                 order_sensitive: bool = False) -> MatchResult:
    if not gold.ok:
        return MatchResult(False, f"gold query failed: {gold.error}", gold_ok=False,
                           gold_error=gold.error)
    if not pred.ok:
        return MatchResult(False, f"pred error: {pred.error}", gold_rows=gold.n_rows)
    a = _bag(pred.rows, order_sensitive)
    b = _bag(gold.rows, order_sensitive)
    if len(a) != len(b):
        return MatchResult(False, f"row count {len(a)} != {len(b)}",
                           pred_rows=len(a), gold_rows=len(b))
    if a != b:
        return MatchResult(False, "row values differ",
                           pred_rows=len(a), gold_rows=len(b))
    return MatchResult(True, "match", pred_rows=len(a), gold_rows=len(b))


def evaluate_pair(db_path: str | os.PathLike[str], pred_sql: str, gold_sql: str,
                  timeout_s: float = 5.0) -> dict[str, Any]:
    """Full single-pair evaluation: executability + execution match + EM."""
    from .sqlutils import has_limit, normalize_sql

    gold = execute_sql(db_path, gold_sql, timeout_s)
    order_sensitive = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE)) and has_limit(gold_sql)
    if not pred_sql.strip():
        return {"executable": False, "exec_match": False, "exact_match": False,
                "reason": "empty prediction", "gold_ok": gold.ok,
                "gold_error": gold.error}
    pred = execute_sql(db_path, pred_sql, timeout_s)
    m = result_match(pred, gold, order_sensitive=order_sensitive)
    em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
    return {
        "executable": pred.ok,
        "exec_match": bool(m.exec_match),
        "exact_match": bool(em),
        "reason": m.reason,
        "gold_ok": gold.ok,
        "gold_error": gold.error,
        "pred_rows": pred.n_rows,
        "gold_rows": gold.n_rows,
        "pred_error": pred.error,
        "order_sensitive": order_sensitive,
    }


# --------------------------------------------------------------------------- #
# corpus-level helpers
# --------------------------------------------------------------------------- #
class DatabaseRegistry:
    """Lazily opens/creates sqlite files for a set of Spider databases."""

    def __init__(self, db_dir: str | os.PathLike[str],
                 databases: dict[str, Database] | None = None):
        self.db_dir = Path(db_dir)
        self.databases = databases or {}

    def path(self, db_id: str) -> Path:
        return self.db_dir / f"{db_id}.sqlite"

    def ensure(self, db_id: str) -> Path:
        p = self.path(db_id)
        if p.exists():
            return p
        if db_id in self.databases:
            return build_sqlite(self.databases[db_id], p)
        raise FileNotFoundError(f"no sqlite for {db_id!r} at {p} and no schema to build it")

    def validate_all(self) -> dict[str, dict[str, Any]]:
        """Sanity check: every database builds and is queryable."""
        out: dict[str, dict[str, Any]] = {}
        for db_id, db in self.databases.items():
            try:
                p = self.ensure(db_id)
                res = execute_sql(p, f'SELECT COUNT(*) FROM "{db.tables[0].name}"' if db.tables else "")
                out[db_id] = {"ok": res.ok, "tables": len(db.tables),
                              "error": res.error}
            except Exception as exc:  # pragma: no cover
                out[db_id] = {"ok": False, "error": str(exc)}
        return out
