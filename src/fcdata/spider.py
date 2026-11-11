"""Spider dataset loading.

Two sources are supported and they share one code path:

* **real Spider** -- point ``paths.spider_dir`` at a Spider 1.0 release and call
  ``load_spider(...)``; it reads ``train_spider.json``, ``dev.json``,
  ``tables.json`` and the ``database/<db_id>/<db_id>.sqlite`` files.
* **bundled fixture** -- ``data/fixture`` holds the exact same file layout
  (``tables.json``, ``train.json``, ``dev.json``, ``databases/*.sqlite``), so the
  entire pipeline can be exercised end-to-end offline with no download.

Nothing here is simulated: a fixture example is a real (question, gold SQL,
database) triple that really executes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .records import Example
from .schema import Database, build_databases, load_rows
from .sqlutils import difficulty_score, has_limit, normalize_sql


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tables(tables_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return _read_json(Path(tables_path))


def load_databases(tables_json: Sequence[dict[str, Any]],
                   sqlite_dir: str | os.PathLike[str] | None = None,
                   with_rows: bool = True) -> dict[str, Database]:
    """Build ``Database`` objects; attach sample rows from the sqlite files."""
    db_dir = Path(sqlite_dir) if sqlite_dir else None
    dbs: dict[str, Database] = {}
    for obj in tables_json:
        db = Database.from_spider_json(obj)
        if with_rows and db_dir is not None:
            p = db_dir / f"{db.db_id}.sqlite"
            if p.exists():
                db.rows = load_rows(p)
        dbs[db.db_id] = db
    return dbs


def _sqlite_path_candidates(root: Path, db_id: str) -> list[Path]:
    return [
        root / "databases" / f"{db_id}.sqlite",
        root / "database" / db_id / f"{db_id}.sqlite",
        root / "database" / f"{db_id}.sqlite",
        root / f"{db_id}.sqlite",
    ]


def find_sqlite(root: Path, db_id: str) -> Path | None:
    for p in _sqlite_path_candidates(root, db_id):
        if p.exists():
            return p
    return None


def load_examples(examples_path: str | os.PathLike[str], split: str,
                  databases: dict[str, Database] | None = None,
                  with_difficulty: bool = True) -> list[Example]:
    """Load a Spider ``*.json`` example file into ``Example`` records."""
    path = Path(examples_path)
    raw = _read_json(path)
    out: list[Example] = []
    for i, obj in enumerate(raw):
        q = (obj.get("question") or "").strip()
        sql = (obj.get("query") or obj.get("sql") or "").strip()
        db_id = obj.get("db_id") or ""
        if not q or not sql or not db_id:
            continue
        ex_id = obj.get("id") or f"{split}_{i:05d}"
        db = databases.get(db_id) if databases else None
        diff = difficulty_score(sql, db) if with_difficulty else 0.0
        out.append(Example(
            id=str(ex_id), db_id=db_id, question=q, sql=sql, split=split,
            difficulty=diff,
            meta={"db_tables": len(db.tables) if db else 0},
        ))
    return out


# --------------------------------------------------------------------------- #
# high level
# --------------------------------------------------------------------------- #
class SpiderData:
    """Everything the pipeline needs from a Spider-format directory."""

    def __init__(self, root: str | os.PathLike[str],
                 databases: dict[str, Database],
                 train: list[Example], dev: list[Example],
                 sqlite_dir: Path):
        self.root = Path(root)
        self.databases = databases
        self.train = train
        self.dev = dev
        self.sqlite_dir = sqlite_dir

    # -- convenience -----------------------------------------------------
    def db(self, db_id: str) -> Database:
        return self.databases[db_id]

    def db_path(self, db_id: str) -> Path:
        p = self.sqlite_dir / f"{db_id}.sqlite"
        if not p.exists():
            raise FileNotFoundError(f"missing sqlite for db {db_id!r} under {self.sqlite_dir}")
        return p

    def stats(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "n_databases": len(self.databases),
            "n_train": len(self.train),
            "n_dev": len(self.dev),
            "avg_difficulty_train": round(
                sum(e.difficulty for e in self.train) / max(1, len(self.train)), 4),
            "kind_counts_train": _count_kinds(self.train),
            "tables_total": sum(len(d.tables) for d in self.databases.values()),
        }


def _count_kinds(examples: Iterable[Example]) -> dict[str, int]:
    from .sqlutils import analyze_sql
    counts = {"join": 0, "aggregate": 0, "group_by": 0, "subquery": 0,
              "set_op": 0, "order_limit": 0, "single_table": 0}
    for e in examples:
        an = analyze_sql(e.sql)
        counts["join"] += int(an.has_join)
        counts["aggregate"] += int(an.has_aggregate)
        counts["group_by"] += int(an.has_group_by)
        counts["subquery"] += int(an.has_subquery)
        counts["set_op"] += int(an.has_set_op)
        counts["order_limit"] += int(an.has_order_by and an.has_limit)
        counts["single_table"] += int(not an.has_join and not an.has_subquery)
    return counts


def load_spider(root: str | os.PathLike[str],
                with_rows: bool = True,
                with_difficulty: bool = True) -> SpiderData:
    """Load a Spider-format directory (real Spider or the bundled fixture).

    Expected layout::

        <root>/tables.json
        <root>/train.json   (or train_spider.json)
        <root>/dev.json
        <root>/databases/<db_id>.sqlite   (or database/<db_id>/<db_id>.sqlite)
    """
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"dataset directory not found: {rootp}")
    tables_path = rootp / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"missing tables.json in {rootp}")
    tables_json = load_tables(tables_path)

    # locate sqlite dir: prefer flat databases/, else real Spider nested layout
    sqlite_dir = rootp / "databases"
    if not sqlite_dir.exists():
        sqlite_dir = rootp / "database"
    dbs = load_databases(tables_json, sqlite_dir, with_rows=with_rows)

    train_path = rootp / "train.json"
    if not train_path.exists():
        train_path = rootp / "train_spider.json"
    dev_path = rootp / "dev.json"

    train = (load_examples(train_path, "train", dbs, with_difficulty)
             if train_path.exists() else [])
    dev = (load_examples(dev_path, "dev", dbs, with_difficulty)
           if dev_path.exists() else [])
    return SpiderData(rootp, dbs, train, dev, sqlite_dir)


def verify_examples(data: SpiderData, examples: Sequence[Example] | None = None,
                    limit: int = 0) -> dict[str, Any]:
    """Execute every gold SQL; a non-executing gold query is a fixture bug."""
    from .dbexec import execute_sql

    exs = list(examples if examples is not None else data.train + data.dev)
    if limit:
        exs = exs[:limit]
    bad: list[dict[str, Any]] = []
    ok = 0
    for e in exs:
        try:
            p = data.db_path(e.db_id)
        except FileNotFoundError as exc:
            bad.append({"id": e.id, "error": str(exc)})
            continue
        res = execute_sql(p, e.sql)
        if res.ok:
            ok += 1
        else:
            bad.append({"id": e.id, "db_id": e.db_id, "sql": e.sql, "error": res.error})
    return {"checked": len(exs), "ok": ok, "failed": len(bad), "failures": bad[:50]}


# --------------------------------------------------------------------------- #
# splitting / sampling helpers used by ablation
# --------------------------------------------------------------------------- #
def stratified_split(examples: Sequence[Example], n: int, seed: int = 17) -> list[Example]:
    """Deterministic sample that keeps the difficulty distribution."""
    import random
    if n <= 0 or n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    buckets: dict[int, list[Example]] = {}
    for e in examples:
        buckets.setdefault(int(e.difficulty * 5), []).append(e)
    out: list[Example] = []
    keys = sorted(buckets)
    per = max(1, n // max(1, len(keys)))
    for k in keys:
        bucket = buckets[k]
        rng.shuffle(bucket)
        out.extend(bucket[:per])
    rng.shuffle(out)
    return out[:n]


def hard_subset(examples: Sequence[Example], fraction: float,
                seed: int = 17) -> list[Example]:
    """Top-``fraction`` by difficulty (the "hard data is scarce" set)."""
    if not examples:
        return []
    ranked = sorted(examples, key=lambda e: e.difficulty, reverse=True)
    k = max(1, int(len(ranked) * fraction))
    return ranked[:k]


def write_spider_dir(root: str | os.PathLike[str],
                     tables_json: Sequence[dict[str, Any]],
                     train: Sequence[dict[str, Any]],
                     dev: Sequence[dict[str, Any]],
                     databases: dict[str, Database]) -> None:
    """Serialize a fixture back into Spider's on-disk format."""
    rootp = Path(root)
    (rootp / "databases").mkdir(parents=True, exist_ok=True)
    (rootp / "tables.json").write_text(json.dumps(tables_json, indent=1), encoding="utf-8")
    (rootp / "train.json").write_text(json.dumps(train, indent=1), encoding="utf-8")
    (rootp / "dev.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")
    from .dbexec import build_sqlite
    for db_id, db in databases.items():
        build_sqlite(db, rootp / "databases" / f"{db_id}.sqlite")
