"""Build the offline Spider-format fixture dataset from ``fcdata.fixtures``.

Why this exists: the real Spider release is a ~400MB download and this
environment has no network. Rather than fake a benchmark, we ship a small but
*genuinely executable* Spider-format dataset so that every number the pipeline
reports (execution accuracy, exact match) is computed by running real SQL
against real SQLite databases.

The generated layout is byte-compatible with Spider 1.0::

    data/fixture/tables.json
    data/fixture/train.json
    data/fixture/dev.json
    data/fixture/databases/<db_id>.sqlite

Swapping in the real Spider release is then a one-line config change
(``paths.spider_dir``) -- see README.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

from .dbexec import build_sqlite, execute_sql, introspect_sqlite
from .fixtures import FIXTURES
from .schema import Database
from .spider import write_spider_dir


def materialize_fixtures(work_dir: Path) -> dict[str, Database]:
    """Run each fixture's DDL+INSERT script into a real sqlite file, then
    introspect it back into a ``Database`` model."""
    work_dir.mkdir(parents=True, exist_ok=True)
    dbs: dict[str, Database] = {}
    for fx in FIXTURES:
        path = work_dir / f"{fx.db_id}.sqlite"
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript(fx.ddl)
            conn.commit()
        finally:
            conn.close()
        db = introspect_sqlite(path, db_id=fx.db_id, with_rows=True, max_rows=80)
        dbs[fx.db_id] = db
    return dbs


def check_examples(dbs: dict[str, Database], work_dir: Path,
                   verbose: bool = True) -> list[dict]:
    """Execute every gold SQL; report failures. A gold query that does not run
    is a fixture bug and must be fixed, never silently dropped."""
    failures: list[dict] = []
    total = 0
    for fx in FIXTURES:
        db_path = work_dir / f"{fx.db_id}.sqlite"
        for q, sql in fx.examples:
            total += 1
            res = execute_sql(db_path, sql, timeout_s=5.0)
            if not res.ok:
                failures.append({"db_id": fx.db_id, "question": q, "sql": sql,
                                 "error": res.error})
            elif verbose:
                print(f"  ok  {fx.db_id:12s} rows={res.n_rows:<3d} {q[:64]}")
    if failures:
        print(f"\n!! {len(failures)}/{total} gold queries FAILED", file=sys.stderr)
        for f in failures:
            print(f"   [{f['db_id']}] {f['sql']}\n     -> {f['error']}", file=sys.stderr)
    else:
        print(f"\nAll {total} gold queries execute successfully.")
    return failures


def build(out_dir: Path, dev_ratio: float = 0.2, seed: int = 17) -> dict:
    """Generate the fixture directory and return a build summary."""
    from .config import scratch_dir

    out_dir = Path(out_dir)
    tmp = scratch_dir("fixture_build")
    dbs = materialize_fixtures(tmp)

    failures = check_examples(dbs, tmp, verbose=False)
    if failures:
        raise SystemExit(
            f"{len(failures)} fixture gold queries do not execute; fix fcdata/fixtures.py "
            f"before generating the dataset.")

    rng = random.Random(seed)
    train: list[dict] = []
    dev: list[dict] = []
    for fx in FIXTURES:
        items = list(fx.examples)
        rng.shuffle(items)
        n_dev = max(1, round(len(items) * dev_ratio))
        dev_items = items[:n_dev]
        train_items = items[n_dev:]
        for i, (q, sql) in enumerate(train_items):
            train.append({"db_id": fx.db_id, "question": q, "query": sql,
                          "id": f"{fx.db_id}_train_{i:03d}"})
        for i, (q, sql) in enumerate(dev_items):
            dev.append({"db_id": fx.db_id, "question": q, "query": sql,
                        "id": f"{fx.db_id}_dev_{i:03d}"})

    rng.shuffle(train)
    rng.shuffle(dev)
    tables_json = [db.to_spider_json() for db in dbs.values()]
    write_spider_dir(out_dir, tables_json, train, dev, dbs)

    summary = {
        "out_dir": str(out_dir),
        "databases": len(dbs),
        "train": len(train),
        "dev": len(dev),
        "tables": sum(len(d.tables) for d in dbs.values()),
        "columns": sum(len(d.tables and [c for t in d.tables for c in t.columns]) for d in dbs.values()),
        "rows": sum(len(r) for d in dbs.values() for r in d.rows.values()),
        "db_ids": sorted(dbs),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/fixture")
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--check", action="store_true", help="only verify gold SQL execution")
    args = ap.parse_args(argv)

    if args.check:
        from .config import scratch_dir
        tmp = scratch_dir("fixture_check")
        dbs = materialize_fixtures(tmp)
        return 1 if check_examples(dbs, tmp) else 0

    summary = build(Path(args.out), args.dev_ratio, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
