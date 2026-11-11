"""Convert a real Spider 1.0 release into the layout this pipeline expects.

The pipeline's dataset loader (``fcdata.spider.load_spider``) reads one directory::

    <root>/tables.json
    <root>/train.json
    <root>/dev.json
    <root>/databases/<db_id>.sqlite

Real Spider ships as::

    spider/
      tables.json
      train_spider.json
      train_others.json
      dev.json
      database/<db_id>/<db_id>.sqlite     <- nested, one directory per database

This script normalizes the second into the first so the same code path serves
both the bundled offline fixture and the real benchmark. It does not modify or
re-download anything; point it at an existing Spider checkout.

Usage
-----
    # from a downloaded Spider directory
    python scripts/prepare_spider.py --spider data/spider_raw --out data/spider

    # or let it fetch the official release first (needs network + unzip)
    python scripts/prepare_spider.py --download --out data/spider

Notes
-----
* Spider is distributed for research use; check its license before redistributing
  the database files. This script only reorganizes files you already have.
* ``train_others.json`` (the 1,034 extra training questions used in some papers)
  is concatenated into ``train.json`` when present, matching the common
  "train on 8,034" protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SPIDER_URL = "https://drive.google.com/uc?export=download&id=1iRDVHLr4mXzrIeoLrVDRW7dNqZcOEysg"
SPIDER_ALT_URL = "https://huggingface.co/datasets/AsadIsmail/spider/resolve/main/spider.zip"


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def download(dest: Path) -> Path:
    """Download and unzip the Spider release. Returns the extracted root."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "spider.zip"
    if not zip_path.exists():
        print(f"downloading Spider -> {zip_path}")
        try:
            urllib.request.urlretrieve(SPIDER_ALT_URL, zip_path)
        except Exception as exc:
            raise SystemExit(
                f"download failed ({exc}).\n"
                f"Download Spider manually from one of:\n"
                f"  {SPIDER_ALT_URL}\n"
                f"  https://yale-lily.github.io/spider\n"
                f"then run: python scripts/prepare_spider.py --spider <extracted_dir> --out data/spider")
    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # the zip usually contains a top-level "spider/" directory
    for candidate in [dest / "spider", dest]:
        if (candidate / "tables.json").exists():
            return candidate
    raise SystemExit(f"no tables.json found after extracting into {dest}")


def prepare(spider_root: Path, out: Path, include_others: bool = True) -> dict:
    """Reorganize a real Spider checkout into the flat pipeline layout."""
    spider_root = Path(spider_root)
    out = Path(out)
    tables = spider_root / "tables.json"
    if not tables.exists():
        raise SystemExit(f"{spider_root} does not look like Spider (no tables.json)")

    train_files = [spider_root / "train_spider.json"]
    if include_others:
        others = spider_root / "train_others.json"
        if others.exists():
            train_files.append(others)
    dev_file = spider_root / "dev.json"
    for f in train_files + [dev_file]:
        if not f.exists():
            raise SystemExit(f"missing required Spider file: {f}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tables, out / "tables.json")

    train: list[dict] = []
    for f in train_files:
        train.extend(_load(f))
    dev = _load(dev_file)
    # stable ids so runs are reproducible
    for i, ex in enumerate(train):
        ex.setdefault("id", f"train_{i:05d}")
    for i, ex in enumerate(dev):
        ex.setdefault("id", f"dev_{i:05d}")
    (out / "train.json").write_text(json.dumps(train, indent=1), encoding="utf-8")
    (out / "dev.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")

    # flatten database/<db_id>/<db_id>.sqlite -> databases/<db_id>.sqlite
    src_db = spider_root / "database"
    if not src_db.exists():
        src_db = spider_root / "databases"
    dst_db = out / "databases"
    dst_db.mkdir(parents=True, exist_ok=True)
    copied = 0
    if src_db.exists():
        for d in sorted(src_db.iterdir()):
            if d.is_dir():
                sqlite = d / f"{d.name}.sqlite"
                if sqlite.exists():
                    shutil.copy2(sqlite, dst_db / sqlite.name)
                    copied += 1
            elif d.suffix == ".sqlite":
                shutil.copy2(d, dst_db / d.name)
                copied += 1

    summary = {
        "out": str(out),
        "n_train": len(train),
        "n_dev": len(dev),
        "n_databases_copied": copied,
        "n_tables_json_entries": len(_load(tables)),
    }
    return summary


def verify(out: Path) -> int:
    """Load the prepared dataset through the pipeline loader and execute gold SQL."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from fcdata.spider import load_spider, verify_examples

    data = load_spider(out)
    print(json.dumps(data.stats(), indent=2))
    result = verify_examples(data)
    print(f"gold SQL: {result['ok']}/{result['checked']} execute")
    if result["failed"]:
        print("first failures:", json.dumps(result["failures"][:5], indent=2))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spider", default=None, help="existing Spider checkout")
    ap.add_argument("--download", action="store_true", help="download Spider first")
    ap.add_argument("--download-dir", default="data/spider_raw")
    ap.add_argument("--out", default="data/spider")
    ap.add_argument("--no-others", action="store_true",
                    help="skip train_others.json (7,000-question protocol instead of 8,034)")
    ap.add_argument("--verify", action="store_true", help="run gold SQL after preparing")
    args = ap.parse_args(argv)

    if args.download:
        root = download(Path(args.download_dir))
    elif args.spider:
        root = Path(args.spider)
    else:
        ap.error("pass --spider <dir> or --download")

    summary = prepare(root, Path(args.out), include_others=not args.no_others)
    print(json.dumps(summary, indent=2))
    if args.verify:
        return verify(Path(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
