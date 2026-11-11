"""Check extraction of the reference model's SPACED tool-call output (dev tool).

The reference model emits ``< tool_call >`` (spaced) because the word tokenizer
round-trips the template markers that way. This checks which parser produces
executable SQL from that shape.

Run:  $env:PYTHONPATH='src'; python scripts/check_extraction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fcdata.format import assistant_sql  # noqa: E402
from fcdata.sqlutils import extract_sql, normalize_sql  # noqa: E402

try:
    from fcdata.eval.infer import extract_pred_sql  # noqa: E402
except ImportError:                                   # pragma: no cover
    extract_pred_sql = None  # type: ignore[assignment]

SQL = "SELECT title FROM student WHERE gpa > 3.5"

CASES = {
    "spaced_full": '< tool_call > {"name":"execute_sql","arguments":{"sql":"' + SQL
                   + '"}} < / tool_call > <|im_end|>',
    "spaced_closers": '< tool_call > {"name":"execute_sql","arguments":{"sql":"' + SQL
                      + '"}}}} < / tool_call > <|im_end|>',
    "compact": '<tool_call>\n{"name": "execute_sql", "arguments": {"sql": "'
               + SQL + '"}}\n</tool_call><|im_end|>',
    "quote_literal_end": "SELECT dept_name FROM department WHERE dept_name = 'Physics'",
    "paren_end": "SELECT COUNT(*) FROM student",
}


def is_clean(sql: str) -> bool:
    """True when nothing but a SQL statement remains."""
    if not sql:
        return False
    norm = normalize_sql(sql)
    if not norm.startswith(("select", "with")):
        return False
    junk = ("tool_call", "im_end", "eot_id", "```")
    if any(j in norm for j in junk):
        return False
    # balanced closers
    return (norm.count("(") == norm.count(")")
            and norm.count("'") % 2 == 0)


def main() -> int:
    print(f"extract_pred_sql available: {extract_pred_sql is not None}\n")
    print(f"{'case':<20}{'extract_sql':<14}{'assistant_sql':<16}{'extract_pred':<14}")
    print("-" * 66)
    for name, text in CASES.items():
        ex = extract_sql(text)
        asql = assistant_sql({"role": "assistant", "content": text}) or ""
        pred = extract_pred_sql(text) if extract_pred_sql else "(n/a)"
        print(f"{name:<20}{('clean' if is_clean(ex) else 'DIRTY'):<14}"
              f"{('clean' if is_clean(asql) else 'DIRTY'):<16}"
              f"{('clean' if is_clean(pred) else 'DIRTY'):<14}")
        print(f"    extract_sql  -> {ex[:95]!r}")
        print(f"    assistant_sql-> {asql[:95]!r}")
        if extract_pred_sql:
            print(f"    extract_pred -> {pred[:95]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
