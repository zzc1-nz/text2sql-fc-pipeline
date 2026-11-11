"""Shared machinery for the synthesis stage.

Why a shared module
-------------------
All three augmentation kinds (CoT, paraphrase, hard negative) follow the same
skeleton:

1. pick source examples deterministically,
2. render a prompt that embeds the *real* schema DDL and the gold SQL,
3. fan the prompts out through :class:`fcdata.providers.AsyncBatchRunner`,
4. turn each ``Completion`` into a :class:`fcdata.records.SynthRecord` that carries
   its own token cost.

If each kind re-implemented that skeleton, token attribution would drift (one kind
counting prompt tokens, another counting the total) and the "token investment per
record" number that the whole project reports would stop being comparable across
kinds. So the skeleton lives here once.

Determinism contract
--------------------
* Source selection is a seeded shuffle of the *sorted* example ids, so the same
  config + dataset always selects the same examples in the same order.
* Record ids are **derived from content** (``synth_<kind>_<source>_<index>``)
  rather than random UUIDs. Re-running the pipeline therefore produces joinable
  artifacts: the validator's output for record X in run A lines up with record X
  in run B, which is what makes an ablation sweep comparable. Only ``created_at``
  (a real wall-clock timestamp, kept for provenance) differs between runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import Config
from ..records import Example, SynthRecord
from ..schema import Database
from ..spider import SpiderData
from ..sqlutils import normalize_sql
from ..tokens import TokenLedger
from ..providers.base import Completion
from ..providers.sqlread import DdlSchema, SqlFacts, parse_ddl, read_sql

#: stage name used for every ledger record written by this stage
SYNTH_STAGE = "synth"


@dataclass
class SourcePlan:
    """A source example together with everything the prompts need, precomputed."""

    example: Example
    db: Database
    ddl: str
    schema: DdlSchema
    facts: SqlFacts

    @property
    def db_id(self) -> str:
        return self.example.db_id

    @property
    def question(self) -> str:
        return self.example.question

    @property
    def gold_sql(self) -> str:
        return self.example.sql

    @property
    def difficulty(self) -> float:
        return self.example.difficulty


def build_sources(cfg: Config, spider_data: SpiderData,
                  examples: Sequence[Example] | None = None) -> list[SourcePlan]:
    """Select and prepare the source examples for synthesis.

    Selection is deliberately *seeded* and *sorted-then-shuffled*: sorting first
    removes any dependence on file order (so a regenerated fixture with the same
    content yields the same corpus), and the seeded shuffle gives the
    ``max_source_examples`` cap a representative slice instead of the first N rows.
    """
    pool = list(examples if examples is not None else spider_data.train)
    # drop examples whose database or SQL we cannot ground prompts in
    usable: list[Example] = []
    for ex in sorted(pool, key=lambda e: str(e.id)):
        db = spider_data.databases.get(ex.db_id)
        if db is None or not db.tables or not ex.question.strip() or not ex.sql.strip():
            continue
        usable.append(ex)
    if cfg.synth.max_source_examples and cfg.synth.max_source_examples > 0:
        rng = random.Random(cfg.synth.seed)
        rng.shuffle(usable)
        usable = usable[:cfg.synth.max_source_examples]
        usable.sort(key=lambda e: str(e.id))
    plans: list[SourcePlan] = []
    for ex in usable:
        db = spider_data.databases[ex.db_id]
        ddl = db.render_ddl()
        schema = parse_ddl(ddl)
        if schema.is_empty:
            continue
        plans.append(SourcePlan(example=ex, db=db, ddl=ddl, schema=schema,
                                facts=read_sql(ex.sql, schema)))
    return plans


def stable_rng(*parts: Any) -> random.Random:
    """Seeded RNG for planning decisions (kind choice, strategy rotation, ...).

    Uses a string hash of the parts rather than ``hash()`` (which Python salts per
    process) so plans are reproducible across runs and machines.
    """
    import hashlib

    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def record_id(kind: str, source_id: str, index: int) -> str:
    """Deterministic, joinable record id (see the module docstring)."""
    safe_source = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                          for ch in str(source_id))[:48]
    return f"synth-{kind}-{safe_source}-{index}"


def make_record(*, kind: str, plan: SourcePlan, index: int, question: str,
                completion: Completion, role: str, model: str,
                reasoning: str = "", original_question: str = "",
                bad_sql: str = "", error_kind: str = "",
                correction_note: str = "",
                meta: dict[str, Any] | None = None) -> SynthRecord:
    """Assemble a ``SynthRecord``, carrying the call's real token cost."""
    return SynthRecord(
        id=record_id(kind, plan.example.id, index),
        source_id=str(plan.example.id),
        db_id=plan.db_id,
        kind=kind,
        question=question,
        sql=plan.gold_sql,
        reasoning=reasoning,
        original_question=original_question,
        bad_sql=bad_sql,
        error_kind=error_kind,
        correction_note=correction_note,
        generator_role=role,
        generator_model=completion.model or model,
        prompt_tokens=int(completion.prompt_tokens),
        completion_tokens=int(completion.completion_tokens),
        difficulty=float(plan.difficulty),
        meta=dict(meta or {}),
    )


def failed_record_meta(completion: Completion) -> dict[str, Any]:
    """Meta payload for a record that could not be produced (never a silent drop)."""
    return {"ok": False, "error": completion.error,
            "cached": completion.cached}


def dedupe_questions(records: Iterable[SynthRecord]) -> tuple[list[SynthRecord], int]:
    """Drop records whose (kind, source) already produced the same question.

    Paraphrase is the kind at risk: two rewrite strategies can converge on the
    same sentence, and training on duplicates silently up-weights that phrasing.
    CoT and hard-negative records are keyed by their (unique) source, so they are
    unaffected in practice.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[SynthRecord] = []
    dropped = 0
    for rec in records:
        key = (rec.kind, rec.source_id, normalize_question(rec.question))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(rec)
    return out, dropped


def normalize_question(text: str) -> str:
    import re

    t = (text or "").lower().strip().rstrip("?.")
    t = re.sub(r"^(?:please|can you|could you|would you)\s+", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class KindReport:
    """Per-kind accounting, surfaced in the stage summary / report.

    The counters are disjoint and cover every planned unit exactly once, which is
    what makes ``planned == ok + failed + skipped_budget + dropped_invalid`` a
    checkable invariant in the tests and in the report.

    ``examples_skipped`` is separate on purpose: for ``hard_negative`` an example
    can be *unusable* (every error kind is a no-op, so no prompt is ever built)
    before any job exists. Folding that into ``dropped_invalid`` would break the
    invariant above and hide the difference between "the teacher failed" and
    "there was nothing to teach".
    """

    kind: str
    planned: int = 0
    ok: int = 0
    failed: int = 0
    skipped_budget: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dropped_duplicate: int = 0
    dropped_invalid: int = 0
    examples_skipped: int = 0
    examples_covered: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def accounted(self) -> int:
        """Planned jobs accounted for; equals ``planned`` when nothing is lost."""
        return (self.ok + self.failed + self.skipped_budget + self.dropped_invalid
                + self.dropped_duplicate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "planned": self.planned, "ok": self.ok,
            "failed": self.failed, "skipped_budget": self.skipped_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_invalid": self.dropped_invalid,
            "examples_skipped": self.examples_skipped,
            "examples_covered": self.examples_covered,
            "errors": self.errors[:5],
        }


def summarize(records: Sequence[SynthRecord], reports: Sequence[KindReport],
              ledger: TokenLedger) -> dict[str, Any]:
    """Stage summary: what was produced, what it cost, and where it came from."""
    by_kind: dict[str, int] = {}
    for rec in records:
        by_kind[rec.kind] = by_kind.get(rec.kind, 0) + 1
    return {
        "stage": SYNTH_STAGE,
        "records": len(records),
        "by_kind": by_kind,
        "kinds": [r.as_dict() for r in reports],
        "prompt_tokens": sum(r.prompt_tokens for r in reports),
        "completion_tokens": sum(r.completion_tokens for r in reports),
        "total_tokens": sum(r.total_tokens for r in reports),
        "ledger_spent_tokens": ledger.spent(SYNTH_STAGE),
        "ledger_spent_usd": round(ledger.spent_usd(SYNTH_STAGE), 6),
    }
