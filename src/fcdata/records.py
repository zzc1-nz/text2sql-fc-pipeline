"""Shared record types that flow through the pipeline.

Stage flow::

    Spider Example
        --synth-->      SynthRecord   (cot | paraphrase | hard_negative)
        --validate-->   ValidatedRecord  (adds ValidationReport)
        --format-->     SFTRecord     (chat messages + tool schema + labels)
        --train-->      (loss curves)
        --eval-->       Prediction

Every stage writes JSONL of its own record type into ``artifacts/NN_*/`` so any
stage can be re-run independently and the token ledger can be joined back to
individual samples.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# stage 0: raw Spider example
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    id: str
    db_id: str
    question: str
    sql: str
    split: str = "train"
    difficulty: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# stage 1: synthesis output
# --------------------------------------------------------------------------- #
SYNTH_KINDS = ("cot", "paraphrase", "hard_negative")


@dataclass
class SynthRecord:
    id: str
    source_id: str
    db_id: str
    kind: str                       # cot | paraphrase | hard_negative
    question: str                   # the question as it should appear in SFT
    sql: str                        # the target SQL
    # kind-specific payload ------------------------------------------------
    reasoning: str = ""             # CoT: explanation of the query
    original_question: str = ""     # paraphrase: the Spider question it rewrote
    bad_sql: str = ""               # hard_negative: the wrong SQL shown to the model
    error_kind: str = ""            # hard_negative: category of the injected error
    correction_note: str = ""       # hard_negative: why bad_sql is wrong
    # provenance / accounting ---------------------------------------------
    generator_role: str = "cheap"
    generator_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    difficulty: float = 0.0
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# stage 2: validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    # format stage
    sql_parses: bool = False
    schema_refs_ok: bool = False
    executes: bool = False
    exec_error: str = ""
    issue_codes: list[str] = field(default_factory=list)
    # semantic stage
    semantic_checked: bool = False
    semantic_score: float = 0.0
    semantic_verdict: str = ""      # consistent | inconsistent | uncertain
    semantic_rationale: str = ""
    # accounting
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    # outcome
    passed: bool = False
    reason: str = ""

    @property
    def judge_tokens(self) -> int:
        return self.judge_prompt_tokens + self.judge_completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatedRecord:
    record: SynthRecord
    report: ValidationReport
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id("val")

    @property
    def kind(self) -> str:
        return self.record.kind

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "record": self.record.as_dict(),
                "report": self.report.as_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidatedRecord":
        return cls(record=SynthRecord(**d["record"]),
                   report=ValidationReport(**d["report"]), id=d.get("id", ""))


# --------------------------------------------------------------------------- #
# stage 3: SFT / function-calling format
# --------------------------------------------------------------------------- #
@dataclass
class ToolSchema:
    """A JSON-schema tool definition, as an OpenAI-compatible function."""
    name: str = "execute_sql"
    description: str = "Execute a read-only SQLite query against the database."
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "A single SQLite SELECT statement."},
        },
        "required": ["sql"],
    })

    def as_openai_tool(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class SFTRecord:
    id: str
    source_id: str
    kind: str                       # raw | cot | paraphrase | hard_negative
    db_id: str
    messages: list[dict[str, Any]]  # full chat including assistant target
    tools: list[dict[str, Any]] = field(default_factory=list)
    target_sql: str = ""
    split: str = "train"
    weight: float = 1.0             # sampling weight (hard-negative resampling)
    difficulty: float = 0.0
    synth_tokens: int = 0           # token investment attributable to this record
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def n_tokens_est(self) -> int:
        from .tokens import estimate_messages_tokens
        return estimate_messages_tokens(self.messages)


# --------------------------------------------------------------------------- #
# stage 4/5: predictions
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    id: str
    example_id: str
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    raw_output: str = ""
    executable: bool = False
    exec_match: bool = False
    exact_match: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# JSONL IO helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            d = r.as_dict() if hasattr(r, "as_dict") else r
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    def _gen() -> Iterator[dict[str, Any]]:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    return _gen()


def read_synth_records(path: str | Path) -> list[SynthRecord]:
    return [SynthRecord(**d) for d in read_jsonl(path)]


def read_validated(path: str | Path) -> list[ValidatedRecord]:
    return [ValidatedRecord.from_dict(d) for d in read_jsonl(path)]


def read_sft(path: str | Path) -> list[SFTRecord]:
    return [SFTRecord(**d) for d in read_jsonl(path)]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))
