"""Token accounting: estimation, per-call usage records, and the stage ledger.

The whole point of the project is the sentence *"token investment in synthesis
turns into downstream capability"*, so token accounting must be first-class,
append-only and impossible to accidentally double count.

Two layers:

``estimate_tokens``
    Cheap deterministic estimate used by the mock provider and as a fallback when
    an API response omits ``usage``. Prefers ``tiktoken`` when installed.

``TokenLedger``
    Thread/async-safe JSONL sink plus in-memory aggregates keyed by
    (stage, role, model). Every provider call must ``record()`` exactly once.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _tiktoken_len(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:  # pragma: no cover
    _ENC = None

    def _tiktoken_len(text: str) -> int:  # type: ignore[misc]
        raise RuntimeError("tiktoken unavailable")


def _heuristic_len(text: str) -> int:
    """Byte-pair-ish heuristic good enough for budgeting (+-10% on English).

    * CJK characters cost ~1 token each.
    * Latin text costs ~1 token per 4 characters.
    * Digits / punctuation are denser than prose, so count them at 1 per 3.
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF:
            cjk += 1
        else:
            other += 1
    return int(cjk + other / 4.0 + 0.5)


def estimate_tokens(text: str) -> int:
    """Deterministic token count for a string (>=1 for non-empty strings)."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return max(1, _tiktoken_len(text))
        except Exception:
            pass
    return max(1, _heuristic_len(text))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Approximate chat prompt cost including per-message framing overhead."""
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content", "")))
        total += 4  # role/formatting overhead per message
        for call in m.get("tool_calls") or []:  # assistant tool-call turns
            total += estimate_tokens(json.dumps(call, ensure_ascii=False))
    return total + 3


# --------------------------------------------------------------------------- #
# usage records
# --------------------------------------------------------------------------- #
@dataclass
class UsageRecord:
    stage: str                 # synth | validate | train | eval | ...
    role: str                  # cheap | judge | trainer
    model: str
    prompt_tokens: int
    completion_tokens: int
    task: str = ""             # cot | paraphrase | hard_negative | judge | ...
    ok: bool = True
    cached: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    sample_id: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def price_usage(prompt_tokens: int, completion_tokens: int,
                price_in_per_mtok: float, price_out_per_mtok: float) -> float:
    return (prompt_tokens / 1_000_000.0) * price_in_per_mtok + (
        completion_tokens / 1_000_000.0) * price_out_per_mtok


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
class TokenLedger:
    """Append-only token ledger with in-memory aggregation.

    Thread-safe: providers are called from a thread pool by default.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None,
                 prices: dict[str, tuple[float, float]] | None = None,
                 reset_file: bool = False):
        self.path = Path(path) if path is not None else None
        self.prices = prices or {}
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if reset_file:
                self.path.write_text("", encoding="utf-8")

    # -- writing ---------------------------------------------------------
    def record(self, rec: UsageRecord) -> UsageRecord:
        if rec.cost_usd == 0.0:
            pin, pout = self.prices.get(rec.role, (0.0, 0.0))
            rec.cost_usd = price_usage(rec.prompt_tokens, rec.completion_tokens, pin, pout)
        with self._lock:
            self._records.append(rec)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
        return rec

    def add(self, stage: str, role: str, model: str, prompt_tokens: int,
            completion_tokens: int, **kwargs: Any) -> UsageRecord:
        return self.record(UsageRecord(stage=stage, role=role, model=model,
                                       prompt_tokens=int(prompt_tokens),
                                       completion_tokens=int(completion_tokens),
                                       **kwargs))

    # -- reading ---------------------------------------------------------
    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def aggregate(self) -> dict[str, dict[str, Any]]:
        """Aggregate by stage, then by role/model inside the stage."""
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            records = list(self._records)
        for r in records:
            st = out.setdefault(r.stage, {
                "calls": 0, "ok_calls": 0, "cached_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "cost_usd": 0.0, "by_role": defaultdict(lambda: {
                    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0, "models": set(),
                }),
            })
            st["calls"] += 1
            st["ok_calls"] += int(r.ok)
            st["cached_calls"] += int(r.cached)
            st["prompt_tokens"] += r.prompt_tokens
            st["completion_tokens"] += r.completion_tokens
            st["total_tokens"] += r.total_tokens
            st["cost_usd"] += r.cost_usd
            br = st["by_role"][r.role]
            br["calls"] += 1
            br["prompt_tokens"] += r.prompt_tokens
            br["completion_tokens"] += r.completion_tokens
            br["total_tokens"] += r.total_tokens
            br["cost_usd"] += r.cost_usd
            br["models"].add(r.model)
        # json-safe
        for st in out.values():
            st["cost_usd"] = round(st["cost_usd"], 6)
            st["by_role"] = {k: {**v, "models": sorted(v["models"]),
                                 "cost_usd": round(v["cost_usd"], 6)}
                             for k, v in st["by_role"].items()}
        return out

    def totals(self) -> dict[str, Any]:
        agg = self.aggregate()
        return {
            "stages": len(agg),
            "calls": sum(s["calls"] for s in agg.values()),
            "total_tokens": sum(s["total_tokens"] for s in agg.values()),
            "prompt_tokens": sum(s["prompt_tokens"] for s in agg.values()),
            "completion_tokens": sum(s["completion_tokens"] for s in agg.values()),
            "cost_usd": round(sum(s["cost_usd"] for s in agg.values()), 6),
        }

    def stage_totals(self) -> dict[str, int]:
        return {k: v["total_tokens"] for k, v in self.aggregate().items()}

    def dump(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "totals": self.totals(),
            "by_stage": self.aggregate(),
        }, indent=2), encoding="utf-8")

    # -- budget ----------------------------------------------------------
    def spent(self, stage: str | None = None) -> int:
        with self._lock:
            records = list(self._records)
        if stage is None:
            return sum(r.total_tokens for r in records)
        return sum(r.total_tokens for r in records if r.stage == stage)

    def spent_usd(self, stage: str | None = None) -> float:
        with self._lock:
            records = list(self._records)
        if stage is None:
            return sum(r.cost_usd for r in records)
        return sum(r.cost_usd for r in records if r.stage == stage)


def load_ledger_jsonl(path: str | os.PathLike[str]) -> list[UsageRecord]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[UsageRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        d.pop("total_tokens", None)
        out.append(UsageRecord(**d))
    return out


def summarize_ledger(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Aggregate a ledger file without instantiating a full ledger."""
    recs = load_ledger_jsonl(path)
    ledger = TokenLedger(None)
    for r in recs:
        ledger.record(r)
    return {"totals": ledger.totals(), "by_stage": ledger.aggregate()}
