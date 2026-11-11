"""Provider abstraction: one ``complete()`` call == exactly one ledger record.

Design notes
------------
**Why an ABC with a split ``complete`` / ``_generate``.**
The contract requires that *every* call records exactly one ``UsageRecord``.
That invariant is easiest to guarantee structurally, not by discipline: the
public :meth:`LLMProvider.complete` is a template method that owns timing, token
accounting and ledger writing, while subclasses only implement
:meth:`LLMProvider._generate` (pure text production). A new provider therefore
cannot forget to account for its calls, and cannot accidentally double-count
either, because the accounting path is not overridable.

**Why errors become ``Completion(ok=False)`` instead of exceptions.**
Synthesis fans out thousands of calls. One flaky HTTP call must not kill the
stage, so failures are *values*: ``ok=False`` plus an ``error`` string. The batch
runner counts them, the caller decides whether to drop the sample. Unexpected
exceptions raised by a provider implementation are caught at the same boundary
and converted, so a provider bug degrades the stage instead of aborting it --
but the error text always carries the exception type, never a silent swallow.

**Why ``Job`` exists.**
The runner needs to know the task/sample identity *before* the call to enforce
budgets and to keep output ordering stable, so the unit of work is a small
dataclass rather than a bare message list.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..tokens import (TokenLedger, UsageRecord, estimate_messages_tokens,
                      estimate_tokens)

# --------------------------------------------------------------------------- #
# result / work units
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    """The result of one provider call, including its own cost accounting."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    error: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ok": self.ok,
            "error": self.error,
            "cached": self.cached,
            "latency_ms": round(self.latency_ms, 3),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


@dataclass
class Job:
    """One unit of provider work."""

    messages: list[dict[str, Any]]
    task: str = ""
    sample_id: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_preview(self) -> str:
        """First user turn, truncated -- only for progress logging."""
        for m in self.messages:
            if m.get("role") == "user":
                return str(m.get("content", ""))[:120]
        return ""


# --------------------------------------------------------------------------- #
# provider base
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Base class for every LLM backend.

    Subclasses implement :meth:`_generate` only; :meth:`complete` is the single
    accounting boundary.
    """

    #: short backend kind, e.g. ``"mock"`` / ``"openai"``
    kind: str = "base"

    def __init__(self, *, name: str, model: str, role: str = "cheap",
                 ledger: TokenLedger | None = None, stage: str = "",
                 temperature: float = 0.7, max_tokens: int = 512,
                 prices: tuple[float, float] | None = None) -> None:
        self.name = name
        self.model = model
        self.role = role
        self.ledger = ledger
        self.stage = stage
        self.default_temperature = float(temperature)
        self.default_max_tokens = int(max_tokens)
        self.prices = prices
        self.n_calls = 0
        # Keep the ledger able to price this role even when the caller built the
        # ledger without a price table (cost columns must never silently be 0
        # for a paid provider).
        if ledger is not None and prices is not None:
            ledger.prices.setdefault(role, prices)

    # -- public API ------------------------------------------------------
    def complete(self, messages: Sequence[Mapping[str, Any]], *,
                 task: str = "", sample_id: str = "",
                 temperature: float | None = None, max_tokens: int | None = None,
                 json_mode: bool = False, stage: str = "") -> Completion:
        """Run one completion and record exactly one ``UsageRecord``."""
        msgs = [dict(m) for m in messages]
        temp = self.default_temperature if temperature is None else float(temperature)
        mtok = self.default_max_tokens if max_tokens is None else int(max_tokens)
        t0 = time.perf_counter()
        try:
            comp = self._generate(msgs, task=task, sample_id=sample_id,
                                  temperature=temp, max_tokens=mtok,
                                  json_mode=json_mode)
        except Exception as exc:  # noqa: BLE001 - provider faults must not abort a stage
            comp = Completion(ok=False,
                              error=f"{type(exc).__name__}: {exc}",
                              model=self.model)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if comp.latency_ms <= 0.0:
            comp.latency_ms = elapsed_ms
        if not comp.model:
            comp.model = self.model
        # Fall back to the deterministic estimator when a backend reports no
        # usage (mock, or an API that omits ``usage``); never leave 0 tokens on
        # a successful call, because the whole project reports token investment.
        if comp.ok:
            if comp.prompt_tokens <= 0:
                comp.prompt_tokens = estimate_messages_tokens(msgs)
            if comp.completion_tokens <= 0:
                comp.completion_tokens = estimate_tokens(comp.text)
        self._record(comp, task=task, sample_id=sample_id,
                     stage=stage or self.stage or "adhoc")
        self.n_calls += 1
        return comp

    # -- accounting ------------------------------------------------------
    def _record(self, comp: Completion, *, task: str, sample_id: str,
                stage: str) -> None:
        if self.ledger is None:
            return
        self.ledger.record(UsageRecord(
            stage=stage, role=self.role, model=comp.model or self.model,
            prompt_tokens=int(comp.prompt_tokens),
            completion_tokens=int(comp.completion_tokens),
            task=task, ok=bool(comp.ok), cached=bool(comp.cached),
            latency_ms=float(comp.latency_ms), sample_id=sample_id,
        ))

    # -- subclass hook ---------------------------------------------------
    @abstractmethod
    def _generate(self, messages: list[dict[str, Any]], *, task: str,
                  sample_id: str, temperature: float, max_tokens: int,
                  json_mode: bool) -> Completion:
        """Produce a completion. Must not touch the ledger or measure time."""

    # -- misc ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "model": self.model,
                "role": self.role, "calls": self.n_calls}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} role={self.role} model={self.model}>"
