"""Fan-out of many provider calls: concurrency, isolation, budget, ordering.

Why a dedicated runner
----------------------
Synthesis issues thousands of independent calls. Three things must hold:

1. **One failure must not sink the batch.** A 429 that survives retries, a
   truncated response, or a provider bug on sample #471 must cost exactly that
   sample. :meth:`AsyncBatchRunner.run` therefore always returns a list aligned
   1:1 with the jobs, with failures represented as ``Completion(ok=False)``.
2. **The budget must be a hard stop.** ``cfg.synth.max_total_tokens`` /
   ``max_cost_usd`` are enforced *before* dispatch, against the live ledger, so a
   runaway sweep cannot spend past the cap. Skipped jobs come back with
   ``ok=False`` and ``error="budget_exhausted"`` -- visible in the artifacts, not
   silently dropped.
3. **Ordering must be stable.** Results are returned in job order regardless of
   completion order, so downstream record construction is deterministic.

The runner is synchronous from the caller's point of view (``run`` blocks) but
executes on a thread pool. Threads, not ``asyncio``: the provider interface is
synchronous and the OpenAI client keeps one pooled connection per thread, which
is simpler to reason about than bridging an event loop for a CPU-bound pipeline.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import Config, ProviderCfg
from ..tokens import TokenLedger, estimate_messages_tokens
from .base import Completion, Job, LLMProvider

#: error marker for jobs that were never dispatched because the budget was spent
BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class BatchStats:
    """What actually happened, for the run summary / report."""

    stage: str = ""
    jobs: int = 0
    dispatched: int = 0
    ok: int = 0
    failed: int = 0
    cached: int = 0
    skipped_budget: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok_rate(self) -> float:
        return self.ok / self.dispatched if self.dispatched else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage, "jobs": self.jobs,
            "dispatched": self.dispatched, "ok": self.ok, "failed": self.failed,
            "cached": self.cached, "skipped_budget": self.skipped_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_ms": round(self.wall_ms, 1),
            "ok_rate": round(self.ok_rate, 4),
            "errors": self.errors[:10],
        }


class BudgetExhausted(RuntimeError):
    """Raised internally when the token/cost cap would be exceeded."""


class AsyncBatchRunner:
    """Run many ``Job``s through a provider with bounded concurrency."""

    def __init__(self, provider: LLMProvider, ledger: TokenLedger, cfg: Config,
                 *, stage: str = "synth", concurrency: int | None = None,
                 max_total_tokens: int | None = None,
                 max_cost_usd: float | None = None) -> None:
        self.provider = provider
        self.ledger = ledger
        self.cfg = cfg
        self.stage = stage
        pcfg: ProviderCfg = cfg.provider(provider.role)
        self.concurrency = max(1, int(concurrency or pcfg.concurrency))
        # explicit arguments win; otherwise fall back to the synth budget
        self.max_total_tokens = (int(max_total_tokens) if max_total_tokens is not None
                                 else int(cfg.synth.max_total_tokens))
        self.max_cost_usd = (float(max_cost_usd) if max_cost_usd is not None
                             else float(cfg.synth.max_cost_usd))
        self.stats = BatchStats(stage=stage)
        self._lock = threading.Lock()
        self._stopped = False
        #: tokens promised to in-flight calls but not yet in the ledger
        self._reserved_tokens = 0

    # -- budget ----------------------------------------------------------
    def budget_remaining(self) -> int | None:
        """Tokens still available, or ``None`` when unlimited."""
        if self.max_total_tokens <= 0:
            return None
        with self._lock:
            reserved = self._reserved_tokens
        return max(0, self.max_total_tokens - self.ledger.spent(self.stage) - reserved)

    def cost_remaining(self) -> float | None:
        if self.max_cost_usd <= 0:
            return None
        return max(0.0, self.max_cost_usd - self.ledger.spent_usd(self.stage))

    def _estimate_call(self, job: Job) -> int:
        """Upper-bound token cost of one call (prompt + the completion cap)."""
        prompt = estimate_messages_tokens(job.messages)
        cap = int(job.max_tokens if job.max_tokens is not None
                  else self.provider.default_max_tokens)
        return prompt + max(0, cap)

    def _reserve(self, job: Job) -> bool:
        """Reserve budget for one call; ``False`` means "do not dispatch".

        Reservation (rather than a check against the ledger alone) is what makes
        the cap hold under concurrency: N in-flight calls have not recorded usage
        yet, so a naive check would let the batch overshoot by N calls' worth of
        tokens. The estimate is prompt + ``max_tokens``, i.e. a true upper bound,
        so the batch may finish slightly *under* the cap but never over it.
        """
        with self._lock:
            if self._stopped:
                return False
            if self.max_total_tokens > 0:
                remaining = (self.max_total_tokens
                             - self.ledger.spent(self.stage) - self._reserved_tokens)
                if self._estimate_call(job) > remaining:
                    self._stopped = True
                    return False
            if self.max_cost_usd > 0 and self.ledger.spent_usd(self.stage) >= self.max_cost_usd:
                self._stopped = True
                return False
            self._reserved_tokens += self._estimate_call(job)
            return True

    def _release(self, tokens: int) -> None:
        with self._lock:
            self._reserved_tokens = max(0, self._reserved_tokens - tokens)

    # -- execution -------------------------------------------------------
    def run(self, jobs: Sequence[Job], stage: str = "") -> list[Completion]:
        """Execute ``jobs``; returns one ``Completion`` per job, in job order."""
        stage = stage or self.stage
        self.stage = stage
        self.stats.stage = stage
        self.stats.jobs += len(jobs)
        if not jobs:
            return []
        results: list[Completion | None] = [None] * len(jobs)
        t0 = time.perf_counter()

        def _work(index: int, job: Job) -> tuple[int, Completion]:
            reserved = self._estimate_call(job)
            if not self._reserve(job):
                comp = Completion(ok=False, error=BUDGET_EXHAUSTED,
                                  model=self.provider.model, meta={"skipped": True})
                self._account(comp)
                return index, comp
            try:
                comp = self.provider.complete(
                    job.messages, task=job.task, sample_id=job.sample_id,
                    temperature=job.temperature, max_tokens=job.max_tokens,
                    json_mode=job.json_mode, stage=stage)
            finally:
                self._release(reserved)
            self._account(comp)
            return index, comp

        workers = min(self.concurrency, len(jobs))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix=f"llm-{stage}") as pool:
            futures: dict[Future[tuple[int, Completion]], int] = {
                pool.submit(_work, i, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    index, comp = fut.result()
                except Exception as exc:  # noqa: BLE001 - isolate a broken job
                    index = idx
                    comp = Completion(ok=False, model=self.provider.model,
                                      error=f"worker crashed: {type(exc).__name__}: {exc}")
                    self._account(comp)
                results[index] = comp

        self.stats.wall_ms += (time.perf_counter() - t0) * 1000.0
        return [c if c is not None else
                Completion(ok=False, error="job produced no result",
                           model=self.provider.model)
                for c in results]

    def _should_skip(self, job: Job) -> bool:
        with self._lock:
            if self._stopped:
                return True
            if self._would_exceed(job):
                self._stopped = True
                return True
        return False

    def _account(self, comp: Completion) -> None:
        with self._lock:
            self.stats.prompt_tokens += comp.prompt_tokens
            self.stats.completion_tokens += comp.completion_tokens
            if comp.error == BUDGET_EXHAUSTED:
                self.stats.skipped_budget += 1
                return
            self.stats.dispatched += 1
            if comp.ok:
                self.stats.ok += 1
                if comp.cached:
                    self.stats.cached += 1
            else:
                self.stats.failed += 1
                if comp.error and len(self.stats.errors) < 10:
                    self.stats.errors.append(comp.error[:200])
    def run_one(self, job: Job, stage: str = "") -> Completion:
        out = self.run([job], stage=stage)
        return out[0]

    def summary(self) -> dict[str, Any]:
        out = self.stats.as_dict()
        out["provider"] = self.provider.describe()
        out["budget"] = {"max_total_tokens": self.max_total_tokens,
                         "max_cost_usd": self.max_cost_usd,
                         "spent_tokens": self.ledger.spent(self.stage),
                         "spent_usd": round(self.ledger.spent_usd(self.stage), 6)}
        return out


def run_jobs(provider: LLMProvider, ledger: TokenLedger, cfg: Config,
             jobs: Iterable[Job], stage: str = "synth",
             progress_every: int = 200) -> list[Completion]:
    """One-shot helper: build a runner, run ``jobs``, print coarse progress.

    Progress uses plain prints (no ``tqdm``: it is not an installed dependency)
    and only reports every ``progress_every`` completed jobs so a 10k-call sweep
    stays quiet.
    """
    job_list = list(jobs)
    runner = AsyncBatchRunner(provider, ledger, cfg, stage=stage)
    total = len(job_list)
    if total == 0:
        return []
    print(f"[{stage}] dispatching {total} provider calls "
          f"(concurrency={runner.concurrency}, model={provider.model})")
    results: list[Completion] = []
    chunk = max(1, progress_every)
    for start in range(0, total, chunk):
        batch = job_list[start:start + chunk]
        results.extend(runner.run(batch, stage=stage))
        done = min(total, start + chunk)
        stats = runner.stats
        print(f"[{stage}] {done}/{total} calls | ok={stats.ok} failed={stats.failed} "
              f"skipped={stats.skipped_budget} tokens={stats.total_tokens}")
    return results
