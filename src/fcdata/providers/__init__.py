"""LLM providers: one accounting boundary, several backends.

Public API (frozen by CONTRACTS.md)::

    build_provider(role, cfg, ledger) -> LLMProvider
    AsyncBatchRunner(provider, ledger, cfg).run(jobs, stage) -> list[Completion]

Why the factory takes a *role* rather than a model name
-------------------------------------------------------
The pipeline distinguishes *cheap bulk generation* from *expensive judging*
(``cfg.providers["cheap"]`` vs ``["judge"]``). A role is a cost/quality contract:
the synthesis stage always asks for ``cheap`` and the validator always asks for
``judge``, and the config decides what those names point at. Swapping the mock
teacher for DeepSeek in ``configs/production.yaml`` therefore changes zero code.

Selection rule: ``cfg.provider(role).name`` is ``"mock"`` -> deterministic offline
teacher; anything else -> OpenAI-compatible HTTP client (``"openai"``,
``"deepseek"``, ``"vllm"``, ... are all the same wire protocol). An unknown name
raises instead of silently falling back, because a typo in a config file must not
quietly produce a mock corpus that looks like a real run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config, ProviderCfg
from ..tokens import TokenLedger
from .base import Completion, Job, LLMProvider
from .mock import MockProvider, stable_seed
from .openai_compat import OpenAICompatProvider, ResponseCache
from .runner import (BUDGET_EXHAUSTED, AsyncBatchRunner, BatchStats, run_jobs)

__all__ = [
    "Completion", "Job", "LLMProvider", "MockProvider", "OpenAICompatProvider",
    "AsyncBatchRunner", "BatchStats", "ResponseCache", "BUDGET_EXHAUSTED",
    "build_provider", "stable_seed", "run_jobs", "cache_path_for",
]

#: provider ``name`` values understood by :func:`build_provider`
MOCK_NAMES = frozenset({"mock", "mock-cheap", "mock-judge", "offline"})
OPENAI_NAMES = frozenset({"openai", "openai_compat", "deepseek", "vllm", "together",
                          "groq", "openrouter", "compatible", "api"})


def cache_path_for(cfg: Config, role: str, model: str) -> Path:
    """Where the disk response cache for ``(role, model)`` lives.

    Keyed by model so switching models never replays another model's answers --
    the cache is a *response* cache, not a *prompt* cache.
    """
    safe_model = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model)
    return cfg.paths.resolve(cfg.paths.cache_dir) / f"{role}__{safe_model}.jsonl"


def build_provider(role: str, cfg: Config, ledger: TokenLedger | None = None,
                   *, stage: str = "") -> LLMProvider:
    """Instantiate the provider configured for ``role``."""
    pcfg: ProviderCfg = cfg.provider(role)
    name = (pcfg.name or "mock").strip().lower()
    prices = (float(pcfg.price_in_per_mtok), float(pcfg.price_out_per_mtok))
    common: dict[str, Any] = dict(
        name=name, model=pcfg.model, role=role, ledger=ledger, stage=stage,
        temperature=float(pcfg.temperature), max_tokens=int(pcfg.max_tokens),
        prices=prices,
    )
    if name in MOCK_NAMES:
        return MockProvider(**common)
    if name in OPENAI_NAMES:
        cache_path = cache_path_for(cfg, role, pcfg.model)
        return OpenAICompatProvider(
            **common,
            base_url=pcfg.base_url,
            api_key_env=pcfg.api_key_env,
            concurrency=int(pcfg.concurrency),
            max_retries=int(pcfg.max_retries),
            timeout_s=float(pcfg.timeout_s),
            cache_path=cache_path,
        )
    raise ValueError(
        f"unknown provider {name!r} for role {role!r}; "
        f"expected one of {sorted(MOCK_NAMES | OPENAI_NAMES)}")
