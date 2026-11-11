"""Deterministic offline teacher.

Why the mock is a first-class provider, not a stub
-------------------------------------------------
The demo run must be fully offline (no network, no API keys), yet the whole point
of the project is the sentence *"token investment in synthesis turns into
downstream capability"*. That claim needs a corpus a model can actually learn
from, so the mock provider is the **teacher of record** for every demo artifact:
the CoT traces, the paraphrases and the judge verdicts in ``artifacts/`` are all
produced here.

Consequences, all deliberate:

* **Grounded, not generic.** Output is generated from the prompt's own
  ``<schema>`` / ``<gold_sql>`` blocks, parsed by
  :mod:`fcdata.providers.sqlread`. Nothing is hard-coded per dataset, so the same
  teacher works on the fixture and on real Spider.
* **Deterministic.** The RNG is seeded from ``sha256(task | sample_id | prompt)``.
  Two runs of the pipeline produce byte-identical JSONL (modulo timestamps),
  which makes "re-run and diff" a real regression test.
* **Honest token accounting.** ``prompt_tokens`` / ``completion_tokens`` come from
  :mod:`fcdata.tokens`, i.e. the same estimator the budget uses. A mock that
  reported zero tokens would break the cost/benefit table.
* **Same prompt contract as the real API.** The mock reads the *same* prompt text
  an OpenAI-compatible endpoint would receive, so if the prompt loses its schema
  block the mock degrades visibly (it raises a diagnostic error) instead of
  silently producing plausible text.

Failure semantics: when the prompt does not contain what the task needs (e.g. a
``cot`` task with no ``<schema>``), the provider returns ``ok=False`` with an
explanatory ``error`` rather than inventing output. That is what a real API would
effectively do by hallucinating, except the mock makes the problem *visible*.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..tokens import estimate_messages_tokens, estimate_tokens
from .base import Completion, LLMProvider
from . import prompts
from .mockwriters import (write_correction, write_cot, write_judge,
                          write_paraphrase)
from .sqlread import DdlSchema, parse_ddl, read_sql

#: tasks this teacher knows how to answer
SUPPORTED_TASKS = ("cot", "paraphrase", "hard_negative", "correction", "judge")


def stable_seed(*parts: str) -> int:
    """A stable 63-bit seed from any strings (``hash()`` is salted per process)."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


class MockProvider(LLMProvider):
    """A schema-aware deterministic teacher used for the offline demo."""

    kind = "mock"

    def __init__(self, *, name: str = "mock", model: str = "mock-cheap-1",
                 role: str = "cheap", ledger: Any = None, stage: str = "",
                 temperature: float = 0.7, max_tokens: int = 512,
                 prices: tuple[float, float] | None = None,
                 seed_salt: str = "") -> None:
        super().__init__(name=name, model=model, role=role, ledger=ledger,
                         stage=stage, temperature=temperature,
                         max_tokens=max_tokens, prices=prices)
        self.seed_salt = seed_salt

    # -- main entry ------------------------------------------------------
    def _generate(self, messages: list[dict[str, Any]], *, task: str,
                  sample_id: str, temperature: float, max_tokens: int,
                  json_mode: bool) -> Completion:
        prompt = prompts.join_messages(messages)
        rng = random.Random(stable_seed(self.seed_salt, task, sample_id, prompt))
        ddl = prompts.extract_schema(prompt)
        schema = parse_ddl(ddl)

        handler = {
            "cot": self._do_cot,
            "paraphrase": self._do_paraphrase,
            "hard_negative": self._do_correction,
            "correction": self._do_correction,
            "judge": self._do_judge,
        }.get(task)
        if handler is None:
            return Completion(
                ok=False, model=self.model,
                error=f"mock provider has no handler for task {task!r}; "
                      f"supported: {', '.join(SUPPORTED_TASKS)}")
        if schema.is_empty:
            return Completion(
                ok=False, model=self.model,
                error=f"task {task!r} requires a <{prompts.SCHEMA_TAG}> block with at "
                      f"least one CREATE TABLE statement; none was found in the prompt")
        text = handler(rng, prompt, schema, max_tokens)
        if not text.strip():
            return Completion(ok=False, model=self.model,
                              error=f"task {task!r} produced no output for "
                                    f"sample {sample_id!r}")
        return Completion(text=text,
                          prompt_tokens=estimate_messages_tokens(messages),
                          completion_tokens=estimate_tokens(text),
                          ok=True, model=self.model, finish_reason="stop",
                          meta={"teacher": "mock", "task": task})

    # -- task handlers ---------------------------------------------------
    def _do_cot(self, rng: random.Random, prompt: str, schema: DdlSchema,
                max_tokens: int) -> str:
        question = prompts.extract_question(prompt)
        gold_sql = prompts.extract_gold_sql(prompt)
        if not gold_sql:
            return ""
        facts = read_sql(gold_sql, schema)
        return write_cot(rng, schema=schema, question=question,
                         gold_sql=gold_sql, facts=facts)

    def _do_paraphrase(self, rng: random.Random, prompt: str, schema: DdlSchema,
                       max_tokens: int) -> str:
        question = prompts.extract_question(prompt)
        if not question:
            return ""
        strategy = prompts.extract_strategy(prompt) or "synonym"
        return write_paraphrase(rng, question=question, strategy=strategy)

    def _do_correction(self, rng: random.Random, prompt: str, schema: DdlSchema,
                       max_tokens: int) -> str:
        question = prompts.extract_question(prompt)
        gold_sql = prompts.extract_gold_sql(prompt)
        bad_sql = prompts.extract_bad_sql(prompt)
        if not bad_sql:
            return ""
        error_kind = prompts.extract_error_kind(prompt)
        return write_correction(rng, schema=schema, question=question,
                                gold_sql=gold_sql, bad_sql=bad_sql,
                                error_kind=error_kind)

    def _do_judge(self, rng: random.Random, prompt: str, schema: DdlSchema,
                  max_tokens: int) -> str:
        question = prompts.extract_question(prompt)
        candidate = prompts.extract_candidate_sql(prompt)
        if not candidate:
            return ""
        gold_sql = prompts.extract_gold_sql(prompt)
        return write_judge(rng, schema=schema, question=question,
                           candidate_sql=candidate, gold_sql=gold_sql)
