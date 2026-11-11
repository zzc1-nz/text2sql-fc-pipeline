"""Paraphrase synthesis: robustness against question phrasing.

Why paraphrases matter for this project
---------------------------------------
A text2SQL model trained only on Spider's single question per query learns a
surface form, not a mapping. Paraphrase augmentation is the cheapest fix: for each
(source question, gold SQL) pair, produce N rewrites and attach the *same* gold
SQL. The student then sees that ``how many`` and ``count the`` are the same query.

The hard part is **not** generating text -- it is guaranteeing semantic
preservation. A rewrite that drops "more than 900" or turns "each department" into
"the department" teaches the model a wrong mapping, which is worse than no data at
all. So every candidate rewrite passes a deterministic guard
(:func:`fcdata.providers.mockwriters.paraphrase_preserves`):

* identical set of numbers and quoted literals,
* every capitalised entity value survives,
* not identical to the original,
* length within a factor of two.

On top of that, this module enforces two dataset-level properties:

* **Strategy rotation.** Rewrites for one source are drawn from *distinct*
  strategies (synonym / reorder / politeness / count-form / clause / compression)
  rather than N draws from one, so a source does not produce three near-identical
  sentences.
* **Deduplication.** If two strategies still converge on the same sentence, the
  duplicate is dropped and counted, because duplicate rows silently up-weight one
  phrasing in the SFT loss.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..config import Config
from ..records import SynthRecord
from ..tokens import TokenLedger
from ..providers import AsyncBatchRunner, Job, build_provider
from ..providers.prompts import PARAPHRASE_STRATEGIES, paraphrase_messages
from ..providers.mockwriters import paraphrase_preserves
from .common import (SYNTH_STAGE, KindReport, SourcePlan, build_sources,
                     make_record, normalize_question, stable_rng)

#: strategies that change surface form without touching content
SAFE_STRATEGIES: tuple[str, ...] = PARAPHRASE_STRATEGIES


def pick_strategies(source_id: str, count: int,
                    start_offset: int = 0) -> list[str]:
    """Deterministic, distinct strategy rotation for one source example.

    The rotation is seeded by the source id, so different examples start at
    different points in the strategy list (giving the corpus a balanced mix
    overall) while a re-run reproduces the same assignment exactly.
    """
    if count <= 0:
        return []
    rng = stable_rng("paraphrase", source_id)
    offset = rng.randrange(len(SAFE_STRATEGIES))
    out: list[str] = []
    for i in range(count):
        out.append(SAFE_STRATEGIES[(offset + start_offset + i) % len(SAFE_STRATEGIES)])
    return out


def _plan_jobs(plans: Sequence[SourcePlan], per_example: int
               ) -> tuple[list[Job], list[tuple[SourcePlan, int, str]]]:
    jobs: list[Job] = []
    keys: list[tuple[SourcePlan, int, str]] = []
    for plan in plans:
        for idx, strategy in enumerate(pick_strategies(str(plan.example.id),
                                                       per_example)):
            jobs.append(Job(
                messages=paraphrase_messages(plan.ddl, plan.question,
                                             plan.gold_sql, strategy,
                                             plan.db_id),
                task="paraphrase",
                sample_id=f"{plan.example.id}#para{idx}",
                # paraphrase is a *style* task: a little heat helps variety, but
                # the deterministic guard rejects anything that drifts in meaning.
                temperature=0.8,
                max_tokens=96,
                meta={"source_id": plan.example.id, "index": idx,
                      "strategy": strategy},
            ))
            keys.append((plan, idx, strategy))
    return jobs, keys


def clean_rewrite(text: str) -> str:
    """Strip fences/labels a teacher might add around a one-line rewrite.

    Quoting is removed *only* when it wraps the whole line (``"List the books."``),
    never when it is part of the sentence: a rewrite of "the year of 'Glass
    Hearts'" must keep both apostrophes, because a lost closing quote silently
    corrupts the entity the model has to copy into SQL.
    """
    import re

    body = (text or "").strip()
    body = re.sub(r"^```(?:text)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    body = re.sub(r"^\s*(?:rewritten?\s+question|paraphrase|output)\s*[:\-]\s*", "",
                  body, flags=re.IGNORECASE)
    for line in body.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        # strip a *wrapping* quote pair only, and only if it is balanced
        if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'":
            inner = line[1:-1]
            if line[0] not in inner:
                line = inner.strip()
        return line
    return ""


def _same_surface(left: str, right: str) -> bool:
    """True when two questions are identical apart from case/whitespace/punctuation."""
    import re

    def _squash(text: str) -> str:
        t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", t).strip()

    return _squash(left) == _squash(right)


def synth_paraphrase(cfg: Config, spider_data: Any, ledger: TokenLedger,
                     *, plans: Sequence[SourcePlan] | None = None,
                     runner: AsyncBatchRunner | None = None
                     ) -> tuple[list[SynthRecord], KindReport]:
    """Generate paraphrase records (same gold SQL, rewritten question)."""
    per_example = int(cfg.synth.paraphrase_per_example)
    report = KindReport(kind="paraphrase")
    if per_example <= 0:
        return [], report
    if plans is None:
        plans = build_sources(cfg, spider_data)
    jobs, keys = _plan_jobs(plans, per_example)
    report.planned = len(jobs)
    if not jobs:
        return [], report

    provider = build_provider("cheap", cfg, ledger, stage=SYNTH_STAGE)
    active_runner = runner or AsyncBatchRunner(provider, ledger, cfg, stage=SYNTH_STAGE)
    completions = active_runner.run(jobs, stage=SYNTH_STAGE)

    records: list[SynthRecord] = []
    seen: set[tuple[str, str]] = set()
    for (plan, idx, strategy), comp in zip(keys, completions):
        report.prompt_tokens += comp.prompt_tokens
        report.completion_tokens += comp.completion_tokens
        if comp.error == "budget_exhausted":
            report.skipped_budget += 1
            continue
        if not comp.ok:
            report.failed += 1
            if comp.error:
                report.errors.append(comp.error[:160])
            continue
        rewrite = clean_rewrite(comp.text)
        if not rewrite:
            report.dropped_invalid += 1
            report.errors.append("empty_rewrite")
            continue
        if not paraphrase_preserves(plan.question, rewrite):
            report.dropped_invalid += 1
            report.errors.append("meaning_drift")
            continue
        # A rewrite that is *literally* the original is a no-op: training on it
        # would duplicate the source question at full token cost. Note this
        # compares the raw surface text, NOT ``normalize_question`` -- that helper
        # deliberately strips conversational frames ("could you", "please"), and a
        # rewrite that only adds such a frame is a legitimate paraphrase.
        if _same_surface(rewrite, plan.question):
            report.dropped_invalid += 1
            if "identical_rewrite" not in report.errors:
                report.errors.append("identical_rewrite")
            continue
        # Two rewrites of the same source that differ only in framing collapse to
        # one training target, so keep the first and count the rest.
        key = (str(plan.example.id), normalize_question(rewrite))
        if key in seen:
            report.dropped_duplicate += 1
            continue
        seen.add(key)
        records.append(make_record(
            kind="paraphrase", plan=plan, index=idx, question=rewrite,
            completion=comp, role="cheap", model=provider.model,
            original_question=plan.question,
            meta={"strategy": strategy,
                  "source_question": plan.question,
                  "length_delta": len(rewrite) - len(plan.question)},
        ))
        report.ok += 1
    return records, report
