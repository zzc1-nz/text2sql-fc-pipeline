"""Validation stage 2 -- the LLM semantic judge.

WHY a semantic judge at all
---------------------------
The format stage proves a record is *executable and schema-valid*, which is not
the same as *correct*: a generated query can run perfectly and still answer a
different question ("students in CS" vs "students not in CS" both execute).
Only a reader that understands the question can catch that, so a second, more
expensive stage asks an LLM to score question<->SQL consistency.

Why it is a *sample* and why the JSON is parsed defensively
-----------------------------------------------------------
The judge is the most expensive model in the pipeline (``cfg.validate.judge_role``),
so ``cfg.validate.semantic_sample`` limits how many records are judged (0 = all)
and the selection is deterministic in ``cfg.seed`` -- a rerun must spend exactly
the same tokens on exactly the same records, otherwise the cost/benefit tables
in the report are not comparable across runs.

Judge output is *not* trusted to be clean JSON: real models wrap it in fences,
prefix it with "Sure, here is my verdict:" or emit a 0-100 score.  :func:`parse_judge_json`
therefore accepts fenced JSON, embedded JSON (balanced-brace scan), prose prefixes
and both 0..1 / 0..100 / 0..10 score scales, then clamps to [0, 1].  A parse
failure is an explicit ``judge_parse_error``, never a silent pass.

The prompt embeds only the DDL of the tables the SQL touches (plus a fallback
slice of the schema), which keeps the judge call small and focused: irrelevant
tables are a known source of judge noise.
"""

from __future__ import annotations

import inspect
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..records import SynthRecord, ValidationReport
from ..schema import Database
from ..sqlutils import analyze_sql
from ..tokens import estimate_messages_tokens, estimate_tokens, price_usage

#: allowed verdict labels (contract)
VERDICTS: tuple[str, ...] = ("consistent", "inconsistent", "uncertain")

#: issue codes produced by this stage (informational: pass/fail is score-driven)
SEMANTIC_CODES: frozenset[str] = frozenset({
    "semantic_consistent",
    "semantic_inconsistent",
    "semantic_uncertain",
    "semantic_low_score",
    "judge_error",
    "judge_parse_error",
    "judge_budget_exhausted",
})

JUDGE_SYSTEM = """You are a strict semantic judge for a text-to-SQL dataset.
You are given a natural-language question, a SQL query, and the database schema.
Decide whether the SQL query, when executed on that schema, returns exactly the
information the question asks for.

Rules:
- Judge meaning, not style. Aliases, join order, formatting and logically
  equivalent phrasing are fine.
- These are inconsistencies: a missing or wrong filter, the wrong aggregate or
  aggregated column, a wrong join, answering a different question, or ignoring an
  ordering/limit the question asks for ("top", "most", "highest", "first").
- Use "uncertain" only when the schema/question genuinely does not determine the
  answer.
Reply with ONLY a JSON object, no prose and no markdown:
{"score": <float between 0 and 1>, "verdict": "consistent|inconsistent|uncertain", "rationale": "<one short sentence>"}"""


# --------------------------------------------------------------------------- #
# JSON extraction (deliberately local: the validate stage depends only on the
# frozen contracts, never on another stage's helpers)
# --------------------------------------------------------------------------- #
def iter_json_objects(text: str) -> Iterable[str]:
    """Yield every balanced ``{...}`` substring, outermost first.

    A real parser is overkill for "pull the JSON out of a chatty response", but a
    naive ``re.search(r"\\{.*\\}")`` is wrong when the response contains two
    objects or a ``}`` inside a string literal, so braces are counted with
    string-literal awareness.
    """
    if not text:
        return
    n = len(text)
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, n):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Best-effort strict-ish parse of a judge reply.

    Returns ``{"score": float in [0,1], "verdict": str, "rationale": str}`` or
    ``None`` when the reply contains no usable verdict.  Accepted inputs::

        {"score": 0.9, "verdict": "consistent", "rationale": "..."}
        ```json ... ```
        Sure! Here is my verdict: {"score": 0.9, ...}
        {"score": 90, ...}            # 0..100 scale
        {"score": "0.75", ...}        # score as string
        {"verdict": "consistent"}     # score derived from the verdict
    """
    if not text:
        return None
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    candidates.extend(iter_json_objects(text))

    seen: set[str] = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        obj: Any = None
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            # single-quoted / python-ish dicts are common enough to be worth one
            # narrow repair attempt (never eval(): judge text is untrusted input)
            try:
                obj = json.loads(cand.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                obj = None
        if isinstance(obj, dict):
            parsed = _coerce_verdict(obj)
            if parsed is not None:
                return parsed
    return None


def _coerce_verdict(obj: dict[str, Any]) -> dict[str, Any] | None:
    raw_score = obj.get("score", obj.get("Score"))
    raw_verdict = obj.get("verdict", obj.get("Verdict", obj.get("label", "")))
    rationale = obj.get("rationale", obj.get("reason", obj.get("explanation", "")))

    verdict = str(raw_verdict or "").strip().lower()
    verdict = verdict if verdict in VERDICTS else ""

    score: float | None = None
    if raw_score is not None:
        try:
            score = float(str(raw_score).strip().rstrip("%"))
        except (TypeError, ValueError):
            score = None
        if score is not None:
            score = _normalize_score(score)
            if str(raw_score).strip().endswith("%"):
                score = min(1.0, score)

    if score is None:
        if verdict == "consistent":
            score = 1.0
        elif verdict == "inconsistent":
            score = 0.0
        elif verdict == "uncertain":
            score = 0.5
        else:
            return None
    if not verdict:
        verdict = "consistent" if score >= 0.75 else (
            "inconsistent" if score <= 0.25 else "uncertain")
    return {"score": round(max(0.0, min(1.0, score)), 4),
            "verdict": verdict,
            "rationale": re.sub(r"\s+", " ", str(rationale or "")).strip()[:400]}


def _normalize_score(score: float) -> float:
    """Map a plausible 0..1 / 0..10 / 0..100 score onto [0, 1]."""
    if score < 0:
        return 0.0
    if score <= 1.0:
        return score
    if score <= 10.0:
        return score / 10.0
    if score <= 100.0:
        return score / 100.0
    return 1.0


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
def render_relevant_ddl(db: Database | None, sql: str, max_tables: int = 8) -> str:
    """DDL for the tables the SQL touches (fallback: the first ``max_tables``)."""
    if db is None:
        return ""
    names: list[str] = []
    for name in analyze_sql(sql, db).tables:
        if name not in names:
            names.append(name)
    tables = [t for t in (db.table(n) for n in names) if t is not None]
    if not tables:
        tables = db.tables[:max_tables]
    lines: list[str] = []
    for t in tables[:max_tables]:
        cols = ", ".join(
            f"{c.name} {c.type.upper()}" + (" PRIMARY KEY" if c.is_primary_key else "")
            for c in t.columns)
        lines.append(f"CREATE TABLE {t.name} ({cols});")
    return "\n".join(lines)


def build_judge_messages(record: SynthRecord, db: Database | None, *,
                         max_tables: int = 8) -> list[dict[str, Any]]:
    """Compact question + SQL + relevant DDL prompt for the judge.

    This is the *fallback* prompt: it is self-contained and works with any
    provider, but it is plain prose.  When :mod:`fcdata.providers.prompts` is
    available, :func:`provider_judge_messages` is preferred instead, because the
    offline mock teacher recovers the schema/question/candidate SQL by parsing the
    provider's XML-ish tag grammar -- feeding it prose makes it fail honestly
    (``ok=False``) rather than score the record.
    """
    ddl = render_relevant_ddl(db, record.sql, max_tables)
    parts = [f"Question: {record.question.strip()}",
             f"SQL:\n{record.sql.strip()}"]
    if ddl:
        parts.append(f"Schema:\n{ddl}")
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def provider_judge_messages(record: SynthRecord, db: Database | None, *,
                            max_tables: int = 8) -> list[dict[str, Any]] | None:
    """Build the judge prompt with the provider's own prompt grammar, if present.

    WHY prefer it: a provider's prompt module is the *contract* between the
    prompt text and the backend.  The offline mock provider has no other view of
    the world -- it re-parses ``<schema>``/``<question>``/``<candidate_sql>`` out
    of the prompt -- so using the same builder is what keeps the offline demo
    representative of a real API run.  Returns ``None`` when the module is absent
    (a different stage may not have landed yet) so the caller falls back.

    The record's own SQL is the *candidate* under review; no reference gold SQL is
    passed, because telling the judge the answer would make the check circular.
    """
    try:
        from ..providers import prompts as provider_prompts  # noqa: PLC0415 - lazy
    except ImportError:
        return None
    ddl = render_relevant_ddl(db, record.sql, max_tables)
    if not ddl:
        return None
    try:
        return provider_prompts.judge_messages(
            ddl, record.question.strip(), record.sql.strip(),
            gold_sql="", db_id=getattr(record, "db_id", "") or "")
    except (AttributeError, TypeError):  # pragma: no cover - provider API drift
        return None


# --------------------------------------------------------------------------- #
# one judge call
# --------------------------------------------------------------------------- #
@dataclass
class SemanticVerdict:
    """Outcome of one judge call (also carries its own token accounting)."""

    score: float = 0.0
    verdict: str = ""
    rationale: str = ""
    ok: bool = False
    error: str = ""
    #: True when the provider replied but the reply was not parseable as a verdict
    parse_failed: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def issue_code(self) -> str:
        if not self.ok:
            return "judge_parse_error" if self.parse_failed else "judge_error"
        if self.verdict == "inconsistent":
            return "semantic_inconsistent"
        if self.verdict == "uncertain":
            return "semantic_uncertain"
        return "semantic_consistent"


def _accepts_stage(provider: Any) -> bool:
    """True when ``provider.complete`` accepts the optional ``stage`` kwarg.

    Checked by introspection rather than by catching ``TypeError``: a blanket
    except would also swallow a genuine ``TypeError`` raised *inside* the
    provider and silently retry the call (double spend, doubled ledger entry).
    """
    try:
        params = inspect.signature(provider.complete).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "stage" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                    for p in params.values())


def judge_one(provider: Any, record: SynthRecord, db: Database | None, *,
              max_tokens: int = 256, max_tables: int = 8,
              stage: str = "validate") -> SemanticVerdict:
    """Ask the judge about one record.

    ``provider`` is any object exposing the contract's
    ``complete(messages, *, task, sample_id, ...) -> Completion``.  The provider
    owns ledger recording (exactly one ``UsageRecord`` per call); this function
    only mirrors the returned usage into the verdict so the report can attribute
    judge tokens per record.

    ``stage="validate"`` is forwarded when the provider supports it, so its ledger
    entry lands in the validation bucket instead of an ``adhoc`` one -- otherwise
    the per-stage token table in the report would be missing the judge cost.
    """
    messages = (provider_judge_messages(record, db, max_tables=max_tables)
                or build_judge_messages(record, db, max_tables=max_tables))
    est_prompt = estimate_messages_tokens(messages)
    kwargs: dict[str, Any] = dict(task="judge", sample_id=record.id,
                                  temperature=0.0, max_tokens=max_tokens,
                                  json_mode=True)
    if _accepts_stage(provider):
        kwargs["stage"] = stage
    try:
        comp = provider.complete(messages, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a provider failure must not kill the batch
        return SemanticVerdict(ok=False, error=f"provider error: {exc}",
                               prompt_tokens=est_prompt)

    raw = str(getattr(comp, "text", "") or "")
    prompt_tokens = int(getattr(comp, "prompt_tokens", 0) or 0) or est_prompt
    completion_tokens = int(getattr(comp, "completion_tokens", 0) or 0) or estimate_tokens(raw)
    if not getattr(comp, "ok", True):
        return SemanticVerdict(ok=False,
                               error=str(getattr(comp, "error", "") or "completion failed"),
                               prompt_tokens=prompt_tokens,
                               completion_tokens=completion_tokens, raw=raw)

    parsed = parse_judge_json(raw)
    if parsed is None:
        return SemanticVerdict(ok=False, error="unparseable judge JSON", parse_failed=True,
                               prompt_tokens=prompt_tokens,
                               completion_tokens=completion_tokens, raw=raw)
    return SemanticVerdict(score=float(parsed["score"]), verdict=str(parsed["verdict"]),
                           rationale=str(parsed["rationale"]), ok=True,
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens, raw=raw)


def apply_verdict(report: ValidationReport, verdict: SemanticVerdict) -> None:
    """Copy a verdict onto a report (semantic fields + token accounting)."""
    report.semantic_checked = True
    report.semantic_score = round(float(verdict.score), 4)
    report.semantic_verdict = verdict.verdict or "uncertain"
    report.semantic_rationale = verdict.rationale
    report.judge_prompt_tokens = int(verdict.prompt_tokens)
    report.judge_completion_tokens = int(verdict.completion_tokens)
    code = verdict.issue_code
    if code not in report.issue_codes:
        report.issue_codes.append(code)
    if not verdict.ok:
        report.reason = (report.reason + " | " if report.reason else "") + \
            f"semantic: {verdict.error}"


# --------------------------------------------------------------------------- #
# deterministic sampling + batch driver
# --------------------------------------------------------------------------- #
def select_semantic_sample(ids: Sequence[str], sample: int, seed: int) -> set[str]:
    """Deterministically pick ``sample`` record ids (0 or too-large = all).

    Sorted-then-``random.Random(seed)`` sampling makes the choice independent of
    the order the synthesizer happened to emit records in, which is what keeps
    judge spend reproducible across reruns.
    """
    unique = sorted(set(ids))
    if sample <= 0 or sample >= len(unique):
        return set(unique)
    rng = random.Random(int(seed) + 991)
    return set(rng.sample(unique, sample))


def run_semantic(records: Sequence[SynthRecord],
                 cfg: Any,
                 provider: Any,
                 databases: dict[str, Database],
                 *,
                 spent_tokens: int = 0,
                 stage: str = "validate") -> dict[str, SemanticVerdict]:
    """Judge a deterministic sample of ``records``; honours the token budget.

    ``records`` should already be the format-passing subset (judging SQL that
    cannot run wastes money).  When ``cfg.validate.max_total_tokens`` is hit the
    remaining records are simply left unjudged -- the caller treats "not checked"
    as "format decides", and the report records ``judge_budget_exhausted``.
    """
    vcfg = cfg.validate
    budget = int(getattr(vcfg, "max_total_tokens", 0) or 0)
    chosen = select_semantic_sample([r.id for r in records],
                                    int(getattr(vcfg, "semantic_sample", 0) or 0),
                                    int(getattr(cfg, "seed", 17)))
    max_tokens = int(cfg.provider(vcfg.judge_role).max_tokens)
    max_tables = int(getattr(cfg.fmt, "max_schema_tables", 12) or 12)

    out: dict[str, SemanticVerdict] = {}
    used = int(spent_tokens)
    for rec in records:
        if rec.id not in chosen:
            continue
        if budget and used >= budget:
            break
        verdict = judge_one(provider, rec, databases.get(rec.db_id),
                            max_tokens=max_tokens, max_tables=max_tables,
                            stage=stage)
        out[rec.id] = verdict
        used += verdict.total_tokens
    return out


def judge_cost_usd(cfg: Any, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of the judge calls, using the role's configured prices."""
    role = cfg.validate.judge_role
    pc = cfg.provider(role)
    return round(price_usage(prompt_tokens, completion_tokens,
                             float(pc.price_in_per_mtok),
                             float(pc.price_out_per_mtok)), 8)
