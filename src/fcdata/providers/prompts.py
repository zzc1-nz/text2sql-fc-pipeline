"""Prompt contracts for every LLM task in the pipeline.

Why a dedicated module
----------------------
Three separate concerns need to agree on prompt text:

1. the **real** OpenAI-compatible provider sends it verbatim to an API,
2. the **mock** provider parses it back out (it has no other view of the world --
   it must recover the schema, question and gold SQL *from the prompt*),
3. the **tests** assert on its structure.

If prompt strings were inlined at call sites, the mock's parser and the real
prompt would drift apart and the offline demo would stop being representative of
the online run. So every prompt lives here, in one grammar:

    <schema>   ... CREATE TABLE block ...   </schema>
    <question> ... natural language ...     </question>
    <gold_sql> ... trusted SQL ...          </gold_sql>
    <bad_sql>  ... corrupted SQL ...        </bad_sql>
    <candidate_sql> ... SQL under review ...</candidate_sql>

XML-ish tags are used deliberately: they survive being embedded in a single
``user`` message, they are unambiguous to a regex, and they are a well-known
prompt-engineering pattern for delimiting untrusted content. Every prompt states
an **explicit output contract** (what the model must emit, in what shape) because
a teacher model whose output we must parse cannot be allowed to improvise.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# tag grammar + extraction helpers
# --------------------------------------------------------------------------- #
SCHEMA_TAG = "schema"
QUESTION_TAG = "question"
GOLD_SQL_TAG = "gold_sql"
BAD_SQL_TAG = "bad_sql"
CANDIDATE_SQL_TAG = "candidate_sql"
ERROR_KIND_TAG = "error_kind"
STRATEGY_TAG = "rewrite_strategy"
DATABASE_TAG = "database"

_TAG_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _tag_re(tag: str) -> re.Pattern[str]:
    pat = _TAG_RE_CACHE.get(tag)
    if pat is None:
        pat = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
        _TAG_RE_CACHE[tag] = pat
    return pat


def tag_block(tag: str, body: str) -> str:
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def extract_tag(text: str, tag: str, default: str = "") -> str:
    """Return the content of ``<tag>...</tag>`` (last occurrence wins)."""
    matches = _tag_re(tag).findall(text or "")
    return matches[-1].strip() if matches else default


def extract_schema(prompt: str) -> str:
    return extract_tag(prompt, SCHEMA_TAG)


def extract_question(prompt: str) -> str:
    return extract_tag(prompt, QUESTION_TAG)


def extract_gold_sql(prompt: str) -> str:
    return extract_tag(prompt, GOLD_SQL_TAG)


def extract_bad_sql(prompt: str) -> str:
    return extract_tag(prompt, BAD_SQL_TAG)


def extract_candidate_sql(prompt: str) -> str:
    return extract_tag(prompt, CANDIDATE_SQL_TAG)


def extract_error_kind(prompt: str) -> str:
    return extract_tag(prompt, ERROR_KIND_TAG)


def extract_strategy(prompt: str) -> str:
    return extract_tag(prompt, STRATEGY_TAG)


def join_messages(messages: Sequence[dict[str, Any]]) -> str:
    """Flatten a chat into one string (used only for cache keys / mock seeding)."""
    parts: list[str] = []
    for m in messages:
        parts.append(f"<<{m.get('role', 'user')}>>")
        parts.append(str(m.get("content", "")))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# CoT: explain how the question maps onto the gold query
# --------------------------------------------------------------------------- #
COT_SYSTEM = """\
You are a senior data analyst writing the *reasoning trace* that teaches a smaller \
model how to translate a natural-language question into SQL.

Rules:
1. Reason only from the provided schema. Never invent a table or a column.
2. Refer to real identifiers exactly as they are spelled in the schema.
3. Work in the order a human would: which tables are needed -> how they are \
connected -> which rows survive the filters -> what is computed -> how the result \
is ordered or limited.
4. 3 to 6 numbered steps. Each step is one or two sentences. No SQL keyword soup: \
explain *why*, not just *what*.
5. Do not restate the question. Do not mention the gold SQL as "the gold SQL".
6. Finish with a line `Final SQL:` followed by the query on a single line.

Output contract: numbered steps, then `Final SQL: <query>`. No markdown headings."""

COT_USER = """\
Write the reasoning trace for this question.

{schema_block}

{question_block}

{gold_block}

The trace must justify every clause of the query: selected columns, tables joined, \
filters, aggregation/grouping, ordering and limit. Output numbered steps then \
`Final SQL: ...`."""


def cot_messages(ddl: str, question: str, gold_sql: str, db_id: str = "") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": COT_USER.format(
            schema_block=tag_block(SCHEMA_TAG, ddl) + _db_hint(db_id),
            question_block=tag_block(QUESTION_TAG, question),
            gold_block=tag_block(GOLD_SQL_TAG, gold_sql),
        )},
    ]


def _db_hint(db_id: str) -> str:
    return f"\n<{DATABASE_TAG}>{db_id}</{DATABASE_TAG}>" if db_id else ""


# --------------------------------------------------------------------------- #
# paraphrase: rewrite the question, keep the meaning
# --------------------------------------------------------------------------- #
PARAPHRASE_SYSTEM = """\
You are a dataset author producing *question paraphrases* for a text-to-SQL \
training set.

Hard constraints:
1. The rewritten question must have exactly the same meaning as the original: \
same entities, same filters, same aggregation, same ordering/limit semantics.
2. Never introduce a table, column, value, or number that is not in the original.
3. Never reveal SQL. The output is a natural-language question only.
4. Keep it a single sentence (a question mark is optional but preferred).
5. It must read like something a real user would type, not like a template.

Output contract: exactly one line containing the rewritten question and nothing \
else -- no quotes, no prefix, no explanation."""

PARAPHRASE_USER = """\
Rewrite the question below using this strategy: {strategy}.

{strategy_hint}

{schema_block}

{question_block}

{strategy_block}
Return only the rewritten question."""

_STRATEGY_HINTS: dict[str, str] = {
    "synonym": ("Replace content words with natural synonyms (e.g. show/list/display, "
                "find/identify, average/mean, number of/count of). Keep every entity "
                "and value identical."),
    "reorder": ("Move a modifier to the front or the back of the sentence, e.g. turn "
                "'Show the names of X that Y' into 'For X that Y, what are the names?'. "
                "Keep the same scope."),
    "politeness": ("Add or remove a conversational lead-in such as 'please' or 'can you "
                   "tell me', but keep the information content identical."),
    "count_form": ("Switch between the 'how many X' form and the 'count the X' / "
                   "'what is the number of X' form."),
    "clause": ("Restructure into a relative clause or a wh-question, e.g. "
               "'Which X ... ?' <-> 'X that ... : what are they?'. Preserve filters."),
    "compression": ("Say the same thing in fewer words, dropping filler while keeping "
                    "every constraint and value."),
}

PARAPHRASE_STRATEGIES: tuple[str, ...] = tuple(_STRATEGY_HINTS)


def paraphrase_messages(ddl: str, question: str, gold_sql: str, strategy: str,
                        db_id: str = "") -> list[dict[str, Any]]:
    if strategy not in _STRATEGY_HINTS:
        strategy = "synonym"
    return [
        {"role": "system", "content": PARAPHRASE_SYSTEM},
        {"role": "user", "content": PARAPHRASE_USER.format(
            strategy=strategy,
            strategy_hint=_STRATEGY_HINTS[strategy],
            schema_block=tag_block(SCHEMA_TAG, ddl) + _db_hint(db_id),
            question_block=tag_block(QUESTION_TAG, question),
            strategy_block=tag_block(STRATEGY_TAG, strategy),
        )},
    ]


# --------------------------------------------------------------------------- #
# hard negative: diagnose the corrupted query
# --------------------------------------------------------------------------- #
CORRECTION_SYSTEM = """\
You are a SQL reviewer. A colleague submitted a query that is *wrong*: it runs, but \
it answers a different question than the one asked.

Write a short diagnosis that a training example can use as a correction note.

Rules:
1. Name the exact column/table/aggregate that is wrong, using the schema spelling.
2. State what the wrong query actually computes, then what it must compute instead.
3. One or two sentences. No preamble, no restating the whole query.
4. Do not mention "the gold query" or "the reference"; explain the bug on its own terms.

Output contract: a single paragraph of at most 45 words, ending with a period."""

CORRECTION_USER = """\
Diagnose the bug in the submitted query.

{schema_block}

{question_block}

{bad_block}

{error_block}

The correct answer to the question is:

{gold_block}

Explain precisely what the submitted query gets wrong and what it should do instead."""


def correction_messages(ddl: str, question: str, bad_sql: str, gold_sql: str,
                        error_kind: str, db_id: str = "") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CORRECTION_SYSTEM},
        {"role": "user", "content": CORRECTION_USER.format(
            schema_block=tag_block(SCHEMA_TAG, ddl) + _db_hint(db_id),
            question_block=tag_block(QUESTION_TAG, question),
            bad_block=tag_block(BAD_SQL_TAG, bad_sql),
            error_block=tag_block(ERROR_KIND_TAG, error_kind),
            gold_block=tag_block(GOLD_SQL_TAG, gold_sql),
        )},
    ]


# --------------------------------------------------------------------------- #
# judge: is the candidate SQL a faithful reading of the question?
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """\
You are a strict but fair grader for text-to-SQL data quality.

You receive a database schema, a natural-language question and a candidate SQL \
query. Decide whether the candidate query, when executed, answers *that* question.

Score 0.0-1.0:
* 1.0 -- fully correct: right tables, joins, filters, literals, aggregation, ordering.
* 0.7-0.9 -- correct in substance; cosmetic difference (extra column, alias, \
equivalent formulation, harmless DISTINCT).
* 0.4-0.6 -- partially right: right tables but a wrong filter/aggregate/ordering, so \
the result set is materially different.
* 0.0-0.3 -- wrong tables, missing join, hallucinated column, or does not execute.

Be conservative about high scores: an unrequested filter, a missing GROUP BY or a \
changed literal makes the answer wrong, not merely different.

Output contract: a single JSON object on one line, no code fence, no prose:
{"score": <float 0..1>, "verdict": "consistent"|"uncertain"|"inconsistent", \
"rationale": "<one sentence naming the decisive evidence>"}"""

JUDGE_USER = """\
Grade the candidate query.

{schema_block}

{question_block}

{candidate_block}

{reference_block}
Reply with the JSON object only."""


def judge_messages(ddl: str, question: str, candidate_sql: str,
                   gold_sql: str = "", db_id: str = "") -> list[dict[str, Any]]:
    reference = ""
    if gold_sql:
        reference = ("For reference only, a known-correct query for this question is:\n"
                     + tag_block(GOLD_SQL_TAG, gold_sql)
                     + "\nGrade the candidate on its own merits; it may differ from the "
                       "reference and still be correct.")
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            schema_block=tag_block(SCHEMA_TAG, ddl) + _db_hint(db_id),
            question_block=tag_block(QUESTION_TAG, question),
            candidate_block=tag_block(CANDIDATE_SQL_TAG, candidate_sql),
            reference_block=reference,
        )},
    ]


# --------------------------------------------------------------------------- #
# judge output parsing (shared by mock, real provider and the validator)
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

VERDICTS = ("consistent", "uncertain", "inconsistent")

#: Score thresholds for the three-way verdict. These live here, in the contract
#: module, because three call sites must agree: the mock teacher's rationale, the
#: JSON parser's repair path, and the validator's pass/fail reporting. A judge
#: whose ``verdict`` disagrees with its own ``score`` is the first thing a reviewer
#: notices in a JSONL dump, so the verdict is always *derived*, never emitted
#: independently.
CONSISTENT_AT = 0.75
INCONSISTENT_BELOW = 0.35


def verdict_for_score(score: float) -> str:
    """Map a numeric score onto the three-way verdict with fixed thresholds."""
    if score >= CONSISTENT_AT:
        return "consistent"
    if score < INCONSISTENT_BELOW:
        return "inconsistent"
    return "uncertain"


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Parse a judge reply into ``{"score","verdict","rationale"}`` or ``None``.

    Tolerates a code fence, leading prose and ``score`` as a percentage; rejects
    out-of-range or non-numeric scores rather than clamping silently, so a broken
    judge is visible in the validation report instead of looking healthy.
    """
    if not text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
    if fence:
        body = fence.group(1).strip()
    m = _JSON_OBJ_RE.search(body)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # last resort: repair trailing commas, a very common model slip
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    raw_score = obj.get("score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    # A judge that answers "85" means 85%, but "1.5" is simply out of range and
    # must be rejected rather than silently rescaled to 1.5%.
    if 2.0 <= score <= 100.0 and float(score).is_integer():
        score = score / 100.0
    if not 0.0 <= score <= 1.0:
        return None
    verdict = str(obj.get("verdict") or "").strip().lower()
    # The score is the graded quantity; the verdict is a *label* for it. If a model
    # (or a mock) emits a label that contradicts its own score, the score wins and
    # the label is recomputed, so the pair in the artifact is always coherent.
    expected = verdict_for_score(score)
    if verdict not in VERDICTS or verdict != expected:
        verdict = expected
    return {
        "score": round(score, 4),
        "verdict": verdict,
        "rationale": str(obj.get("rationale") or "").strip(),
    }
