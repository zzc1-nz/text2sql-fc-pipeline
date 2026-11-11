"""Stage 3 -- turn validated records into *function-calling* chat SFT data.

WHY this module is the product
------------------------------
The deliverable of this project is not "SQL data", it is *function-calling* data:
a student model must learn to answer a question by emitting a structured tool
call, not by chatting SQL back.  That makes the exact target format the core
asset, so it is defined once, here, and every consumer (trainer, inference,
evaluator) renders through the same functions.  Training and inference prompts are
byte-identical *by construction*:

    render_prompt(messages[:-1], tpl) + render_target_text(record, tpl)
        == render_messages(messages, tpl)

which the tests assert for all three templates.

Three representations, deliberately kept apart
----------------------------------------------
``messages`` (the canonical, template-free form)
    OpenAI wire format.  The assistant turn is
    ``{"role": "assistant", "content": <reasoning or "">,
       "tool_calls": [{"id": ..., "type": "function",
                       "function": {"name": "execute_sql",
                                    "arguments": "<a JSON *string*>"}}]}``.
    ``arguments`` being a JSON **string** is not cosmetic: it is what the OpenAI
    API, Qwen and Llama-3.1 tool-call formats all require, and it is what the
    ``fcdata.records.ToolSchema`` contract advertises.  Everything downstream
    (trainer masking, inference parsing) consumes this form.

``render_messages`` / ``render_prompt`` / ``render_target_text`` (the text form)
    Applies one of the three chat templates (``qwen``, ``llama3``, ``plain``).
    These are the *only* functions that know about special tokens.

``tool_schema`` (the tool form)
    The ``tools=[...]`` body passed next to the chat messages.

Per-kind assistant targets (the curriculum)
-------------------------------------------
``raw``            question -> plain tool call (the Spider gold pair).
``paraphrase``     rewritten question -> plain tool call (robustness to phrasing).
``cot``            question -> a short natural-language plan, *then* the call.
                   The plan must not contain the final SQL verbatim (that would
                   leak the answer and defeat the point of the reasoning span);
                   :func:`sanitize_reasoning` strips fenced/SQL sentences and, if
                   nothing survives, falls back to a plan derived from the query
                   structure so the CoT span is never empty.
``hard_negative``  question + the wrong SQL + its real failure symptom ->
                   the corrected tool call.  The symptom is obtained by actually
                   executing the wrong SQL, so the user turn describes a real bug
                   ("no such column: X") rather than an invented one.

Determinism
-----------
Tool-call ids are derived from ``sha1(source|kind|sql)`` rather than ``uuid4`` so
two runs of the pipeline produce byte-identical corpora (essential for the
ablation and scale-curve comparisons to be reproducible).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

from ..dbexec import execute_sql, result_match
from ..records import SFTRecord, SynthRecord, ToolSchema
from ..schema import Database
from ..sqlutils import analyze_sql, extract_sql, normalize_sql, tokenize

# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
TEMPLATES: tuple[str, ...] = ("qwen", "llama3", "plain")
DEFAULT_TEMPLATE = "qwen"

#: end-of-turn marker, i.e. the exact text that follows the assistant body
_END_OF_TURN: dict[str, str] = {
    "qwen": "<|im_end|>\n",
    "llama3": "<|eot_id|>",
    "plain": "\n\n",
}
#: document start marker (llama3 has one, the others do not)
_BEGIN: dict[str, str] = {"qwen": "", "llama3": "<|begin_of_text|>", "plain": ""}
_ROLE_LABEL = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}

#: default tool name, matching ``records.ToolSchema``
DEFAULT_TOOL_NAME = "execute_sql"


def _check_template(template: str) -> str:
    if template not in TEMPLATES:
        raise ValueError(f"unknown chat template {template!r}; expected one of {TEMPLATES}")
    return template


def _begin(template: str) -> str:
    return _BEGIN[_check_template(template)]


def _header(template: str, role: str) -> str:
    """The text that opens a turn (everything before the body)."""
    if template == "qwen":
        return f"<|im_start|>{role}\n"
    if template == "llama3":
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n"
    return f"### {_ROLE_LABEL.get(role, role.title())}:\n"


def _block(template: str, role: str, body: str) -> str:
    """A complete turn: header + body + end-of-turn marker."""
    return _header(template, role) + body + _END_OF_TURN[template]


# --------------------------------------------------------------------------- #
# tool definition
# --------------------------------------------------------------------------- #
def tool_definitions(cfg: Any = None) -> list[dict[str, Any]]:
    """The OpenAI ``tools`` list stored on every :class:`SFTRecord`.

    ``cfg.fmt.tool_name`` is honoured so a caller can rename the function without
    touching this module; the JSON-schema itself comes from the frozen
    :class:`fcdata.records.ToolSchema` contract (single source of truth).
    """
    schema = ToolSchema()
    if cfg is not None:
        schema.name = getattr(cfg.fmt, "tool_name", DEFAULT_TOOL_NAME) or DEFAULT_TOOL_NAME
    return [schema.as_openai_tool()]


def tool_schema(cfg: Any = None) -> dict[str, Any]:
    """OpenAI request-body shape: ``{"tools": [{"type": "function", ...}]}``."""
    return {"tools": tool_definitions(cfg)}


def tool_name(cfg: Any = None) -> str:
    return (getattr(cfg.fmt, "tool_name", DEFAULT_TOOL_NAME) if cfg is not None
            else DEFAULT_TOOL_NAME) or DEFAULT_TOOL_NAME


# --------------------------------------------------------------------------- #
# message construction
# --------------------------------------------------------------------------- #
_TOOL_HINT = {
    "qwen": (
        'Respond with exactly one tool call and nothing else:\n'
        '<tool_call>\n'
        '{"name": "execute_sql", "arguments": {"sql": "<SQLite SELECT statement>"}}\n'
        '</tool_call>'
    ),
    "llama3": (
        'Respond with exactly one tool call and nothing else:\n'
        '<|python_tag|>{"name": "execute_sql", "parameters": {"sql": "<SQLite SELECT statement>"}}'
    ),
    "plain": (
        'Respond with exactly one JSON tool call and nothing else:\n'
        '{"tool_calls": [{"type": "function", "function": '
        '{"name": "execute_sql", "arguments": "{\\"sql\\": \\"<SQLite SELECT statement>\\"}"}}]}'
    ),
}


def build_system_message(cfg: Any, db: Database | None = None,
                         template: str | None = None) -> dict[str, Any]:
    """System turn: task instruction + available tool + (optionally) the DDL.

    The DDL lives in the system turn and the question stays a pure question in
    the user turn, which is what makes the same builder usable for both training
    prompts and inference prompts without the user turn leaking answer-shaped
    hints.  The tool specification is included because a plain-text renderer has
    no other channel to tell the model that ``execute_sql`` exists.
    """
    tpl = _check_template(template or getattr(cfg.fmt, "chat_template", DEFAULT_TEMPLATE))
    parts = [str(getattr(cfg.fmt, "system_prompt", "")).strip()]
    parts.append("# Tool\n" + _TOOL_HINT[tpl].replace("execute_sql", tool_name(cfg)))
    if getattr(cfg.fmt, "include_schema_in_prompt", True) and db is not None:
        max_tables = int(getattr(cfg.fmt, "max_schema_tables", 12) or 0)
        ddl = db.render_ddl(max_tables)
        if ddl:
            parts.append("# Database schema\n" + ddl)
    return {"role": "system", "content": "\n\n".join(p for p in parts if p)}


def build_user_message(cfg: Any, question: str, *, bad_sql: str = "",
                       symptom: str = "", correction_note: str = "") -> dict[str, Any]:
    """User turn.

    Plain question for raw/paraphrase/cot; for a hard negative it presents the
    failing attempt (wrong SQL + the *real* symptom observed when running it) and
    asks for the corrected call -- the error-correction half of the curriculum.
    """
    q = (question or "").strip()
    if not bad_sql:
        return {"role": "user", "content": q}
    lines = [q, "", f"A previous attempt called {tool_name(cfg)} with this SQL, "
                    f"which is wrong:", "", "```sql", bad_sql.strip(), "```", ""]
    note = " ".join(x for x in (symptom, correction_note) if x).strip()
    if note:
        lines.append(f"Problem: {note}")
        lines.append("")
    lines.append(f"Call {tool_name(cfg)} again with the corrected SQL.")
    return {"role": "user", "content": "\n".join(lines)}


def build_assistant_message(cfg: Any, sql: str, *, reasoning: str = "",
                            kind: str = "raw", source_id: str = "") -> dict[str, Any]:
    """Assistant target: a real OpenAI-style tool call with JSON-string arguments.

    The SQL is normalised through :func:`fcdata.sqlutils.extract_sql` first, so
    ``extract_sql(target_sql) == target_sql`` holds for every record -- the trainer
    and the evaluator can rely on that round-trip.
    """
    clean_sql = extract_sql(sql or "") or (sql or "").strip()
    call_id = "call_" + hashlib.sha1(
        f"{source_id}|{kind}|{clean_sql}".encode("utf-8")).hexdigest()[:16]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": (reasoning or "").strip(),
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name(cfg),
                "arguments": json.dumps({"sql": clean_sql}, ensure_ascii=False),
            },
        }],
    }
    return message


# --------------------------------------------------------------------------- #
# reasoning hygiene (the CoT span must not leak the answer verbatim)
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FENCE_BLOCK = re.compile(r"```(?:sql)?.*?```", re.DOTALL | re.IGNORECASE)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains_sql(text: str, sql: str) -> bool:
    """True when ``text`` reproduces ``sql`` verbatim (modulo case/whitespace)."""
    if not sql.strip():
        return False
    t, s = _norm_text(text), _norm_text(sql).rstrip(";").strip()
    if not s:
        return False
    if s in t:
        return True
    # also catch "the query is: select a, b from t where c = 1" style leakage when
    # only the statement's body was copied (>= 25 chars of exact overlap)
    probe = s[: max(25, min(len(s), 60))]
    return len(probe) >= 25 and probe in t


def sanitize_reasoning(reasoning: str, sql: str, *, max_chars: int = 700) -> str:
    """Strip SQL from a CoT explanation, keeping the natural-language plan.

    The synthesized reasoning often *quotes* the final query; a target that spells
    out the SQL in prose and then repeats it in the tool call teaches the model to
    answer twice.  Fenced blocks and sentences containing a full ``SELECT ... FROM``
    statement are dropped; if nothing survives, the caller falls back to
    :func:`derive_plan` so the reasoning span is never empty.
    """
    if not reasoning:
        return ""
    text = _FENCE_BLOCK.sub(" ", reasoning).strip()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        s = sentence.strip()
        if not s:
            continue
        if _contains_sql(s, sql):
            continue
        if re.search(r"\bselect\b", s, re.I) and re.search(r"\bfrom\b", s, re.I):
            continue
        kept.append(s)
    out = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if _contains_sql(out, sql):
        return ""
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + " ..."
    return out


def derive_plan(sql: str, db: Database | None = None) -> str:
    """A short, natural plan derived from the query *structure* (no SQL text).

    Used when the generator's reasoning is empty or is nothing but the query
    itself: the CoT span stays informative and schema-grounded instead of being
    silently dropped (an empty reasoning turn would make ``cot`` and ``raw``
    identical, which would corrupt the ablation).
    """
    an = analyze_sql(sql, db)
    steps: list[str] = []
    if an.tables:
        steps.append("read from " + ", ".join(an.tables))
    if an.has_join:
        steps.append("join the tables on their matching keys")
    if re.search(r"\bwhere\b", sql, re.I):
        steps.append("filter the rows with the conditions stated in the question")
    if an.has_aggregate:
        steps.append("aggregate the rows" + (" per group" if an.has_group_by else ""))
    if an.has_subquery:
        steps.append("use a subquery for the inner condition")
    if an.has_set_op:
        steps.append("combine two result sets")
    if an.has_order_by:
        steps.append("sort the result" + (" and keep only the first rows" if an.has_limit else ""))
    if not steps:
        steps = ["translate the question into a single SELECT statement"]
    plan = "I need to " + "; ".join(steps) + "."
    return plan[0].upper() + plan[1:]


# --------------------------------------------------------------------------- #
# rendering: messages -> text
# --------------------------------------------------------------------------- #
def _coerce_messages(record: Any) -> list[dict[str, Any]]:
    """Accept an SFTRecord, a ``{"messages": [...]}`` dict, a message, or a list."""
    if isinstance(record, SFTRecord):
        return list(record.messages)
    if isinstance(record, dict):
        if "messages" in record and isinstance(record["messages"], list):
            return list(record["messages"])
        if "role" in record:
            return [record]
        raise ValueError("dict record has neither 'messages' nor 'role'")
    if isinstance(record, (list, tuple)):
        return list(record)
    raise TypeError(f"cannot read messages from {type(record)!r}")


def _call_arguments(call: dict[str, Any]) -> Any:
    fn = call.get("function") or call
    return fn.get("arguments", fn.get("parameters", {}))


def _assistant_body(message: dict[str, Any], template: str) -> str:
    """Body of the assistant turn in the given template's native tool-call syntax."""
    content = str(message.get("content") or "").strip()
    calls = message.get("tool_calls") or []
    if template == "plain":
        payload: dict[str, Any] = {"content": content}
        if calls:
            payload["tool_calls"] = [
                {"id": c.get("id", ""), "type": c.get("type", "function"),
                 "function": {"name": (c.get("function") or {}).get("name", DEFAULT_TOOL_NAME),
                              "arguments": _as_arg_string(c)}}
                for c in calls]
        return json.dumps(payload, ensure_ascii=False)
    parts: list[str] = []
    if content:
        parts.append(content)
    for call in calls:
        parts.append(_render_call(call, template))
    return "\n\n".join(parts)


def _as_arg_string(call: dict[str, Any]) -> str:
    """Normalise arguments to a JSON *string* (the wire-format requirement)."""
    raw = _call_arguments(call)
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def _render_call(call: dict[str, Any], template: str) -> str:
    name = (call.get("function") or {}).get("name") or call.get("name") or DEFAULT_TOOL_NAME
    arg_text = _as_arg_string(call)
    if template == "qwen":
        # Qwen2.5's chat template inserts a string `arguments` verbatim, so this
        # reproduces exactly what the tokenizer-side template would emit.
        return ('<tool_call>\n{"name": %s, "arguments": %s}\n</tool_call>'
                % (json.dumps(name), arg_text))
    # llama-3.1 canonical tool-call form: {"name": ..., "parameters": {...}}
    try:
        params: Any = json.loads(arg_text)
    except (json.JSONDecodeError, ValueError):
        params = {"sql": arg_text}
    return "<|python_tag|>" + json.dumps({"name": name, "parameters": params},
                                         ensure_ascii=False)


def render_messages(messages: Sequence[dict[str, Any]],
                    template: str = DEFAULT_TEMPLATE) -> str:
    """Render a complete conversation (including the assistant target)."""
    tpl = _check_template(template)
    return _begin(tpl) + "".join(
        _block(tpl, str(m.get("role", "user")), _message_body(m, tpl)) for m in messages)


def render_prompt(messages: Sequence[dict[str, Any]],
                  template: str = DEFAULT_TEMPLATE) -> str:
    """Render prompt-only turns and append the assistant generation prefix.

    This is the *exact* prefix the model sees at inference time; feed the result
    straight to ``generate()``.  ``render_prompt(msgs[:-1])`` followed by
    ``render_target_text(msgs)`` reproduces :func:`render_messages` byte for byte.
    """
    tpl = _check_template(template)
    return _begin(tpl) + "".join(
        _block(tpl, str(m.get("role", "user")), _message_body(m, tpl)) for m in messages
    ) + _header(tpl, "assistant")


def render_target_text(record: Any, template: str = DEFAULT_TEMPLATE) -> str:
    """Render only the assistant label text (body + end-of-turn token).

    ``record`` may be an :class:`SFTRecord`, a ``{"messages": [...]}`` dict, a
    single message dict, or a raw message list; the last message must be the
    assistant target.  The trainer uses this for the supervised span, the
    evaluator uses it to verify the format.
    """
    tpl = _check_template(template)
    messages = _coerce_messages(record)
    if not messages:
        raise ValueError("cannot render a target from an empty message list")
    last = messages[-1]
    if str(last.get("role", "")) != "assistant":
        raise ValueError(f"last message must be the assistant target, got {last.get('role')!r}")
    return _assistant_body(last, tpl) + _END_OF_TURN[tpl]


def render_record_prompt(record: Any, template: str = DEFAULT_TEMPLATE) -> str:
    """Prompt text for one SFTRecord (all turns except the assistant target)."""
    messages = _coerce_messages(record)
    return render_prompt(messages[:-1], template)


def render_supervised_pair(record: Any,
                           template: str = DEFAULT_TEMPLATE) -> tuple[str, str]:
    """``(prompt_text, target_text)`` -- the trainer's supervised span.

    Concatenating the two returns exactly :func:`render_messages`, so the label
    span is the *literal* suffix of the full conversation: the trainer can encode
    the concatenation and mask everything except the tail of length
    ``len(target_text)`` without re-deriving any offsets.
    """
    messages = _coerce_messages(record)
    return (render_prompt(messages[:-1], template),
            render_target_text(messages, template))


def _message_body(message: dict[str, Any], template: str) -> str:
    role = str(message.get("role", "user"))
    if role == "assistant":
        return _assistant_body(message, template)
    return str(message.get("content") or "")


# --------------------------------------------------------------------------- #
# parsing model output back into a tool call
# --------------------------------------------------------------------------- #
def _call_to_dict(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or call
    name = str(fn.get("name") or call.get("name") or "")
    raw_args = fn.get("arguments", fn.get("parameters", call.get("arguments")))
    arguments: dict[str, Any] = {}
    if isinstance(raw_args, dict):
        arguments = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            arguments = parsed if isinstance(parsed, dict) else {"sql": str(parsed)}
        except (json.JSONDecodeError, ValueError):
            arguments = {"sql": raw_args}
    elif isinstance(call.get("sql"), str):
        # bare {"name": ..., "sql": ...} object (a model that skips the wrapper)
        arguments = {"sql": call["sql"]}
    sql = str(arguments.get("sql") or arguments.get("query") or "").strip()
    return {"name": name, "arguments": arguments, "sql": sql,
            "call_id": str(call.get("id") or "")}


def parse_tool_call(message: Any) -> dict[str, Any] | None:
    """Extract ``{"name", "arguments", "sql", "call_id"}`` from a message or text.

    Handles the canonical wire dict, the Qwen ``<tool_call>`` wrapper, the
    Llama-3.1 ``<|python_tag|>`` prefix, a bare JSON object, and the ``plain``
    template's full OpenAI message JSON.  Returns ``None`` when there is no tool
    call (e.g. the model answered in prose).
    """
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
        if calls and isinstance(calls[0], dict):
            return _call_to_dict(calls[0])
        if isinstance(message.get("function"), dict):
            return _call_to_dict(message)
        if message.get("name") and ("arguments" in message or "parameters" in message):
            return _call_to_dict(message)
        text = str(message.get("content") or "")
    else:
        text = str(message or "")

    if not text.strip():
        return None
    # 1. explicit wrappers
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if m:
        parsed = _parse_json_object(m.group(1))
        if parsed is not None:
            return _call_to_dict(parsed)
    m = re.search(r"<\|python_tag\|>\s*(\{.*\})", text, re.DOTALL)
    if m:
        parsed = _parse_json_object(m.group(1))
        if parsed is not None:
            return _call_to_dict(parsed)
    # 2. any embedded JSON object that looks like a call
    for chunk in _iter_json_objects(text):
        parsed = _parse_json_object(chunk)
        if parsed is None:
            continue
        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list) and parsed["tool_calls"]:
            return _call_to_dict(parsed["tool_calls"][0])
        if "function" in parsed or "name" in parsed or "sql" in parsed:
            return _call_to_dict(parsed)
    return None


def assistant_sql(message: Any) -> str:
    """The SQL a model produced: tool-call argument first, prose fallback second.

    The evaluator must not depend on the model being well-behaved, so this never
    raises: a parseable call wins, otherwise :func:`fcdata.sqlutils.extract_sql`
    scans the raw text.
    """
    call = parse_tool_call(message)
    if call and call.get("sql"):
        return call["sql"]
    if isinstance(message, dict):
        text = str(message.get("content") or "")
        for c in message.get("tool_calls") or []:
            text += " " + _as_arg_string(c)
    else:
        text = str(message or "")
    return extract_sql(text)


def _iter_json_objects(text: str) -> Iterable[str]:
    """Yield balanced ``{...}`` substrings, outermost first (string-aware)."""
    n = len(text)
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
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


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- #
# prompt builders used by inference / evaluation
# --------------------------------------------------------------------------- #
def _resolve_db(cfg: Any, example: Any, db: Database | None,
                spider_data: Any) -> Database | None:
    if db is not None:
        return db
    if spider_data is None:
        return None
    db_id = getattr(example, "db_id", "") or (example.get("db_id") if isinstance(example, dict) else "")
    return spider_data.databases.get(db_id)


def build_eval_messages(cfg: Any, example: Any, db: Database | None = None, *,
                        spider_data: Any = None,
                        template: str | None = None) -> list[dict[str, Any]]:
    """System + user turns for one raw Spider example (no assistant target).

    The evaluator's prompt builder.  It is the *same* code path as training, so
    the only difference between a train and an eval prompt is the example.
    """
    tpl = template or getattr(cfg.fmt, "chat_template", DEFAULT_TEMPLATE)
    question = getattr(example, "question", "") or (
        example.get("question", "") if isinstance(example, dict) else "")
    database = _resolve_db(cfg, example, db, spider_data)
    return [build_system_message(cfg, database, tpl),
            build_user_message(cfg, str(question))]


def build_eval_prompt(cfg: Any, example: Any, db: Database | None = None, *,
                      spider_data: Any = None,
                      template: str | None = None) -> str:
    """Fully rendered inference prompt (ends with the assistant generation prefix).

    Feed the returned string directly to ``generate()``; do not append anything.
    """
    tpl = template or getattr(cfg.fmt, "chat_template", DEFAULT_TEMPLATE)
    return render_prompt(build_eval_messages(cfg, example, db, spider_data=spider_data,
                                             template=tpl), tpl)


# --------------------------------------------------------------------------- #
# one record -> messages (dispatches on kind)
# --------------------------------------------------------------------------- #
def _hard_negative_symptom(record: SynthRecord, db_path: Any) -> str:
    """Describe *what actually goes wrong* with ``bad_sql`` (real execution).

    Preferred over the generator's own description because it is verifiable: the
    user turn of an error-correction sample should state the symptom the database
    really produced.
    """
    if db_path is None:
        return ""
    bad = (record.bad_sql or "").strip()
    if not bad:
        return ""
    bad_res = execute_sql(db_path, bad, timeout_s=2.0)
    if not bad_res.ok:
        return f"SQLite rejected it with: {bad_res.error}."
    gold_res = execute_sql(db_path, record.sql, timeout_s=2.0)
    if gold_res.ok:
        match = result_match(bad_res, gold_res)
        if not match.exec_match:
            return f"It runs, but it returns the wrong rows ({match.reason})."
    return "It runs, but it does not answer the question that was asked."


def build_record_messages(cfg: Any, record: SynthRecord | Any, db: Database | None,
                          *, db_path: Any = None,
                          template: str | None = None) -> list[dict[str, Any]]:
    """Build the full message list (system, user, assistant) for one record.

    ``record`` is a :class:`SynthRecord`; the kind selects the curriculum variant.
    A raw :class:`fcdata.records.Example` is also accepted (treated as ``raw``) so
    the same builder can produce the raw Spider half of the corpus.
    """
    tpl = template or getattr(cfg.fmt, "chat_template", DEFAULT_TEMPLATE)
    kind = str(getattr(record, "kind", "raw") or "raw")
    question = str(getattr(record, "question", "") or "")
    sql = str(getattr(record, "sql", "") or "")
    source_id = str(getattr(record, "source_id", "") or getattr(record, "id", "") or "")

    reasoning = ""
    symptom = correction_note = ""
    bad_sql = ""
    if kind == "cot":
        reasoning = sanitize_reasoning(str(getattr(record, "reasoning", "") or ""), sql)
        if not reasoning:
            reasoning = derive_plan(sql, db)
    elif kind == "hard_negative":
        symptom = _hard_negative_symptom(record, db_path)
        correction_note = str(getattr(record, "correction_note", "") or "").strip()
        bad_sql = str(getattr(record, "bad_sql", "") or "").strip()

    messages = [
        build_system_message(cfg, db, tpl),
        build_user_message(cfg, question, bad_sql=bad_sql, symptom=symptom,
                           correction_note=correction_note),
        build_assistant_message(cfg, sql, reasoning=reasoning, kind=kind,
                                source_id=source_id),
    ]
    return messages


def build_raw_messages(cfg: Any, example: Any, db: Database | None,
                       *, template: str | None = None) -> list[dict[str, Any]]:
    """Messages for a plain Spider example (``kind="raw"``)."""
    return build_record_messages(cfg, example, db, template=template)


# --------------------------------------------------------------------------- #
# verification helper (used by tests and by build_sft_records' self-check)
# --------------------------------------------------------------------------- #
def validate_target(message: dict[str, Any], expected_sql: str,
                    template: str = DEFAULT_TEMPLATE) -> list[str]:
    """Check one assistant target; returns a list of problems (empty = ok).

    This is the machine-checkable statement of the product contract:
    the target must be a real tool call, ``arguments`` must be a JSON *string*,
    the ``sql`` inside it must round-trip through ``extract_sql``, and the rendered
    text must parse back to the same SQL.
    """
    problems: list[str] = []
    calls = message.get("tool_calls") or []
    if not calls:
        problems.append("no tool_calls")
        return problems
    fn = calls[0].get("function") or {}
    if fn.get("name") != DEFAULT_TOOL_NAME and not fn.get("name"):
        problems.append("missing function name")
    args = fn.get("arguments")
    if not isinstance(args, str):
        problems.append(f"arguments is {type(args).__name__}, expected JSON string")
        return problems
    try:
        parsed = json.loads(args)
    except (json.JSONDecodeError, ValueError):
        problems.append("arguments is not valid JSON")
        return problems
    sql = str(parsed.get("sql", "")) if isinstance(parsed, dict) else ""
    if not sql:
        problems.append("arguments has no sql")
        return problems
    if extract_sql(sql) != sql:
        problems.append("sql does not round-trip through extract_sql")
    if expected_sql and normalize_sql(sql) != normalize_sql(expected_sql):
        problems.append("sql differs from target_sql")
    rendered = _assistant_body(message, _check_template(template))
    parsed_back = parse_tool_call(rendered)
    if not parsed_back or not parsed_back.get("sql"):
        problems.append("rendered target is not parseable as a tool call")
    elif normalize_sql(parsed_back["sql"]) != normalize_sql(sql):
        problems.append("rendered target parses to a different sql")
    return problems
