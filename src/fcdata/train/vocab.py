"""Word-level tokenizer for the CPU reference SFT backend.

Why not a BPE tokenizer?
-----------------------
The reference backend exists so the *whole* pipeline can be trained and evaluated
offline on a laptop CPU in minutes. A subword tokenizer would need ``tokenizers``
or ``sentencepiece`` (not installed) or a hand-written BPE trainer (a lot of code
for no scientific gain here). A **word-level** tokenizer over a 2-4k vocabulary
is:

* fast -- one dict lookup per token, no merge loop,
* deterministic -- the vocabulary is a pure function of the corpus and the
  requested size (frequency desc, then alphabetical), so two runs produce
  byte-identical ids,
* faithful to SQL -- the interesting structure of a text2sql corpus is *lexical*
  (identifiers, keywords, literals, punctuation), not morphological. Keeping
  ``count``, ``student``, ``>=`` and ``'`` as distinct atomic symbols makes the
  causal-transformer learning problem honest: the model has to learn SQL syntax
  from token order, exactly like a real model has to learn it from subwords.

Tokenization rules
------------------
1. Multi-character operators (``>=``, ``<=``, ``<>``, ``!=``, ``||``) are single
   tokens -- splitting them would destroy the only multi-char symbols SQL has.
2. Numeric literals (``3.14``) stay together; a decimal point inside a number is
   part of the number, not punctuation.
3. Identifiers/words are ``[A-Za-z_][A-Za-z0-9_$]*`` (SQL-ish identifiers).
4. **Everything else non-space is isolated** as its own token: ``( ) , . = < > ' "
   ; * + - / % [ ] { } : | ?`` and the chat-template markers ``<|im_start|>``.
   Punctuation isolation is what lets ``decode`` put the SQL back together
   exactly; it is also why ``table.column`` and ``'a string'`` survive a
   round-trip.

Special tokens (ids are fixed so a checkpoint never has to store them twice):
``<pad>=0 <bos>=1 <eos>=2 <mask>=3 <unk>=4``. ``<unk>`` is not in the task
spec but is *required*: at evaluation time the dev split (and any model output)
can contain words that were not in the training corpus, and a tokenizer that
cannot encode them would either crash or silently drop information.

The tokenizer is serialisable (``to_dict``/``from_dict``/``save``/``load``) so it
travels inside ``model.pt`` and the evaluator never has to re-fit it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"

#: Order matters: ids are positions in this tuple.
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, UNK_TOKEN)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
UNK_ID = 4

#: Multi-character SQL operators, longest first so ``>=`` wins over ``>``.
#: ``<|`` and ``|>`` are the chat-template marker halves (``<|im_start|>``); keeping
#: them atomic is what lets the tokenizer rebuild template markers exactly.
MULTI_CHAR_OPS: tuple[str, ...] = ("||", ">=", "<=", "<>", "!=", "<|", "|>")

_TOKEN_RE = re.compile(
    r"(?:>=|<=|<>|!=|\|\||<\|"       # multi-char operators + template markers
    r"|\|>)"
    r"|(?:\d+(?:\.\d+)?)"            # numeric literal, keeps 3.14 together
    r"|(?:[A-Za-z_][A-Za-z0-9_$]*)"  # word / SQL identifier
    r"|(?:\S)"                       # any other single non-space character
)

_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

#: Punctuation that hugs the token before it (no space in front).
_NO_SPACE_BEFORE = frozenset({")", "]", "}", ",", ";", ".", ":", "|>"})
#: Punctuation after which the next token hugs (no space behind). ``:`` is here
#: so JSON's ``"key":"value"`` survives; SQL does not use a bare ``:``.
_NO_SPACE_AFTER = frozenset({"(", "[", "{", ".", "$", "@", "<|", "|>", ":"})
#: Tokens an **opening** quote hugs when it directly follows them (``{"sql":``,
#: ``('A',``). Everything else gets a space before the quote, so ``= 'Physics'``
#: reads naturally. Quotes themselves are handled by the state machine below,
#: not by these sets -- an opening and a closing quote need opposite spacing, so
#: a single character set cannot express both.
_QUOTE_HUGS_AFTER = frozenset({"(", "[", "{", ",", ":", "<|", "|>"})

#: SQL functions whose ``(`` should hug the name, so generated SQL reads
#: ``COUNT(*)`` rather than ``COUNT (*)``. A curated set (not "any identifier")
#: because ``SELECT (a)`` must *not* become ``SELECT(a)`` -- SQLite would read
#: that as a call to an unknown function named ``SELECT``.
_FUNC_TOKENS = frozenset({
    "count", "sum", "avg", "min", "max", "total", "abs", "round", "length",
    "lower", "upper", "substr", "trim", "ltrim", "rtrim", "replace", "instr",
    "coalesce", "ifnull", "nullif", "cast", "date", "datetime", "strftime",
    "julianday", "printf", "group_concat", "distinct",
})


# --------------------------------------------------------------------------- #
# tokenization primitives
# --------------------------------------------------------------------------- #
def split_tokens(text: str) -> list[str]:
    """Split raw text into atomic tokens (no vocabulary lookup).

    Special tokens (``<pad>`` etc.) are kept whole so that a corpus containing
    literal ``<mask>`` markers never gets shredded into ``<`` ``mask`` ``>``.
    """
    if not text:
        return []
    out: list[str] = []
    for part in _SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIAL_TOKENS:
            out.append(part)
        else:
            out.extend(_TOKEN_RE.findall(part))
    return out


def detokenize(tokens: Sequence[str]) -> str:
    """Rebuild readable text from tokens.

    This is the inverse of :func:`split_tokens` and the reason the tokenizer is
    usable at all: the evaluator decodes generated ids and feeds the result to
    ``extract_sql`` / ``json.loads``, so a tokenizer that loses whitespace would
    silently corrupt every prediction.

    The output is **canonical**, not byte-identical to the input. That is a
    deliberate, documented choice:

    * Whitespace is not information in SQL *or* in JSON, so runs of spaces and
      newlines are canonicalised to a single space (newlines become spaces).
    * Spacing *inside* a quoted region follows the same rules as outside, which
      is what makes ``"sql":"SELECT * FROM student"`` and ``'New York'`` come
      back with their internal spaces intact (a literal's value includes them).
    * The one thing that cannot be recovered is leading/trailing whitespace
      *within* a literal (``'  padded  '``) or a literal that intentionally
      contains a newline. Neither occurs in the SQL this pipeline produces, and
      the round-trip test pins the canonical form.

    Rules, in priority order:

    1. A closing quote and closing punctuation (``)``, ``]``, ``,``, ``;``, ``.``,
       ``:``) hug the preceding token.
    2. An opening quote and opening punctuation (``(``, ``[``, ``{``) hug the
       following token; a SQL function name hugs its ``(`` -- but only for a
       curated set, because ``SELECT (a)`` must stay ``SELECT (a)``.
    3. Everything else is separated by a single space.
    """
    out: list[str] = []
    #: Delimiters we are currently inside, innermost last. A stack (not a single
    #: char) because tool-call JSON nests SQL literals inside JSON strings:
    #: ``{"sql":"SELECT * WHERE name = 'New York'"}`` is ``"`` -> ``'`` -> ``"``.
    stack: list[str] = []
    hug_next = False         # next token is the first content token of a literal

    def emit(tok: str) -> None:
        """Append ``tok`` applying the spacing rules."""
        if not out:
            out.append(tok)
            return
        prev = out[-1]
        if tok in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            out.append(tok)
        elif tok == "(" and prev.lower() in _FUNC_TOKENS:
            out.append(tok)
        else:
            out.append(" ")
            out.append(tok)

    for tok in tokens:
        if tok in ("'", '"'):
            if stack and tok == stack[-1]:
                out.append(tok)                 # closing delimiter hugs content
                stack.pop()
                hug_next = False
            elif not stack:
                stack.append(tok)
                hug_next = True
                if out and out[-1] in _QUOTE_HUGS_AFTER:
                    out.append(tok)
                else:
                    emit(tok)
            else:
                # nested opener: an SQL literal inside a JSON string
                stack.append(tok)
                hug_next = True
                emit(tok)
            continue

        if hug_next:
            out.append(tok)
            hug_next = False
        else:
            emit(tok)

    return "".join(out).strip()


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
class WordTokenizer:
    """Deterministic word-level tokenizer with a fixed special-token prefix."""

    def __init__(self, tokens: Sequence[str] | None = None) -> None:
        toks = list(tokens) if tokens else list(SPECIAL_TOKENS)
        self.tokens: list[str] = toks
        self._stoi: dict[str, int] = {t: i for i, t in enumerate(toks)}
        if len(self._stoi) != len(toks):
            raise ValueError("tokenizer vocabulary contains duplicate tokens")

    # -- construction ----------------------------------------------------
    def fit(self, corpus_texts: Iterable[str], vocab_size: int = 4096) -> "WordTokenizer":
        """Fit the vocabulary on ``corpus_texts``.

        Ordering: special tokens first (fixed ids), then corpus tokens by
        ``(-frequency, token)`` so the result is a pure function of the corpus --
        no hash ordering, no randomness, reproducible across machines.
        """
        counts: Counter[str] = Counter()
        for text in corpus_texts:
            counts.update(split_tokens(text))
        for special in SPECIAL_TOKENS:
            counts.pop(special, None)

        budget = max(len(SPECIAL_TOKENS), int(vocab_size)) - len(SPECIAL_TOKENS)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen = [tok for tok, _ in ranked[:budget]]
        self.tokens = list(SPECIAL_TOKENS) + chosen
        self._stoi = {t: i for i, t in enumerate(self.tokens)}
        return self

    @classmethod
    def from_texts(cls, corpus_texts: Iterable[str], vocab_size: int = 4096) -> "WordTokenizer":
        return cls().fit(corpus_texts, vocab_size)

    # -- properties ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def mask_id(self) -> int:
        return MASK_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def __len__(self) -> int:
        return len(self.tokens)

    # -- encode / decode -------------------------------------------------
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False,
               max_len: int = 0) -> list[int]:
        """Encode text to ids. Unknown tokens map to ``<unk>``."""
        ids = [self._stoi.get(tok, UNK_ID) for tok in split_tokens(text or "")]
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        if max_len and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True,
               detokenize_output: bool = True) -> str:
        """Decode ids back to text.

        ``skip_special`` drops ``<pad>``/``<bos>``/``<eos>``/``<mask>``; ``<unk>``
        is kept as the literal ``<unk>`` so a caller can see that something was
        lost instead of silently getting a hole in the SQL.
        """
        toks: list[str] = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.tokens):
                continue
            tok = self.tokens[i]
            if skip_special and tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN):
                continue
            toks.append(tok)
        return detokenize(toks) if detokenize_output else " ".join(toks)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "type": "word", "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WordTokenizer":
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokenizer payload is missing a non-empty 'tokens' list")
        if list(tokens[: len(SPECIAL_TOKENS)]) != list(SPECIAL_TOKENS):
            raise ValueError("tokenizer payload has an unexpected special-token prefix")
        return cls(tokens=[str(t) for t in tokens])

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def build_tokenizer(corpus_texts: Iterable[str], vocab_size: int = 4096) -> WordTokenizer:
    """One-liner used by the trainer and by tests."""
    return WordTokenizer().fit(corpus_texts, vocab_size)
