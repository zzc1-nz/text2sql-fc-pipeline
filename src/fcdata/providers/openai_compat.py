"""OpenAI-compatible chat-completions client (the production path).

Why this shape
--------------
* **``httpx`` if present, ``urllib`` otherwise.** ``httpx`` is the nicer client
  (connection pooling, HTTP/2-capable) and is what the repo already ships, but a
  production box must not fail to import the pipeline just because the optional
  dependency is missing. Both paths share the same request body, the same retry
  policy and the same response parsing, so switching transports cannot change
  semantics.
* **A real retry policy.** 429 and 5xx are transient; 4xx (except 408/409/429) is
  not. Retries use exponential backoff with full jitter and honour ``Retry-After``
  when the server sends it. Non-retryable failures return ``ok=False`` on the
  first attempt so a bad API key costs one request, not five.
* **A disk response cache keyed by sha256(model + messages + params).** Synthesis
  is re-run constantly (ablation sweeps, debugging, report regeneration) and the
  same prompts come back every time. The cache turns a re-run into a free replay
  and makes the artifacts reproducible even against a non-deterministic API. Cache
  hits are reported as ``cached=True`` and still carry token counts so cost math
  stays comparable.
* **Usage from the response, never guessed.** ``usage.prompt_tokens`` /
  ``completion_tokens`` are used when the API returns them; the base class only
  falls back to the estimator when they are absent (some compatible servers omit
  ``usage`` on streamed or cached responses).

The client is import-safe offline: constructing it never performs I/O, and the
API key is read lazily from the environment named by ``api_key_env``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..tokens import estimate_messages_tokens, estimate_tokens
from .base import Completion, LLMProvider

#: HTTP status codes that are worth retrying
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ResponseCache:
    """Append-only JSONL response cache.

    One JSON object per line, keyed by the request fingerprint. Loading is lazy
    and tolerant: a corrupt line (partial write from a killed run) is skipped
    rather than aborting the stage. Writes are flushed and fsync'd so a crash
    mid-run cannot leave a half-written record that a later read would trust.
    """

    def __init__(self, path: str | os.PathLike[str] | None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = bool(enabled and self.path is not None)
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] | None = None
        self.hits = 0
        self.misses = 0

    # -- key -------------------------------------------------------------
    @staticmethod
    def fingerprint(model: str, messages: list[dict[str, Any]],
                    params: dict[str, Any]) -> str:
        payload = json.dumps({"model": model, "messages": messages,
                              "params": params}, sort_keys=True,
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- io --------------------------------------------------------------
    def _load(self) -> dict[str, dict[str, Any]]:
        if self._index is not None:
            return self._index
        index: dict[str, dict[str, Any]] = {}
        if self.enabled and self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = obj.get("key")
                if isinstance(key, str):
                    index[key] = obj
        self._index = index
        return index

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            hit = self._load().get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key: str, model: str, text: str, prompt_tokens: int,
            completion_tokens: int, finish_reason: str = "") -> None:
        if not self.enabled or self.path is None:
            return
        record = {"key": key, "model": model, "text": text,
                  "prompt_tokens": int(prompt_tokens),
                  "completion_tokens": int(completion_tokens),
                  "finish_reason": finish_reason, "ts": time.time()}
        with self._lock:
            self._load()[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self._load()) if self.enabled else 0}


class OpenAICompatProvider(LLMProvider):
    """Client for any ``/chat/completions`` endpoint (OpenAI, DeepSeek, vLLM, ...)."""

    kind = "openai"

    def __init__(self, *, name: str = "openai", model: str, base_url: str,
                 api_key_env: str = "OPENAI_API_KEY", role: str = "cheap",
                 ledger: Any = None, stage: str = "",
                 temperature: float = 0.7, max_tokens: int = 512,
                 prices: tuple[float, float] | None = None,
                 concurrency: int = 8, max_retries: int = 3,
                 timeout_s: float = 60.0,
                 cache_path: str | os.PathLike[str] | None = None,
                 cache_enabled: bool = True,
                 api_key: str | None = None) -> None:
        super().__init__(name=name, model=model, role=role, ledger=ledger,
                         stage=stage, temperature=temperature,
                         max_tokens=max_tokens, prices=prices)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self._api_key_override = api_key
        self.concurrency = max(1, int(concurrency))
        self.max_retries = max(0, int(max_retries))
        self.timeout_s = float(timeout_s)
        self.cache = ResponseCache(cache_path, enabled=cache_enabled)
        self._client: Any = None
        self._client_lock = threading.Lock()

    # -- key handling ----------------------------------------------------
    @property
    def api_key(self) -> str:
        if self._api_key_override is not None:
            return self._api_key_override
        return os.environ.get(self.api_key_env, "")

    def available(self) -> bool:
        """True when the provider has everything it needs to make a real call."""
        return bool(self.base_url and self.api_key)

    # -- transport -------------------------------------------------------
    def _endpoint(self) -> str:
        if not self.base_url:
            raise ValueError("base_url is empty; cannot call an OpenAI-compatible API")
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _http_post(self, url: str, payload: dict[str, Any],
                   headers: dict[str, str]) -> tuple[int, dict[str, Any], str]:
        """POST JSON and return ``(status, json_body_or_empty, error_text)``."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            import httpx  # type: ignore
        except ImportError:
            return self._post_urllib(url, body, headers)
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout_s,
                                            headers={"Content-Type": "application/json"})
        try:
            resp = self._client.post(url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            return 0, {}, f"{type(exc).__name__}: {exc}"
        try:
            parsed = resp.json()
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        if resp.status_code >= 400:
            return resp.status_code, parsed, resp.text[:500]
        return resp.status_code, parsed, ""

    def _post_urllib(self, url: str, body: bytes,
                     headers: dict[str, str]) -> tuple[int, dict[str, Any], str]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status = int(getattr(resp, "status", 200))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except OSError:
                detail = str(exc)
            return int(exc.code), _safe_json(detail), detail
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {}, f"{type(exc).__name__}: {exc}"
        return status, _safe_json(raw), ""

    # -- generation ------------------------------------------------------
    def _generate(self, messages: list[dict[str, Any]], *, task: str,
                  sample_id: str, temperature: float, max_tokens: int,
                  json_mode: bool) -> Completion:
        key = self.cache.fingerprint(self.model, messages, {
            "temperature": temperature, "max_tokens": max_tokens,
            "json_mode": bool(json_mode),
        })
        cached = self.cache.get(key)
        if cached is not None:
            return Completion(text=str(cached.get("text", "")),
                              prompt_tokens=int(cached.get("prompt_tokens") or 0),
                              completion_tokens=int(cached.get("completion_tokens") or 0),
                              ok=True, cached=True, model=self.model,
                              finish_reason=str(cached.get("finish_reason") or "cache"),
                              latency_ms=0.0, meta={"cache": "hit"})

        api_key = self.api_key
        if not api_key:
            return Completion(ok=False, model=self.model,
                              error=f"missing API key: environment variable "
                                    f"{self.api_key_env!r} is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        url = self._endpoint()

        last_error = ""
        for attempt in range(self.max_retries + 1):
            status, data, err_text = self._http_post(url, payload, headers)
            if status == 200:
                return self._parse_response(data, messages, key, json_mode)
            last_error = self._format_error(status, data, err_text)
            if status not in RETRYABLE_STATUS or attempt >= self.max_retries:
                break
            delay = self._backoff_delay(attempt, data)
            time.sleep(delay)
        return Completion(ok=False, model=self.model, error=last_error)

    def _backoff_delay(self, attempt: int, data: dict[str, Any]) -> float:
        """Exponential backoff with full jitter, honouring ``Retry-After``."""
        retry_after = _retry_after_seconds(data)
        if retry_after is not None:
            return min(30.0, max(0.0, retry_after))
        base = min(30.0, 0.6 * (2 ** attempt))
        return random.uniform(0.0, base) + 0.1

    def _parse_response(self, data: dict[str, Any], messages: list[dict[str, Any]],
                        key: str, json_mode: bool) -> Completion:
        try:
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            if isinstance(text, list):     # some servers return content parts
                text = "".join(str(p.get("text", "")) for p in text
                               if isinstance(p, dict))
        except (AttributeError, IndexError, TypeError) as exc:
            return Completion(ok=False, model=self.model,
                              error=f"malformed API response: {type(exc).__name__}: {exc}")
        finish_reason = str(choice.get("finish_reason") or "")
        usage = data.get("usage") or {}
        prompt_tokens = _as_int(usage.get("prompt_tokens"),
                                estimate_messages_tokens(messages))
        completion_tokens = _as_int(usage.get("completion_tokens"),
                                    estimate_tokens(text))
        if not text.strip():
            return Completion(ok=False, model=self.model,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              error=f"empty completion (finish_reason={finish_reason!r})")
        if finish_reason == "length":
            # Truncated output is not a usable training target; report it as a
            # soft failure so the synthesis stage can retry with more tokens.
            return Completion(ok=False, model=self.model, text=text,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              finish_reason=finish_reason,
                              error=f"output truncated at max_tokens={self.default_max_tokens}")
        self.cache.put(key, self.model, text, prompt_tokens, completion_tokens,
                       finish_reason)
        return Completion(text=text, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, ok=True,
                          model=str(data.get("model") or self.model),
                          finish_reason=finish_reason)

    @staticmethod
    def _format_error(status: int, data: dict[str, Any], err_text: str) -> str:
        detail = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")
            elif err:
                detail = str(err)
            detail = detail or str(data.get("message") or "")
        detail = detail or (err_text or "").strip()
        detail = re.sub(r"\s+", " ", detail)[:400]
        prefix = "transport error" if status == 0 else f"HTTP {status}"
        return f"{prefix}: {detail}" if detail else prefix

    # -- misc ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"base_url": self.base_url, "cache": self.cache.stats(),
                     "available": self.available()})
        return info

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _safe_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _as_int(value: Any, fallback: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return fallback


def _retry_after_seconds(data: dict[str, Any]) -> float | None:
    """Read ``Retry-After`` from a few shapes an API might use."""
    candidates: list[Any] = []
    if isinstance(data, dict):
        candidates.append(data.get("retry_after"))
        err = data.get("error")
        if isinstance(err, dict):
            candidates.append(err.get("retry_after"))
    for cand in candidates:
        try:
            return float(cand)
        except (TypeError, ValueError):
            continue
    return None
