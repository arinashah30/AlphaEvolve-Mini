"""Model backends: Gemini (live, Google Gen AI SDK) and a deterministic mock.

Every request attempt, including retries, is charged to a shared AttemptBudget
before it is sent. Only transient errors are retried (at most `max_retries`
times, exponential backoff). Auth/billing/invalid-model errors stop the run.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field

from .config import Config
from .prompts import PROGRAM_BEGIN, PROGRAM_END

MOCK_LABEL = "SYNTHETIC MOCK FIXTURE - not model output, not experimental evidence"

TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
FATAL_HTTP = {400, 401, 403, 404}


class BudgetExhausted(Exception):
    pass


class ProviderSetupError(Exception):
    pass


class AttemptBudget:
    def __init__(self, limit: int, used: int = 0):
        self.limit = limit
        self.used = used

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> None:
        if self.used >= self.limit:
            raise BudgetExhausted(f"total request-attempt limit {self.limit} reached")
        self.used += 1


@dataclass
class Attempt:
    attempt: int
    status: str  # ok | transient_error | fatal_error | budget_exhausted
    started_at: float
    latency_s: float | None = None
    http_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    usage: dict | None = None  # None means unknown, never zero


@dataclass
class ProviderResponse:
    ok: bool
    fatal: bool
    text: str | None
    attempts: list[Attempt] = field(default_factory=list)
    finish_reason: str | None = None
    model_version: str | None = None
    response_id: str | None = None
    synthetic: bool = False
    non_text_parts: list | None = None  # e.g. stray function_call parts (logged, never executed or used)

    def to_dict(self) -> dict:
        return asdict(self)


class Provider:
    name = "base"
    live = False
    model = None

    def propose(self, prompt: str, slot_key: str, budget: AttemptBudget) -> ProviderResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------- mock

_PARAMS_RE = re.compile(r"# mock-params a=([-0-9.e]+) b=([-0-9.e]+) c=([-0-9.e]+)")


class MockProvider(Provider):
    """Deterministic synthetic proposals for offline integration testing.

    Output depends only on (prompt, slot_key), so identical prompts at the same
    slot give identical output. It perturbs a parametric family of heuristics
    around whatever program is in the prompt and sometimes emits invalid or
    crashing code to exercise failure accounting.
    """

    name = "mock"
    live = False
    model = "mock-deterministic-v1"

    def propose(self, prompt: str, slot_key: str, budget: AttemptBudget) -> ProviderResponse:
        try:
            budget.consume()
        except BudgetExhausted as exc:
            return ProviderResponse(False, False, None, [Attempt(1, "budget_exhausted", time.time(), error_message=str(exc))], synthetic=True)
        seed = int(hashlib.sha256((prompt + "\x00" + slot_key).encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        text = self._make(prompt, rng)
        usage = {"prompt_token_count": len(prompt) // 4, "candidates_token_count": len(text) // 4,
                 "thoughts_token_count": None, "cached_content_token_count": None,
                 "total_token_count": len(prompt) // 4 + len(text) // 4, "synthetic": True}
        return ProviderResponse(True, False, text, [Attempt(1, "ok", time.time(), 0.0, usage=usage)],
                                finish_reason="STOP", model_version=self.model, synthetic=True)

    def _make(self, prompt: str, rng: random.Random) -> str:
        program = prompt.split(PROGRAM_BEGIN, 1)[-1].split(PROGRAM_END, 1)[0]
        m = _PARAMS_RE.search(program)
        a, b, c = (float(x) for x in m.groups()) if m else (1.0, 1.0, 0.0)
        roll = rng.random()
        header = f"[{MOCK_LABEL}]\n"
        if roll < 0.12:
            return header + "Invalid on purpose.\n```python\nimport math\ndef score(job_cpu, job_mem, free_cpu, free_mem):\n    return -math.fabs(free_cpu - job_cpu)\n```\n"
        if roll < 0.17:
            return header + "Crashes on purpose.\n```python\ndef score(job_cpu, job_mem, free_cpu, free_mem):\n    return 1 / (free_cpu - free_cpu)\n```\n"
        if roll < 0.20:
            return header + "No code block on purpose."
        a = round(max(0.05, a + rng.gauss(0, 0.35)), 3)
        b = round(max(0.05, b + rng.gauss(0, 0.35)), 3)
        c = round(max(0.0, c + rng.gauss(0, 0.4)), 3)
        return header + (
            "Weighted residuals plus a balance penalty.\n```python\n"
            "def score(job_cpu, job_mem, free_cpu, free_mem):\n"
            f"    # mock-params a={a} b={b} c={c}\n"
            "    rc = (free_cpu - job_cpu) / 1000\n"
            "    rm = (free_mem - job_mem) / 1000\n"
            f"    return -({a} * rc + {b} * rm) - {c} * abs(rc - rm)\n"
            "```\n"
        )


# --------------------------------------------------------------------------- gemini


def _non_text_parts(cand) -> list | None:
    """Summarize non-text parts (e.g. a stray function_call) so failures are visible in logs."""
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    out = []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None:
            out.append({"kind": "function_call", "name": getattr(fc, "name", None),
                        "args_preview": str(getattr(fc, "args", None))[:2000]})
        elif getattr(part, "text", None) is None and not getattr(part, "thought", False):
            out.append({"kind": "other"})
    return out or None


def _usage_dict(meta) -> dict | None:
    if meta is None:
        return None
    keys = ("prompt_token_count", "candidates_token_count", "thoughts_token_count",
            "cached_content_token_count", "tool_use_prompt_token_count", "total_token_count")
    return {k: getattr(meta, k, None) for k in keys}


class GeminiProvider(Provider):
    name = "gemini"
    live = True

    def __init__(self, cfg: Config, sleep=time.sleep):
        p = cfg.provider
        if not p.model:
            raise ProviderSetupError("provider.model is not set in the config; set an exact current Gemini model ID")
        key = os.environ.get(p.api_key_env)
        if not key:
            raise ProviderSetupError(f"environment variable {p.api_key_env} is not set")
        from google import genai
        from google.genai import types

        self._types = types
        self._key = key
        self.model = p.model
        self.cfg = cfg
        self._sleep = sleep
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=int(p.request_timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),  # we own retries and counting
            ),
        )

    def _gen_config(self):
        p, t = self.cfg.provider, self._types
        kwargs = dict(temperature=p.temperature, max_output_tokens=p.max_output_tokens, candidate_count=1)
        if p.thinking_budget is not None:
            kwargs["thinking_config"] = t.ThinkingConfig(thinking_budget=p.thinking_budget)
        elif p.thinking_level is not None:
            kwargs["thinking_config"] = t.ThinkingConfig(thinking_level=t.ThinkingLevel(p.thinking_level))
        return t.GenerateContentConfig(**kwargs)

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "<redacted>")[:500]

    def propose(self, prompt: str, slot_key: str, budget: AttemptBudget) -> ProviderResponse:
        from google.genai import errors

        try:
            import httpx
            transient_exc: tuple = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
        except ImportError:  # pragma: no cover
            transient_exc = ()

        attempts: list[Attempt] = []
        max_retries = self.cfg.search.max_retries
        for n in range(1, max_retries + 2):
            try:
                budget.consume()
            except BudgetExhausted as exc:
                attempts.append(Attempt(n, "budget_exhausted", time.time(), error_message=str(exc)))
                return ProviderResponse(False, False, None, attempts)
            started = time.time()
            t0 = time.monotonic()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=self._gen_config())
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                kind = "transient_error" if (code in TRANSIENT_HTTP or (code or 0) >= 500) else "fatal_error"
                attempts.append(Attempt(n, kind, started, time.monotonic() - t0, code, type(exc).__name__,
                                        self._scrub(f"{getattr(exc, 'status', '')} {getattr(exc, 'message', '')}")))
            except transient_exc as exc:
                attempts.append(Attempt(n, "transient_error", started, time.monotonic() - t0, None,
                                        type(exc).__name__, self._scrub(str(exc))))
            except Exception as exc:  # unknown failure: stop rather than spend budget blindly
                attempts.append(Attempt(n, "fatal_error", started, time.monotonic() - t0, None,
                                        type(exc).__name__, self._scrub(str(exc))))
            else:
                cand = (resp.candidates or [None])[0]
                finish = getattr(cand, "finish_reason", None) if cand else None
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # non-text parts are recorded separately below
                        text = resp.text
                except Exception:
                    text = None
                attempts.append(Attempt(n, "ok", started, time.monotonic() - t0, 200, usage=_usage_dict(resp.usage_metadata)))
                return ProviderResponse(True, False, text, attempts,
                                        finish_reason=str(finish) if finish is not None else None,
                                        model_version=getattr(resp, "model_version", None),
                                        response_id=getattr(resp, "response_id", None),
                                        non_text_parts=_non_text_parts(cand) if cand else None)
            if attempts[-1].status == "fatal_error":
                return ProviderResponse(False, True, None, attempts)
            if n <= max_retries:
                self._sleep(self.cfg.search.backoff_initial_s * (2 ** (n - 1)))
        return ProviderResponse(False, False, None, attempts)


def make_provider(name: str, cfg: Config, live: bool) -> Provider:
    if name == "mock":
        if live:
            raise ProviderSetupError("--live cannot be used with the mock provider")
        return MockProvider()
    if name == "gemini":
        if not live:
            raise ProviderSetupError("the gemini provider needs the explicit --live flag")
        return GeminiProvider(cfg)
    raise ProviderSetupError(f"unknown provider {name!r}")
