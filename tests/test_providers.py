import types

import pytest
from google.genai import errors

from alphaevolve_mini import config
from alphaevolve_mini.providers import AttemptBudget, GeminiProvider, MockProvider, ProviderSetupError, make_provider


def gcfg(**provider):
    return config.from_dict({"name": "g", "provider": {"model": "test-model-id", **provider},
                             "search": {"max_retries": 2, "backoff_initial_s": 0.0}})


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def ok_response(text="```python\ndef score(job_cpu, job_mem, free_cpu, free_mem):\n    return 0\n```"):
    usage = types.SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30,
                                  cached_content_token_count=None, tool_use_prompt_token_count=None, total_token_count=150)
    cand = types.SimpleNamespace(finish_reason="STOP")
    return types.SimpleNamespace(text=text, candidates=[cand], usage_metadata=usage, model_version="v", response_id="r")


def provider(monkeypatch, outcomes, **p):
    monkeypatch.setenv("GEMINI_API_KEY", "sekret-key")
    prov = GeminiProvider(gcfg(**p), sleep=lambda s: None)
    prov.client = types.SimpleNamespace(models=FakeModels(outcomes))
    return prov


def err(cls, code, status):
    return cls(code, {"error": {"code": code, "status": status, "message": "msg sekret-key"}})


def test_requires_model_and_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderSetupError, match="GEMINI_API_KEY"):
        GeminiProvider(gcfg())
    with pytest.raises(ProviderSetupError, match="model"):
        GeminiProvider(config.from_dict({"name": "g"}))
    with pytest.raises(ProviderSetupError, match="--live"):
        make_provider("gemini", gcfg(), live=False)
    with pytest.raises(ProviderSetupError):
        make_provider("mock", gcfg(), live=True)


def test_success_records_usage(monkeypatch):
    prov = provider(monkeypatch, [ok_response()])
    b = AttemptBudget(10)
    r = prov.propose("p", "k", b)
    assert r.ok and b.used == 1 and r.attempts[0].usage["thoughts_token_count"] == 30


def test_transient_retried_at_most_twice(monkeypatch):
    prov = provider(monkeypatch, [err(errors.ServerError, 503, "UNAVAILABLE")] * 3)
    b = AttemptBudget(10)
    r = prov.propose("p", "k", b)
    assert not r.ok and not r.fatal and len(r.attempts) == 3 and b.used == 3
    prov = provider(monkeypatch, [err(errors.ClientError, 429, "RESOURCE_EXHAUSTED"), ok_response()])
    r = prov.propose("p", "k", AttemptBudget(10))
    assert r.ok and [a.status for a in r.attempts] == ["transient_error", "ok"]


@pytest.mark.parametrize("code,status", [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND"), (400, "INVALID_ARGUMENT")])
def test_fatal_errors_stop_immediately(monkeypatch, code, status):
    prov = provider(monkeypatch, [err(errors.ClientError, code, status), ok_response()])
    r = prov.propose("p", "k", AttemptBudget(10))
    assert r.fatal and len(r.attempts) == 1 and prov.client.models.calls == 1
    assert "sekret-key" not in (r.attempts[0].error_message or "")


def test_budget_counts_every_attempt(monkeypatch):
    prov = provider(monkeypatch, [err(errors.ServerError, 500, "INTERNAL")] * 3)
    b = AttemptBudget(2)
    r = prov.propose("p", "k", b)
    assert b.used == 2 and r.attempts[-1].status == "budget_exhausted" and prov.client.models.calls == 2


def test_mock_is_deterministic_and_labeled():
    m = MockProvider()
    a = m.propose("prompt", "0:1", AttemptBudget(5))
    b = m.propose("prompt", "0:1", AttemptBudget(5))
    assert a.text == b.text and "SYNTHETIC" in a.text and a.synthetic


def test_thinking_level_config(monkeypatch):
    prov = provider(monkeypatch, [], thinking_level="MEDIUM")
    tc = prov._gen_config().thinking_config
    assert tc.thinking_level.value == "MEDIUM" and tc.thinking_budget is None
    with pytest.raises(config.ConfigError):
        gcfg(thinking_level="SOME")
    with pytest.raises(config.ConfigError):
        gcfg(thinking_level="LOW", thinking_budget=100)


def test_non_text_parts_logged(monkeypatch):
    fc = types.SimpleNamespace(name="run_code", args={"code": "x"})
    part = types.SimpleNamespace(function_call=fc, text=None, thought=None)
    resp = ok_response(text=None)
    resp.candidates = [types.SimpleNamespace(finish_reason="STOP", content=types.SimpleNamespace(parts=[part]))]
    r = provider(monkeypatch, [resp]).propose("p", "k", AttemptBudget(3))
    assert r.ok and r.text is None and r.non_text_parts[0]["name"] == "run_code"
