"""Phase 3 tests: context building, demo mode, and error handling.

No test makes a live API call. LLM calls are replaced with fakes.
"""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

import ai_assistant

from ai_assistant import (
    DEFAULT_QUESTION,
    ERROR,
    MODE_DEMO,
    MODE_LIVE,
    NO_RECORD,
    OK,
    SYSTEM_PROMPT,
    AssistantConfig,
    answer_question,
    build_context,
    build_user_message,
    classify_error,
    deterministic_answer,
    extract_customer_id,
    redact,
)
from demo_data import DEMO_CUSTOMER_ID
from demo_lab import DemoLab

FAKE_KEY = "sk-ant-test-0123456789abcdef"
LIVE = AssistantConfig(provider="anthropic", model="test-model", api_key=FAKE_KEY, timeout=5)
NO_KEY = AssistantConfig(provider="anthropic", model="test-model", api_key=None, timeout=5)

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, code):
    return cls("boom", response=httpx2.Response(code, request=_REQUEST), body=None)


def _lab_with_target(defect: bool = True) -> DemoLab:
    lab = DemoLab()
    lab.set_defect(defect)
    lab.run_pipeline()
    return lab


class FakeLLM:
    """Records what the model would receive and answers from the status in it."""

    def __init__(self):
        self.calls = []

    def __call__(self, config, system, user):
        self.calls.append((system, user))
        record = json.loads(user.split("<record>")[1].split("</record>")[0])
        status = record["subscription_status"]
        return f"{'Yes' if status == 'active' else 'No'}, subscription_status is {status}."


# --- Configuration ---------------------------------------------------------------

def test_config_defaults_to_anthropic_without_key():
    config = AssistantConfig.from_env({})
    assert config.provider == "anthropic"
    assert config.model
    assert not config.live_available
    assert "ANTHROPIC_API_KEY" in config.problem


def test_config_reads_provider_model_key_and_timeout():
    config = AssistantConfig.from_env({
        "AI_PROVIDER": "OpenAI", "AI_MODEL": "some-model",
        "OPENAI_API_KEY": "sk-x", "AI_TIMEOUT_SECONDS": "7.5",
    })
    assert (config.provider, config.model, config.api_key, config.timeout) == (
        "openai", "some-model", "sk-x", 7.5)
    assert config.live_available


def test_config_openai_needs_only_a_key():
    config = AssistantConfig.from_env({"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-x"})
    assert config.live_available
    assert config.label == "openai / gpt-5-mini"


def test_config_openai_ignores_anthropic_gateway_and_key():
    config = AssistantConfig.from_env({
        "AI_PROVIDER": "openai", "ANTHROPIC_API_KEY": "a", "ANTHROPIC_AUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://localhost:1",
    })
    assert not config.live_available
    assert "OPENAI_API_KEY" in config.problem
    assert "via" not in config.label


# --- OpenAI provider with a fake client --------------------------------------------

OPENAI_LIVE = AssistantConfig(provider="openai", model="gpt-test", api_key="sk-openai-test-123456",
                              timeout=5)
_OPENAI_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _fake_openai(monkeypatch, content="No, subscription_status is expired.", refusal=None):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            message = SimpleNamespace(content=content, refusal=refusal)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    return calls


def test_openai_call_sends_system_prompt_and_record(monkeypatch):
    calls = _fake_openai(monkeypatch)
    lab = _lab_with_target(defect=True)
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, OPENAI_LIVE)

    assert answer.outcome == OK and answer.is_live_llm
    assert answer.text == "No, subscription_status is expired."
    assert answer.source_label == "openai / gpt-test"
    assert calls["client"]["api_key"] == "sk-openai-test-123456"
    assert calls["client"]["timeout"] == 5
    request = calls["create"]
    assert request["model"] == "gpt-test"
    assert request["max_completion_tokens"] == ai_assistant.MAX_OUTPUT_TOKENS
    system, user = request["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert '"subscription_status": "expired"' in user["content"]


def test_openai_refusal_and_empty_reply_are_errors(monkeypatch):
    lab = _lab_with_target()
    _fake_openai(monkeypatch, content=None, refusal="I can't help with that.")
    refused = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, OPENAI_LIVE)
    _fake_openai(monkeypatch, content="")
    empty = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, OPENAI_LIVE)
    assert (refused.error_kind, empty.error_kind) == ("refusal", "empty")
    assert refused.fallback_text and empty.fallback_text


def _openai_status_error(cls, code, body=None):
    return cls("boom", response=httpx2.Response(code, request=_OPENAI_REQUEST), body=body)


@pytest.mark.parametrize("exc, kind", [
    (openai.APITimeoutError(request=_OPENAI_REQUEST), "timeout"),
    (openai.APIConnectionError(request=_OPENAI_REQUEST), "network"),
    (_openai_status_error(openai.RateLimitError, 429), "rate_limit"),
    (_openai_status_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}), "quota"),
    (_openai_status_error(openai.AuthenticationError, 401), "auth"),
    (_openai_status_error(openai.NotFoundError, 404), "not_found"),
    (_openai_status_error(openai.InternalServerError, 503), "server"),
])
def test_openai_errors_are_classified(exc, kind):
    def failing(*_):
        raise exc

    lab = _lab_with_target()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, OPENAI_LIVE, failing)
    assert (answer.outcome, answer.error_kind) == (ERROR, kind)
    assert "sk-openai-test-123456" not in answer.error_detail


def test_config_rejects_unknown_provider_and_bad_timeout():
    config = AssistantConfig.from_env({"AI_PROVIDER": "mystery", "AI_TIMEOUT_SECONDS": "soon"})
    assert "Unsupported AI_PROVIDER" in config.problem
    assert config.timeout == 20.0


def test_config_accepts_gateway_token_and_base_url():
    config = AssistantConfig.from_env({
        "ANTHROPIC_AUTH_TOKEN": "gateway-token", "ANTHROPIC_BASE_URL": "http://localhost:53693",
    })
    assert config.live_available
    assert (config.api_key, config.auth_token) == (None, "gateway-token")
    assert config.label == "anthropic / claude-opus-5-5 via localhost:53693"
    assert "gateway-token" not in config.label


def test_gateway_vars_ignored_for_openai():
    config = AssistantConfig.from_env({
        "AI_PROVIDER": "openai", "AI_MODEL": "m", "ANTHROPIC_AUTH_TOKEN": "gateway-token",
    })
    assert config.auth_token is None
    assert "OPENAI_API_KEY" in config.problem


def test_error_detail_never_contains_gateway_token():
    config = AssistantConfig(provider="anthropic", model="m", api_key=None, timeout=5,
                             auth_token="gw-secret-token")

    def leaky(*_):
        raise RuntimeError("rejected token gw-secret-token")

    lab = _lab_with_target()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, config, leaky)
    assert "gw-secret-token" not in answer.error_detail


def test_config_label_never_contains_key():
    assert FAKE_KEY not in LIVE.label
    assert FAKE_KEY not in repr(LIVE.problem)


# --- Context building ------------------------------------------------------------

def test_extract_customer_id():
    assert extract_customer_id(DEFAULT_QUESTION) == 1042
    assert extract_customer_id("status of 1039 please") == 1039
    assert extract_customer_id("is my subscription active?") is None


def test_context_contains_only_needed_fields_from_current_target():
    lab = _lab_with_target(defect=True)
    context = build_context(lab.get_target_record(DEMO_CUSTOMER_ID))
    assert context == {
        "customer_id": 1042, "plan": "Premium",
        "subscription_status": "expired", "expiry_date": "2026-12-31",
    }
    assert "customer_name" not in context


def test_user_message_contains_record_and_question():
    message = build_user_message(DEFAULT_QUESTION, {"customer_id": 1042, "subscription_status": "active"})
    assert '"subscription_status": "active"' in message
    assert message.endswith(f"Question: {DEFAULT_QUESTION}")


def test_system_prompt_restricts_to_record():
    assert "ONLY the customer record" in SYSTEM_PROMPT
    assert "does not contain the information" in SYSTEM_PROMPT


# --- Demo mode ---------------------------------------------------------------------

def test_demo_mode_follows_target_data_and_never_calls_llm():
    def must_not_call(*_):
        raise AssertionError("demo mode must not call an LLM")

    defective = _lab_with_target(defect=True)
    clean = _lab_with_target(defect=False)
    bad = answer_question(DEFAULT_QUESTION, defective.get_target_record, MODE_DEMO, LIVE, must_not_call)
    good = answer_question(DEFAULT_QUESTION, clean.get_target_record, MODE_DEMO, LIVE, must_not_call)

    assert bad.mode == good.mode == MODE_DEMO
    assert not bad.is_live_llm
    assert bad.text.startswith("No.") and "'expired'" in bad.text
    assert good.text.startswith("Yes.") and "'active'" in good.text


def test_demo_mode_is_deterministic():
    lab = _lab_with_target()
    answers = {answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_DEMO, NO_KEY).text
               for _ in range(5)}
    assert len(answers) == 1


def test_deterministic_answer_handles_missing_status():
    assert "cannot tell" in deterministic_answer({"customer_id": 1})


def test_unknown_customer_returns_no_record_without_calling_llm():
    lab = _lab_with_target()
    fake = FakeLLM()
    for question in ("Is customer 9999 active?", "Is my subscription active?"):
        answer = answer_question(question, lab.get_target_record, MODE_LIVE, LIVE, fake)
        assert answer.outcome == NO_RECORD
        assert answer.context is None
    assert fake.calls == []


# --- Live mode with a fake LLM -----------------------------------------------------

def test_live_mode_sends_only_current_target_record():
    lab = _lab_with_target(defect=True)
    fake = FakeLLM()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, LIVE, fake)

    assert answer.outcome == OK and answer.is_live_llm
    assert answer.text.startswith("No")
    system, user = fake.calls[0]
    assert system == SYSTEM_PROMPT
    assert '"subscription_status": "expired"' in user
    assert "Ana Gelashvili" not in user   # name is not sent
    assert "1043" not in user             # other customers are not sent
    assert answer.source_label == "anthropic / test-model"


def test_question_is_truncated():
    lab = _lab_with_target()
    answer = answer_question(DEFAULT_QUESTION + " x" * 500, lab.get_target_record,
                             MODE_DEMO, NO_KEY)
    assert len(answer.question) == 300


# --- Error handling ----------------------------------------------------------------

@pytest.mark.parametrize("exc, kind", [
    (anthropic.APITimeoutError(request=_REQUEST), "timeout"),
    (anthropic.APIConnectionError(request=_REQUEST), "network"),
    (_status_error(anthropic.RateLimitError, 429), "rate_limit"),
    (_status_error(anthropic.AuthenticationError, 401), "auth"),
    (_status_error(anthropic.PermissionDeniedError, 403), "auth"),
    (_status_error(anthropic.NotFoundError, 404), "not_found"),
    (_status_error(anthropic.BadRequestError, 400), "bad_request"),
    (_status_error(anthropic.InternalServerError, 500), "server"),
    (RuntimeError("surprise"), "unknown"),
])
def test_api_errors_become_labelled_errors_with_fallback(exc, kind):
    def failing(*_):
        raise exc

    lab = _lab_with_target(defect=True)
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, LIVE, failing)
    assert classify_error(exc, "anthropic") == kind
    assert answer.outcome == ERROR
    assert answer.error_kind == kind
    assert not answer.is_live_llm
    assert answer.fallback_text.startswith("No.")
    assert answer.context["subscription_status"] == "expired"


def test_missing_key_in_live_mode_is_a_config_error_not_a_crash():
    lab = _lab_with_target()
    fake = FakeLLM()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, NO_KEY, fake)
    assert answer.outcome == ERROR
    assert answer.error_kind == "config"
    assert "ANTHROPIC_API_KEY" in answer.text
    assert fake.calls == []


def test_empty_model_reply_is_an_error():
    lab = _lab_with_target()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, LIVE,
                             lambda *_: "")
    assert (answer.outcome, answer.error_kind) == (ERROR, "empty")


def test_error_detail_never_contains_api_key():
    def leaky(*_):
        raise RuntimeError(f"bad key {FAKE_KEY} and header Bearer abc.def")

    lab = _lab_with_target()
    answer = answer_question(DEFAULT_QUESTION, lab.get_target_record, MODE_LIVE, LIVE, leaky)
    assert FAKE_KEY not in answer.error_detail
    assert "abc.def" not in answer.error_detail
    assert "[redacted]" in answer.error_detail


def test_redact_masks_key_like_strings():
    assert redact("token sk-proj-ABCDEF123456 here", None) == "token [redacted] here"
