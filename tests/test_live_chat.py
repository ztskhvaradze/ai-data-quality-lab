"""Phase 3+ tests: Live Chat grounding across any customer in the dataset.

No test makes a live API call.
"""

from ai_assistant import (
    CHAT_CONTEXT_FIELDS,
    MODE_DEMO,
    NO_RECORD,
    OK,
    AssistantConfig,
    resolve_customer_id,
)
from demo_lab import DemoLab

NO_KEY = AssistantConfig(provider="anthropic", model="test-model", api_key=None, timeout=5)

DIRECTORY = [
    (1038, "Nino Beridze"),
    (1039, "Giorgi Kapanadze"),
    (1042, "Ana Gelashvili"),
]


# --- resolve_customer_id ------------------------------------------------------------

def test_resolve_customer_id_prefers_explicit_id():
    assert resolve_customer_id("Is customer 1039 active?", DIRECTORY, last_customer_id=1042) == 1039


def test_resolve_customer_id_matches_full_name():
    assert resolve_customer_id("What plan does Giorgi Kapanadze have?", DIRECTORY) == 1039


def test_resolve_customer_id_matches_partial_name():
    assert resolve_customer_id("Is Nino still subscribed?", DIRECTORY) == 1038


def test_resolve_customer_id_falls_back_to_last_customer():
    assert resolve_customer_id("What about her plan?", DIRECTORY, last_customer_id=1042) == 1042


def test_resolve_customer_id_returns_none_when_nothing_matches():
    assert resolve_customer_id("What's the weather like?", DIRECTORY) is None


# --- DemoLab.ask_chat ----------------------------------------------------------------

def _lab_with_target() -> DemoLab:
    lab = DemoLab()
    lab.set_defect(True)
    lab.run_pipeline()
    return lab


def test_ask_chat_resolves_by_name_and_grounds_in_target_record():
    lab = _lab_with_target()
    turn = lab.ask_chat("What plan does Giorgi Kapanadze have?", mode=MODE_DEMO, config=NO_KEY)
    assert turn.answer.outcome == OK
    assert turn.answer.customer_id == 1039
    assert turn.answer.context["plan"] == "Premium"
    assert set(CHAT_CONTEXT_FIELDS) >= set(turn.answer.context)


def test_ask_chat_follow_up_reuses_previous_customer():
    lab = _lab_with_target()
    lab.ask_chat("Is customer 1044's subscription active?", mode=MODE_DEMO, config=NO_KEY)
    follow_up = lab.ask_chat("What plan do they have?", mode=MODE_DEMO, config=NO_KEY)
    assert follow_up.answer.customer_id == 1044


def test_ask_chat_reflects_the_injected_defect_for_the_demo_customer():
    lab = _lab_with_target()
    turn = lab.ask_chat("Is customer 1042's subscription active?", mode=MODE_DEMO, config=NO_KEY)
    assert turn.answer.context["subscription_status"] == "expired"
    assert turn.answer.text.startswith("No.")


def test_ask_chat_with_unresolvable_customer_returns_no_record():
    lab = _lab_with_target()
    turn = lab.ask_chat("What's the weather like today?", mode=MODE_DEMO, config=NO_KEY)
    assert turn.answer.outcome == NO_RECORD


def test_ask_chat_requires_target():
    lab = DemoLab()
    try:
        lab.ask_chat("Is customer 1042 active?")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Run the pipeline first" in str(exc)


def test_reset_clears_chat_history():
    lab = _lab_with_target()
    lab.ask_chat("Is customer 1042 active?", mode=MODE_DEMO, config=NO_KEY)
    assert lab.chat
    lab.reset()
    assert lab.chat == []


def test_clear_chat_leaves_other_state_untouched():
    lab = _lab_with_target()
    lab.ask_chat("Is customer 1042 active?", mode=MODE_DEMO, config=NO_KEY)
    lab.clear_chat()
    assert lab.chat == []
    assert lab.target is not None
