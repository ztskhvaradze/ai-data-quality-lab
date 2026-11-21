"""Headless tests of app.py using Streamlit's AppTest (no browser involved)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def html(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def page_html(at) -> str:
    """Rendered markdown without the CSS block (which names every effect class)."""
    return "\n".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style>"))


def card_values(at) -> list[str]:
    lab = at.session_state.lab
    return [lab.pipeline_status, lab.basic_status, lab.targeted_status]


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception


def test_initial_render_shows_not_run_and_disabled_checks(at):
    assert card_values(at) == ["NOT RUN", "NOT RUN", "NOT RUN"]
    assert "⏸ NOT RUN" in html(at)
    assert "✅ PASSED" not in html(at)
    assert at.button(key="btn_basic_checks").disabled
    assert at.button(key="btn_targeted_checks").disabled
    assert at.button(key="btn_repair").disabled
    assert "Synthetic demo" in html(at)


def test_full_presenter_sequence(at):
    click(at, "btn_reset")                                          # A
    click(at, "btn_run_pipeline")                                   # B, C
    assert card_values(at) == ["SUCCEEDED", "NOT RUN", "NOT RUN"]
    assert "✅ SUCCEEDED" in html(at)

    click(at, "btn_basic_checks")                                   # D
    assert card_values(at) == ["SUCCEEDED", "PASSED", "NOT RUN"]
    assert "Everything looks green so far" not in html(at)

    click(at, "btn_targeted_checks")                                # E
    assert card_values(at) == ["SUCCEEDED", "PASSED", "FAILED"]
    page = html(at)
    assert "The pipeline is green. The data is wrong." not in page
    assert "❌ FAILED" in page
    issue_table = at.dataframe[0].value
    assert set(issue_table["customer_id"]) == {1042}
    assert set(issue_table["expected"]) == {"active"}
    assert set(issue_table["actual"]) == {"expired"}

    click(at, "btn_repair")                                         # F
    assert card_values(at) == ["SUCCEEDED", "NOT RUN", "NOT RUN"]
    assert "The data is wrong" not in html(at)
    assert at.session_state.defect_toggle is False

    click(at, "btn_basic_checks")                                   # G
    click(at, "btn_targeted_checks")
    assert card_values(at) == ["SUCCEEDED", "PASSED", "PASSED"]
    assert "All checks pass on the current target data." in html(at)


def test_reset_clears_results_and_restores_defect_toggle(at):
    click(at, "btn_run_pipeline")
    click(at, "btn_basic_checks")
    click(at, "btn_targeted_checks")
    click(at, "btn_repair")
    click(at, "btn_reset")
    assert card_values(at) == ["NOT RUN", "NOT RUN", "NOT RUN"]
    assert at.session_state.defect_toggle is True
    assert at.session_state.lab.defect_enabled is True
    assert len(at.session_state.lab.events) == 1


def test_defect_toggle_off_gives_clean_run(at):
    at.toggle(key="defect_toggle").set_value(False).run()
    click(at, "btn_run_pipeline")
    click(at, "btn_basic_checks")
    click(at, "btn_targeted_checks")
    assert card_values(at) == ["SUCCEEDED", "PASSED", "PASSED"]


def test_rerunning_pipeline_hides_old_failure(at):
    click(at, "btn_run_pipeline")
    click(at, "btn_basic_checks")
    click(at, "btn_targeted_checks")
    click(at, "btn_run_pipeline")
    assert card_values(at) == ["SUCCEEDED", "NOT RUN", "NOT RUN"]
    assert "Detected issue" not in "\n".join(s.value for s in at.subheader)


# --- Phase 3: Live Chat panel (demo mode only, never a live API call) -------------

@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch):
    # Empty values also stop load_dotenv() from filling these in from a local .env.
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("DQ_ANIMATIONS", "off")


def captions(at) -> str:
    return "\n".join(c.value for c in at.caption)


def open_chat(at):
    at.button(key="btn_open_chat").click().run()
    assert not at.exception


def ask_chat(at, question):
    """Ask Live Chat a question. The dialog stays open (tracked in session_state)."""
    if not at.chat_input:
        open_chat(at)
    at.chat_input[0].set_value(question).run()
    assert not at.exception


def test_live_chat_disabled_until_pipeline_runs(at):
    assert at.button(key="btn_open_chat").disabled


def test_live_chat_defaults_to_labelled_demo_mode_without_key(at):
    click(at, "btn_run_pipeline")
    open_chat(at)
    assert "Deterministic demo mode" in captions(at)
    assert "ANTHROPIC_API_KEY" in captions(at)


def test_live_chat_answers_reflect_defect_and_repair():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "btn_run_pipeline")
    ask_chat(at, "Is customer 1042's subscription active?")
    page = "\n".join(m.value for m in at.markdown)
    assert "SIMULATED" in page and "deterministic demo mode" in page
    assert "LIVE LLM RESPONSE" not in page
    assert at.session_state.lab.chat[-1].answer.context["subscription_status"] == "expired"
    assert at.session_state.lab.chat[-1].answer.text.startswith("No.")

    click(at, "btn_repair")
    ask_chat(at, "Is customer 1042's subscription active?")
    assert at.session_state.lab.chat[-1].answer.context["subscription_status"] == "active"


def test_live_chat_resolves_other_customers_by_name():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "btn_run_pipeline")
    ask_chat(at, "What plan does Giorgi Kapanadze have?")
    turn = at.session_state.lab.chat[-1]
    assert turn.answer.customer_id == 1039
    assert turn.answer.context["plan"] == "Premium"


def test_live_chat_stays_open_and_shows_the_answer_immediately():
    # Regression test: the dialog used to close the instant a question was
    # submitted, because it was only re-invoked from inside a one-shot
    # `if st.button(...)` check instead of a persistent open/close flag.
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "btn_run_pipeline")
    ask_chat(at, "Is customer 1042's subscription active?")
    assert at.session_state.live_chat_open is True
    assert len(at.chat_message) == 2
    assert len(at.chat_input) == 1


def test_close_button_closes_the_chat():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "btn_run_pipeline")
    open_chat(at)
    at.button(key="btn_close_chat").click().run()
    assert at.session_state.live_chat_open is False
    assert len(at.chat_input) == 0


def test_selecting_live_without_key_still_uses_demo_mode():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.radio(key="ai_mode").set_value("live").run()
    click(at, "btn_run_pipeline")
    ask_chat(at, "Is customer 1042's subscription active?")
    turn = at.session_state.lab.chat[-1]
    assert turn.answer.mode == "demo"
    assert "LIVE LLM RESPONSE" not in "\n".join(m.value for m in at.markdown)


def test_reset_clears_chat_history():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "btn_run_pipeline")
    ask_chat(at, "Is customer 1042's subscription active?")
    click(at, "btn_reset")
    assert at.session_state.lab.chat == []


# --- Presentation polish: one-shot animations and decluttered layout ---------------

def test_step_animation_lists_real_check_results(at):
    click(at, "btn_run_pipeline")
    click(at, "btn_basic_checks")
    page = html(at)
    assert "✅ Customer IDs are unique" in page
    assert "✅ Row count matches source" in page

    click(at, "btn_targeted_checks")
    assert "❌ Subscription status matches source" in html(at)


def test_celebration_only_after_fixing_a_seen_failure(at):
    for key in ("btn_run_pipeline", "btn_basic_checks", "btn_targeted_checks",
                "btn_repair", "btn_basic_checks", "btn_targeted_checks"):
        click(at, key)
    assert "dq-celebrate" in page_html(at)


def test_no_celebration_on_clean_first_run(at):
    at.toggle(key="defect_toggle").set_value(False).run()
    for key in ("btn_run_pipeline", "btn_basic_checks", "btn_targeted_checks"):
        click(at, key)
    assert "All checks pass on the current target data." in html(at)
    assert "dq-celebrate" not in page_html(at)


def test_reset_forgets_the_seen_failure(at):
    for key in ("btn_run_pipeline", "btn_basic_checks", "btn_targeted_checks", "btn_reset"):
        click(at, key)
    at.toggle(key="defect_toggle").set_value(False).run()
    for key in ("btn_run_pipeline", "btn_basic_checks", "btn_targeted_checks"):
        click(at, key)
    assert "dq-celebrate" not in page_html(at)
