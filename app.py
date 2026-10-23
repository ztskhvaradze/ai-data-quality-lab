"""AI Data Quality Lab - Streamlit UI with AI assistant (Phase 3).

Run with:  streamlit run app.py

All state changes go through DemoLab; this file only renders state and wires
buttons to DemoLab actions. API credentials are read from .env and never shown.
"""

from collections.abc import Callable
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_assistant import (
    ERROR,
    MODE_DEMO,
    MODE_LIVE,
    NO_RECORD,
    AssistantConfig,
)
from demo_data import AS_OF_DATE, DEMO_CUSTOMER_ID
from demo_lab import (
    FAILED,
    NOT_RUN,
    PASSED,
    STEP_BASIC_CHECKS,
    STEP_REPAIR,
    STEP_RUN_PIPELINE,
    STEP_TARGETED_CHECKS,
    SUCCEEDED,
    DemoLab,
)

# Variables already set in the environment win over values in .env.
load_dotenv()

st.set_page_config(page_title="AI Data Quality Lab", page_icon="🧪", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
.dq-subtitle {font-size: 1.25rem; color: #444; margin-top: -0.6rem;}
.dq-synthetic {background: #eef2ff; color: #1e3a8a; border: 1px solid #c7d2fe;
    border-radius: 6px; padding: 0.4rem 0.8rem; font-size: 0.95rem; margin: 0.5rem 0 1rem;}
.dq-card {border-radius: 10px; border: 2px solid; border-left-width: 12px;
    padding: 0.8rem 1.1rem; height: 100%;}
.dq-card .dq-label {font-size: 1.05rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.03em;}
.dq-card .dq-value {font-size: 2.1rem; font-weight: 800; line-height: 1.25; margin: 0.2rem 0;}
.dq-card .dq-sub {font-size: 0.95rem;}
.dq-ok   {background: #e8f5e9; border-color: #2e7d32; color: #1b4d20;}
.dq-bad  {background: #fdecea; border-color: #c62828; color: #7f1d1d;}
.dq-none {background: #f3f4f6; border-color: #9ca3af; color: #374151;}
.dq-banner {border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 1.45rem;
    font-weight: 700; margin: 1rem 0 0.5rem;}
.dq-banner small {display: block; font-size: 1rem; font-weight: 400; margin-top: 0.2rem;}
.dq-banner-bad  {background: #c62828; color: #fff;}
.dq-banner-ok   {background: #2e7d32; color: #fff;}
.dq-banner-hint {background: #fff8e1; color: #6b4e00; border: 2px dashed #e0b000;}
div.stButton > button {font-size: 1.05rem; font-weight: 600; min-height: 3rem;}
.dq-badge {display: inline-block; border-radius: 999px; padding: 0.1rem 0.7rem;
    font-size: 0.85rem; font-weight: 700; letter-spacing: 0.03em;}
.dq-badge-live  {background: #0284c7; color: #fff;}
.dq-badge-demo  {background: #9333ea; color: #fff;}
.dq-badge-error {background: #c62828; color: #fff;}
.dq-badge-info  {background: #6b7280; color: #fff;}
</style>
"""

ICONS = {SUCCEEDED: "✅", PASSED: "✅", FAILED: "❌", NOT_RUN: "⏸"}
CARD_CLASS = {SUCCEEDED: "dq-ok", PASSED: "dq-ok", FAILED: "dq-bad", NOT_RUN: "dq-none"}

ROW_HIGHLIGHT = "background-color: #fff4cc; color: #111; font-weight: 600"
CELL_MISMATCH = "background-color: #f8c9c4; color: #7f1d1d; font-weight: 800"


# --- State and actions ---------------------------------------------------------

MODE_LABELS = {MODE_LIVE: "Live LLM", MODE_DEMO: "Deterministic demo mode"}

# Spans several customers so the audience sees Live Chat isn't limited to the
# one scripted demo customer.
SUGGESTED_CHAT_QUESTIONS = [
    "Is customer 1042's subscription active?",
    "What plan does Giorgi Kapanadze have?",
    "When does customer 1044's subscription expire?",
    "Is Mariam Lomidze still subscribed?",
]


def get_lab() -> DemoLab:
    if "lab" not in st.session_state:
        st.session_state.lab = DemoLab()
    if "defect_toggle" not in st.session_state:
        st.session_state.defect_toggle = st.session_state.lab.defect_enabled
    return st.session_state.lab


def effective_mode(config: AssistantConfig) -> str:
    """Live only when the presenter chose it and credentials are configured."""
    return MODE_LIVE if st.session_state.get("ai_mode") == MODE_LIVE and config.live_available \
        else MODE_DEMO


def perform(action: Callable[[DemoLab], object]) -> None:
    """Button callback: run a DemoLab action, keep widgets in sync, surface errors."""
    lab = st.session_state.lab
    try:
        action(lab)
        st.session_state.flash = None
    except ValueError as exc:
        st.session_state.flash = str(exc)
    st.session_state.defect_toggle = lab.defect_enabled
    # Main demo-sequence buttons return focus to the main flow; only the
    # explicit Live Chat button (or widgets inside the dialog) should open it.
    st.session_state.live_chat_open = False


def on_defect_toggle() -> None:
    st.session_state.lab.set_defect(st.session_state.defect_toggle)


# --- Rendering helpers -------------------------------------------------------------

def status_card(label: str, status: str, lines: list[str]) -> str:
    sub = "<br>".join(escape(line) for line in lines)
    return (
        f'<div class="dq-card {CARD_CLASS[status]}">'
        f'<div class="dq-label">{escape(label)}</div>'
        f'<div class="dq-value">{ICONS[status]} {status}</div>'
        f'<div class="dq-sub">{sub}</div></div>'
    )


def banner(kind: str, title: str, detail: str) -> None:
    st.markdown(
        f'<div class="dq-banner dq-banner-{kind}">{escape(title)}<small>{escape(detail)}</small></div>',
        unsafe_allow_html=True,
    )


def passed_count(results) -> str:
    return f"{sum(c.passed for c in results)} of {len(results)} checks passed"


def styled_records(df: pd.DataFrame, status_col: str, mismatch_ids: set[int]):
    """Highlight the demo customer's row and any status cell flagged by a check."""
    def style_row(row: pd.Series) -> list[str]:
        styles = [ROW_HIGHLIGHT if row["customer_id"] == DEMO_CUSTOMER_ID else "" for _ in row]
        if row["customer_id"] in mismatch_ids:
            styles[list(row.index).index(status_col)] = CELL_MISMATCH
        return styles

    return df.style.apply(style_row, axis=1)


def source_display(lab: DemoLab) -> pd.DataFrame:
    return lab.source.rename(columns={"customer_name": "name"})


def target_display(lab: DemoLab) -> pd.DataFrame:
    df = lab.target.copy()
    df["expiry_date"] = df["expiry_date"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"customer_name": "name"})
    if lab.targeted_results is not None:
        flagged = {m.customer_id for m in lab.detected_mismatches}
        df["check"] = ["❌ mismatch" if cid in flagged else "✅ ok" for cid in df["customer_id"]]
    return df


# --- Page sections -------------------------------------------------------------

def render_header() -> None:
    st.title("🧪 AI Data Quality Lab")
    st.markdown(
        '<div class="dq-subtitle"><b>The Pipeline Is Green. The AI Is Wrong.</b> '
        "Finding data quality failures before they reach AI applications.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dq-synthetic">ℹ️ <b>Synthetic demo.</b> All customers and data are fictional. '
        "The defect is deliberately injected by the demo code to show one reproducible failure mode. "
        "It is not a real incident.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(lab: DemoLab, config: AssistantConfig) -> None:
    with st.sidebar:
        st.header("Presenter options")
        st.toggle(
            "Inject deliberate defect on next pipeline run",
            key="defect_toggle",
            on_change=on_defect_toggle,
        )
        st.caption(
            f"When on, the transformation applies a stale legacy override that sets customer "
            f"{DEMO_CUSTOMER_ID} to 'expired'. Reset turns it back on; Repair turns it off."
        )
        st.divider()
        st.subheader("Live Chat")
        if "ai_mode" not in st.session_state:
            st.session_state.ai_mode = MODE_LIVE if config.live_available else MODE_DEMO
        st.radio("Assistant mode", [MODE_LIVE, MODE_DEMO], key="ai_mode",
                 format_func=MODE_LABELS.get)
        if config.live_available:
            st.caption(f"Live provider: **{config.label}** · API key: configured")
        else:
            st.caption(f"Live LLM unavailable: {config.problem}")
        st.caption("Demo mode applies a fixed template to the same record. It never calls an "
                   "LLM, and its answers are always labelled as simulated.")
        if "live_chat_open" not in st.session_state:
            st.session_state.live_chat_open = False
        if st.button("💬 Live Chat", key="btn_open_chat", width="stretch",
                     help="Open a free-form chat about any customer, grounded in the current "
                          "target data.", disabled=lab.target is None):
            st.session_state.live_chat_open = True
        if lab.target is None:
            st.caption("Run the pipeline first to enable Live Chat.")
        # Re-opened on every rerun (not just the click that set the flag), so the
        # dialog survives reruns triggered by widgets inside it (chat_input, chips).
        if st.session_state.live_chat_open:
            render_live_chat_dialog(lab, config)
        st.divider()
        st.caption(f"Demo customer: **{DEMO_CUSTOMER_ID}**")
        st.caption(f"Business rule date (fixed): **{AS_OF_DATE.isoformat()}**")


def render_controls(lab: DemoLab) -> None:
    nxt = lab.next_step()
    no_target = lab.target is None
    basic_ok = lab.basic_status == PASSED

    buttons = [
        ("① Run pipeline", STEP_RUN_PIPELINE, lambda l: l.run_pipeline(), False,
         "Transform source records into the target table the AI reads."),
        ("② Run basic checks", STEP_BASIC_CHECKS, lambda l: l.run_basic_checks(), no_target,
         "Schema, IDs present and unique, row counts."),
        ("③ Run targeted checks", STEP_TARGETED_CHECKS, lambda l: l.run_targeted_checks(),
         not basic_ok,
         "Compare business values with source and business rules."),
        ("④ Repair target", STEP_REPAIR, lambda l: l.repair_target(), no_target,
         "Remove the defect and rebuild the target from source."),
    ]
    disabled_help = {
        STEP_BASIC_CHECKS: "Run the pipeline first.",
        STEP_TARGETED_CHECKS: "Run the basic checks first (they must pass).",
        STEP_REPAIR: "Run the pipeline first.",
    }

    cols = st.columns([1, 1, 1, 1, 0.8])
    for col, (label, step, action, disabled, caption) in zip(cols, buttons):
        with col:
            st.button(
                label,
                key=f"btn_{step}",
                type="primary" if step == nxt else "secondary",
                disabled=disabled,
                help=disabled_help.get(step) if disabled else caption,
                on_click=perform,
                args=(action,),
                width="stretch",
            )
            st.caption(caption)
    with cols[-1]:
        st.button(
            "↺ Reset demo",
            key="btn_reset",
            on_click=perform,
            args=(lambda l: l.reset(),),
            width="stretch",
            help="Restore the initial state: defect on, target empty, no checks run.",
        )
        st.caption("Start over from a clean state.")

    if st.session_state.get("flash"):
        st.warning(f"⚠️ {st.session_state.flash}")


def render_status(lab: DemoLab) -> None:
    run = lab.last_run
    if run is None:
        pipeline_lines = ["Pipeline has not run yet", " "]
    elif run.succeeded:
        defect = "ON (deliberate)" if run.defect_injected else "OFF"
        pipeline_lines = [f"Ran without errors · {run.rows_written} rows written",
                          f"Deliberate defect in this run: {defect}"]
    else:
        pipeline_lines = ["Technical error during transformation", run.error or ""]

    basic_lines = ["Schema · non-null IDs · unique IDs · row count"]
    basic_lines.append(passed_count(lab.basic_results) if lab.basic_results is not None
                       else "Not run against current target")

    targeted_lines = ["Status vs source · status vs expiry-date rule"]
    targeted_lines.append(passed_count(lab.targeted_results) if lab.targeted_results is not None
                          else "Not run against current target")

    c1, c2, c3 = st.columns(3)
    c1.markdown(status_card("Pipeline execution", lab.pipeline_status, pipeline_lines),
                unsafe_allow_html=True)
    c2.markdown(status_card("Basic technical checks", lab.basic_status, basic_lines),
                unsafe_allow_html=True)
    c3.markdown(status_card("Business data validation", lab.targeted_status, targeted_lines),
                unsafe_allow_html=True)

    statuses = (lab.pipeline_status, lab.basic_status, lab.targeted_status)
    if statuses == (SUCCEEDED, PASSED, FAILED):
        banner("bad", "The pipeline is green. The data is wrong.",
               "Execution succeeded and basic checks passed, but business validation found "
               "incorrect values that an AI application would consume.")
    elif statuses == (SUCCEEDED, PASSED, NOT_RUN):
        banner("hint", "Everything looks green so far…",
               "The pipeline succeeded and basic checks passed. But is the data actually correct?")
    elif statuses == (SUCCEEDED, PASSED, PASSED):
        banner("ok", "All checks pass on the current target data.",
               "Business values now match the source and the expiry-date rule.")


def render_issue(lab: DemoLab) -> None:
    if lab.targeted_status != FAILED:
        return
    st.subheader("🔎 Detected issue")
    rows = [
        {
            "customer_id": m.customer_id,
            "field": m.field,
            "expected": m.expected,
            "actual": m.actual,
            "detected by": check.name,
            "evidence": m.reason,
        }
        for check in lab.targeted_results
        for m in check.mismatches
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for cid in sorted({m.customer_id for m in lab.detected_mismatches}):
        src = lab.source[lab.source["customer_id"] == cid].iloc[0]
        actual = next(m.actual for m in lab.detected_mismatches if m.customer_id == cid)
        st.markdown(
            f"**In plain words:** the source system says customer **{cid}** is "
            f"**{src['status']}** (subscription runs until {src['expiry_date']}), but the target "
            f"table says **{actual}**. Nothing crashed and no row went missing, so the basic "
            f"checks could not see it. An AI assistant reading this target would tell the "
            f"customer their subscription has {actual}."
        )


def render_data(lab: DemoLab) -> None:
    st.subheader("📋 Source vs target records")
    st.caption(f"Customer {DEMO_CUSTOMER_ID} is highlighted. "
               "Status cells turn red only when a targeted check has flagged them.")
    flagged = {m.customer_id for m in lab.detected_mismatches}

    left, right = st.columns(2)
    with left:
        st.markdown("**Source system** (system of record)")
        st.dataframe(styled_records(source_display(lab), "status", set()),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Target table** (what the AI application will read)")
        if lab.target is None:
            st.info("Target is empty. Click **① Run pipeline** to build it.")
        else:
            st.dataframe(styled_records(target_display(lab), "subscription_status", flagged),
                         hide_index=True, width="stretch")


def render_check_details(lab: DemoLab) -> None:
    results = (lab.basic_results or ()) + (lab.targeted_results or ())
    if not results:
        return
    with st.expander("Check details"):
        for check in results:
            st.markdown(f"{ICONS[check.status]} **{check.status}** · *{check.category}* · "
                        f"{check.name}: {check.details}")


def badge(kind: str, text: str) -> str:
    return f'<span class="dq-badge dq-badge-{kind}">{escape(text)}</span>'


@st.dialog("💬 Live Chat", width="large")
def render_live_chat_dialog(lab: DemoLab, config: AssistantConfig) -> None:
    mode = effective_mode(config)
    if mode == MODE_LIVE:
        st.caption(f"🟢 Live LLM · {config.label} · each answer is grounded in one record "
                   "retrieved from the current target table")
    else:
        reason = "" if config.live_available else f" (Live LLM unavailable: {config.problem})"
        st.caption(f"🧪 Deterministic demo mode · template answers, not an LLM{reason}")

    if lab.target is None:
        st.info("Run the pipeline first so there is target data to chat about.")
        if st.button("✕ Close", key="btn_close_chat_no_target"):
            st.session_state.live_chat_open = False
            st.rerun()
        return

    st.caption("⚖️ AI answers and data validation are separate: validation checks the data, "
               "not the answer. An LLM can still misread or misinterpret correct data.")

    for turn in lab.chat:
        answer = turn.answer
        with st.chat_message("user"):
            st.write(answer.question)
        with st.chat_message("assistant"):
            if answer.outcome == NO_RECORD:
                st.markdown(badge("info", "NO RECORD · no model called"), unsafe_allow_html=True)
                st.write(answer.text)
            elif answer.outcome == ERROR:
                st.markdown(badge("error", f"LIVE LLM ERROR · {answer.source_label}"),
                            unsafe_allow_html=True)
                st.write(answer.text)
                st.caption(f"Details: {answer.error_detail}")
                st.markdown(badge("demo", "FALLBACK · deterministic demo mode, not an LLM"),
                            unsafe_allow_html=True)
                st.write(answer.fallback_text or "")
            elif answer.is_live_llm:
                st.markdown(badge("live", f"LIVE LLM RESPONSE · {answer.source_label}"),
                            unsafe_allow_html=True)
                st.write(answer.text)
            else:
                st.markdown(badge("demo", "SIMULATED · deterministic demo mode, not an LLM"),
                            unsafe_allow_html=True)
                st.write(answer.text)

            if answer.context:
                with st.expander("Sources"):
                    st.caption("Record retrieved from the target table and sent to the model:")
                    st.dataframe(pd.DataFrame([answer.context]), hide_index=True, width="stretch")

    if not lab.chat:
        st.caption("Try asking:")
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTED_CHAT_QUESTIONS):
            if cols[i % 2].button(suggestion, key=f"chat_chip_{i}", width="stretch"):
                with st.spinner("Asking…" if mode == MODE_LIVE else "Building demo answer…"):
                    lab.ask_chat(suggestion, mode=mode, config=config)
                st.rerun()

    question = st.chat_input("Ask about any customer in the database…")
    if question:
        with st.spinner("Asking…" if mode == MODE_LIVE else "Building demo answer…"):
            lab.ask_chat(question, mode=mode, config=config)
        st.rerun()

    clear_col, close_col = st.columns(2)
    with clear_col:
        if lab.chat and st.button("🗑 Clear chat", key="btn_clear_chat", width="stretch"):
            lab.clear_chat()
            st.rerun()
    with close_col:
        if st.button("✕ Close", key="btn_close_chat", width="stretch"):
            st.session_state.live_chat_open = False
            st.rerun()


def render_timeline(lab: DemoLab) -> None:
    st.subheader("🕒 Event log")
    with st.container(border=True):
        st.markdown("\n".join(f"- {escape(e)}" for e in lab.events))


def render_lesson() -> None:
    st.subheader("💡 What this demonstrates")
    st.markdown(
        "- **Pipeline succeeded** only means the code ran without a technical error.\n"
        "- **Basic checks** confirm structure: columns, IDs, row counts. "
        "A wrong *value* passes them.\n"
        "- **Targeted business checks** compare values with the source and with business "
        "rules, which is where this defect shows up.\n"
        "- AI applications trust their input data. The assistant gave a fluent answer from a "
        "wrong record and had no way to know. Catch defects before the data is released.\n\n"
        "*Limits:* data quality checks reduce some data-related risks; they do not guarantee "
        "correct AI answers. An LLM can still misread valid data. This is one controlled "
        "failure mode."
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    lab = get_lab()
    config = AssistantConfig.from_env()
    render_sidebar(lab, config)
    render_header()
    render_controls(lab)
    render_status(lab)

    main_col, side_col = st.columns([3, 1], gap="large")
    with main_col:
        render_issue(lab)
        render_data(lab)
        render_check_details(lab)
    with side_col:
        render_timeline(lab)
        render_lesson()


main()
