"""AI Data Quality Lab - Streamlit UI with AI assistant (Phase 3).

Run with:  streamlit run app.py

All state changes go through DemoLab; this file only renders state and wires
buttons to DemoLab actions. API credentials are read from .env and never shown.
"""

import os
import time
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
from demo_data import DEMO_CUSTOMER_ID
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

# Palette matches the talk deck's EPAM dark theme: navy background, cyan/purple/blue
# accents (#00F6FF, #B896FF, #7BA8FF, #428DFF, #00F4AB), teal = good, red = bad, orange = attention.
CSS = """
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px;}

/* --- Header ------------------------------------------------------------- */
.dq-title-row {font-size: 1.05rem; font-weight: 800; letter-spacing: 0.12em;
    color: #7BA8FF; text-transform: uppercase;}
.dq-synthetic-pill {display: inline-block; margin-left: 0.7rem; background: #15213A;
    color: #7BA8FF; border: 1px solid #2A4066; border-radius: 999px; padding: 0.12rem 0.7rem;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    vertical-align: middle;}
.dq-headline {font-size: 2rem; font-weight: 800; line-height: 1.15; margin-top: 0.15rem;
    background: linear-gradient(90deg, #00F6FF 0%, #7BA8FF 55%, #B896FF 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;}
.dq-subtitle {font-size: 1.05rem; color: #9AA7C7; margin-top: 0.15rem;}

/* --- Workflow stepper ---------------------------------------------------- */
.dq-flow-step {border-radius: 12px; border: 1.5px solid #27354E; background: #121A2C;
    padding: 0.6rem 0.6rem 0.65rem; text-align: center; min-height: 7.4rem;
    transition: border-color 0.3s, box-shadow 0.3s;}
.dq-flow-step .dq-flow-num {font-size: 0.68rem; letter-spacing: 0.1em; color: #64748B;
    font-weight: 700; text-transform: uppercase;}
.dq-flow-step .dq-flow-icon {font-size: 1.7rem; line-height: 1.3;}
.dq-flow-step .dq-flow-icon.dq-icon-ai {filter: drop-shadow(0 0 6px rgba(184,150,255,0.6));}
.dq-flow-step .dq-flow-label {font-size: 0.98rem; font-weight: 700; color: #E6EAF5;}
.dq-flow-step .dq-flow-status {font-size: 0.78rem; margin-top: 0.15rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em;}
.dq-flow-pending {opacity: 0.6;}
.dq-flow-pending .dq-flow-status {color: #64748B;}
.dq-flow-active {border-color: #00F6FF; box-shadow: 0 0 16px rgba(0, 246, 255, 0.28);}
.dq-flow-active .dq-flow-status {color: #00F6FF;}
.dq-flow-done {border-color: #00F4AB;}
.dq-flow-done .dq-flow-status {color: #00F4AB;}
.dq-flow-attention {border-color: #F87171; box-shadow: 0 0 16px rgba(248, 113, 113, 0.32);}
.dq-flow-attention .dq-flow-status {color: #F87171;}
.dq-flow-arrow {text-align: center; color: #3A4A66; font-size: 1.5rem; margin-top: 2.5rem;}
.dq-ai-callout {margin: 0.5rem 0 0.2rem; padding: 0.55rem 0.9rem; border-radius: 10px;
    background: #1A1530; border: 1px solid #3C2E66; color: #D8CBFF; font-size: 0.98rem;}
.dq-ai-callout b {color: #B896FF;}

/* --- Status cards --------------------------------------------------------- */
.dq-card {border-radius: 10px; border: 2px solid; border-left-width: 12px;
    padding: 0.7rem 1.1rem; height: 100%; transition: background 0.4s, border-color 0.4s;}
.dq-card .dq-label {font-size: 1.0rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; opacity: 0.85;}
.dq-card .dq-value {font-size: 2.05rem; font-weight: 800; line-height: 1.25; margin: 0.2rem 0;}
.dq-card .dq-sub {font-size: 1.02rem; color: #CBD5E1;}
.dq-ok   {background: #0D2A22; border-color: #00F4AB; color: #00F4AB;}
.dq-bad  {background: #2B1620; border-color: #F87171; color: #F87171;}
.dq-none {background: #172333; border-color: #2A3A52; color: #94A3B8;}

/* --- Result banner --------------------------------------------------------- */
.dq-banner {border-radius: 10px; padding: 1rem 1.25rem; margin: 0.7rem 0 0.4rem;}
.dq-banner-title {display: block; font-size: 1.55rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.015em;}
.dq-banner small {display: block; font-size: 1.05rem; font-weight: 400; margin-top: 0.3rem;
    text-transform: none; opacity: 0.95;}
.dq-banner-checks {margin-top: 0.6rem; font-size: 1rem; font-weight: 400; opacity: 0.95;}
.dq-banner-checks div {margin: 0.2rem 0;}
.dq-banner-checks b {font-weight: 700;}
.dq-banner-bad  {background: #301B1F; color: #F2F5FF; border: 2px solid #F87171;}
.dq-banner-ok   {background: #00F4AB; color: #0E1424;}
.dq-banner-hint {background: #2A1B10; color: #FF9752; border: 2px dashed #FF9752;}

/* --- Issue card ------------------------------------------------------------- */
.dq-metric-label {font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;
    letter-spacing: 0.04em;}
.dq-metric-value {font-size: 1.3rem; font-weight: 800; color: #E6EAF5; line-height: 1.3;
    word-break: break-word;}
.dq-metric-value.dq-metric-ok {color: #00F4AB;}
.dq-metric-value.dq-metric-bad {color: #F87171;}

/* --- Source / target tables -------------------------------------------------- */
.dq-table-title {font-weight: 800; font-size: 1.05rem; letter-spacing: 0.03em;}
.dq-table-sub {display: block; font-size: 0.82rem; font-weight: 400; color: #94A3B8;}
.dq-accent-source {color: #7BA8FF;}
.dq-accent-target {color: #B896FF;}

/* --- Controls / badges ------------------------------------------------------- */
div.stButton > button {font-size: 1.02rem; font-weight: 600; min-height: 2.9rem;}
div.stButton > button[kind="primary"] {color: #0E1424;}
.dq-badge {display: inline-block; border-radius: 999px; padding: 0.15rem 0.85rem;
    font-size: 0.95rem; font-weight: 700; letter-spacing: 0.03em;}
.dq-badge-live  {background: #428DFF; color: #0E1424;}
.dq-badge-demo  {background: #B896FF; color: #0E1424;}
.dq-badge-error {background: #F87171; color: #fff;}
.dq-badge-info  {background: #334155; color: #fff;}
/* Streamlit captions (sidebar explanations, table/timeline hints) default to small text
   that is hard to read from projector distance; bring them closer to body size. */
[data-testid="stCaptionContainer"], .stCaption {font-size: 1rem !important;}

/* Live Chat dialog: near-full-screen and large type so the audience can read it. */
div[data-testid="stDialog"] div[role="dialog"] {
    width: 94vw !important; max-width: 94vw !important; min-height: 85vh;
}
div[data-testid="stDialog"] [data-testid="stChatMessage"] {padding: 1rem 1.25rem;}
div[data-testid="stDialog"] [data-testid="stChatMessage"] p,
div[data-testid="stDialog"] [data-testid="stChatMessage"] li {
    font-size: 1.6rem !important; line-height: 1.5;
}
div[data-testid="stDialog"] [data-testid="stChatMessage"] .dq-badge {font-size: 1.1rem;}
div[data-testid="stDialog"] [data-testid="stChatInput"] textarea {font-size: 1.5rem !important;}
div[data-testid="stDialog"] .stButton button p {font-size: 1.3rem !important;}
div[data-testid="stDialog"] [data-testid="stCaptionContainer"] {font-size: 1.3rem !important;}

/* One-shot moments, applied only on the rerun right after the action. */
@keyframes dq-glow-bad {0%, 100% {box-shadow: 0 0 0 rgba(248, 113, 113, 0);}
    50% {box-shadow: 0 0 32px rgba(248, 113, 113, 0.8);}}
@keyframes dq-glow-ok {0%, 100% {box-shadow: 0 0 0 rgba(0, 244, 171, 0);}
    50% {box-shadow: 0 0 32px rgba(0, 244, 171, 0.8);}}
@keyframes dq-pop {0% {transform: scale(0.9); opacity: 0;} 100% {transform: scale(1); opacity: 1;}}
@keyframes dq-suspense {0%, 100% {border-color: #FF9752;} 50% {border-color: #5C3A1E;}}
.dq-reveal    {animation: dq-pop 0.5s ease-out, dq-glow-bad 1.2s ease-in-out 0.5s 3;}
.dq-celebrate {animation: dq-pop 0.5s ease-out, dq-glow-ok 1.2s ease-in-out 0.5s 2;}
.dq-banner-hint {animation: dq-suspense 1.8s ease-in-out 3;}
</style>
"""

ICONS = {SUCCEEDED: "✅", PASSED: "✅", FAILED: "❌", NOT_RUN: "⏸"}
CARD_CLASS = {SUCCEEDED: "dq-ok", PASSED: "dq-ok", FAILED: "dq-bad", NOT_RUN: "dq-none"}
BANNER_ICON = {"bad": "🚨", "hint": "👀", "ok": "✅"}

ROW_HIGHLIGHT = "background-color: #211B42; color: #D8CBFF; font-weight: 600"
CELL_MISMATCH = "background-color: #F87171; color: #FFFFFF; font-weight: 800"

BASIC_CHECK_TYPES = "Schema · non-null IDs · unique IDs · row count"
TARGETED_CHECK_TYPES = "Status vs source · status vs expiry-date rule"

# Pause between animated steps, so the audience can follow each one.
# DQ_ANIMATIONS=off removes the pauses (used by the tests).
STEP_DELAY_SECONDS = 0.0 if os.getenv("DQ_ANIMATIONS", "on").lower() == "off" else 0.5
# Longer dwell on the fully-revealed step list before it collapses to a summary line,
# so the audience has time to read it instead of watching it disappear immediately.
STEP_COLLAPSE_DELAY_SECONDS = 0.0 if os.getenv("DQ_ANIMATIONS", "on").lower() == "off" else 2.0


# --- State and actions ---------------------------------------------------------

MODE_LABELS = {MODE_LIVE: "Live LLM", MODE_DEMO: "Deterministic demo mode"}

STEP_RESET = "reset"

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
    if "live_chat_open" not in st.session_state:
        st.session_state.live_chat_open = False
    return st.session_state.lab


def effective_mode(config: AssistantConfig) -> str:
    """Live only when the presenter chose it and credentials are configured."""
    return MODE_LIVE if st.session_state.get("ai_mode") == MODE_LIVE and config.live_available \
        else MODE_DEMO


def perform(action: Callable[[DemoLab], object], step: str) -> None:
    """Button callback: run a DemoLab action, keep widgets in sync, surface errors.

    `step` is remembered for exactly one render so its animation plays once.
    """
    lab = st.session_state.lab
    try:
        action(lab)
        st.session_state.flash = None
        st.session_state.just_did = step
    except ValueError as exc:
        st.session_state.flash = str(exc)
        st.session_state.just_did = None
    if step == STEP_RESET:
        st.session_state.saw_failure = False
    st.session_state.defect_toggle = lab.defect_enabled
    # Main demo-sequence buttons return focus to the main flow; only the
    # explicit Ask AI button (or widgets inside the dialog) should open it.
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


def banner(kind: str, title: str, detail: str, effect: str = "", extra: str = "") -> None:
    icon = BANNER_ICON.get(kind, "")
    st.markdown(
        f'<div class="dq-banner dq-banner-{kind} {effect}">'
        f'<span class="dq-banner-title">{icon} {escape(title)}</span>'
        f'<small>{escape(detail)}</small>{extra}</div>',
        unsafe_allow_html=True,
    )


def check_types_banner(targeted_status: str) -> str:
    """Highlighted breakdown of what each check category covers, shown in the status banner."""
    if targeted_status == NOT_RUN:
        targeted_icon, targeted_label = "🔍", "Targeted checks next"
    else:
        targeted_icon, targeted_label = ICONS[targeted_status], "Targeted checks"
    return (
        '<div class="dq-banner-checks">'
        f'<div>{ICONS[PASSED]} <b>Basic checks:</b> {escape(BASIC_CHECK_TYPES)}</div>'
        f'<div>{targeted_icon} <b>{escape(targeted_label)}:</b> {escape(TARGETED_CHECK_TYPES)}</div>'
        '</div>'
    )


def play_steps(label: str, steps: list[str], done_label: str, state: str = "complete") -> None:
    """Reveal steps one at a time in a status box, then collapse it to a summary line."""
    with st.status(label, expanded=True) as box:
        # One element redrawn with a growing list; separate elements per step would
        # briefly show stale lines from the previous animation.
        lines = st.empty()
        for i in range(1, len(steps) + 1):
            time.sleep(STEP_DELAY_SECONDS)
            lines.markdown("\n\n".join(steps[:i]))
        time.sleep(STEP_COLLAPSE_DELAY_SECONDS)
        box.update(label=done_label, state=state, expanded=False)


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
    left, right = st.columns([6, 1], vertical_alignment="top")
    with left:
        st.markdown(
            '<div class="dq-title-row">AI Data Quality Lab '
            '<span class="dq-synthetic-pill">Synthetic demo</span></div>'
            '<div class="dq-headline">The Pipeline Is Green. The Data Is Wrong.</div>'
            '<div class="dq-subtitle">Finding data quality failures before they '
            'reach AI applications.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.button(
            "↺ Reset demo",
            key="btn_reset",
            on_click=perform,
            args=(lambda l: l.reset(), STEP_RESET),
            width="stretch",
            help="Restore the initial state: defect on, target empty, no checks run.",
        )
    st.caption("Fictional customers; the defect is deliberately injected to show one "
               "reproducible failure mode. Not a real incident.")


def render_sidebar(lab: DemoLab, config: AssistantConfig) -> None:
    with st.sidebar:
        st.toggle(
            "Inject deliberate defect on next pipeline run",
            key="defect_toggle",
            on_change=on_defect_toggle,
        )
        if "ai_mode" not in st.session_state:
            st.session_state.ai_mode = MODE_LIVE if config.live_available else MODE_DEMO
        st.radio("Assistant mode", [MODE_LIVE, MODE_DEMO], key="ai_mode",
                 format_func=MODE_LABELS.get)


def _flow_state(done: bool, active: bool, attention: bool) -> str:
    if attention:
        return "attention"
    if done:
        return "done"
    if active:
        return "active"
    return "pending"


FLOW_STATUS_TEXT = {"pending": "Pending", "active": "▶ Up next", "done": "✓ Done",
                     "attention": "! Needs attention"}


def _stage_header(step_no: int, icon: str, title: str, state: str, icon_class: str = "") -> None:
    st.markdown(
        f'<div class="dq-flow-step dq-flow-{state}">'
        f'<div class="dq-flow-num">Step {step_no}</div>'
        f'<div class="dq-flow-icon {icon_class}">{icon}</div>'
        f'<div class="dq-flow-label">{escape(title)}</div>'
        f'<div class="dq-flow-status">{FLOW_STATUS_TEXT[state]}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_workflow(lab: DemoLab, config: AssistantConfig) -> None:
    """The five-stage demo sequence: each stage card wraps the real action button."""
    nxt = lab.next_step()
    no_target = lab.target is None
    basic_ok = lab.basic_status == PASSED
    needs_repair = lab.basic_status == FAILED or lab.targeted_status == FAILED
    repaired_ever = any("Repair:" in e for e in lab.events)

    state_pipeline = _flow_state(lab.pipeline_status == SUCCEEDED,
                                  nxt == STEP_RUN_PIPELINE, lab.pipeline_status == FAILED)
    state_basic = _flow_state(lab.basic_status == PASSED,
                               nxt == STEP_BASIC_CHECKS, lab.basic_status == FAILED)
    state_ai = _flow_state(len(lab.chat) > 0, not no_target and len(lab.chat) == 0, False)
    state_targeted = _flow_state(lab.targeted_status == PASSED,
                                  nxt == STEP_TARGETED_CHECKS, lab.targeted_status == FAILED)
    state_repair = _flow_state(repaired_ever and not needs_repair,
                                nxt == STEP_REPAIR, needs_repair and nxt == STEP_REPAIR)

    cols = st.columns([2, 0.3, 2, 0.3, 2, 0.3, 2, 0.3, 2], vertical_alignment="center")

    with cols[0]:
        _stage_header(1, "📥", "Run Pipeline", state_pipeline)
        st.button("Run Pipeline", key="btn_run_pipeline", width="stretch",
                   type="primary" if nxt == STEP_RUN_PIPELINE else "secondary",
                   on_click=perform, args=(lambda l: l.run_pipeline(), STEP_RUN_PIPELINE),
                   help="Transform source records into the target table the AI reads.")
    with cols[1]:
        st.markdown('<div class="dq-flow-arrow">→</div>', unsafe_allow_html=True)
    with cols[2]:
        _stage_header(2, "🧪", "Basic Checks", state_basic)
        st.button("Run Basic Checks", key="btn_basic_checks", width="stretch",
                   type="primary" if nxt == STEP_BASIC_CHECKS else "secondary",
                   disabled=no_target, on_click=perform,
                   args=(lambda l: l.run_basic_checks(), STEP_BASIC_CHECKS),
                   help="Run the pipeline first." if no_target else
                        "Schema, IDs present and unique, row counts.")
    with cols[3]:
        st.markdown('<div class="dq-flow-arrow">→</div>', unsafe_allow_html=True)
    with cols[4]:
        _stage_header(3, "💬", "Ask AI", state_ai, icon_class="dq-icon-ai")
        if st.button("Open Live Chat", key="btn_open_chat", width="stretch",
                     type="primary" if state_ai == "active" else "secondary",
                     disabled=no_target,
                     help="Open a free-form chat about any customer, grounded in the "
                          "current target data."):
            st.session_state.live_chat_open = True
    with cols[5]:
        st.markdown('<div class="dq-flow-arrow">→</div>', unsafe_allow_html=True)
    with cols[6]:
        _stage_header(4, "🎯", "Targeted Checks", state_targeted)
        st.button("Run Targeted Checks", key="btn_targeted_checks", width="stretch",
                   type="primary" if nxt == STEP_TARGETED_CHECKS else "secondary",
                   disabled=not basic_ok, on_click=perform,
                   args=(lambda l: l.run_targeted_checks(), STEP_TARGETED_CHECKS),
                   help="Run the basic checks first (they must pass)." if not basic_ok else
                        "Compare business values with source and business rules.")
    with cols[7]:
        st.markdown('<div class="dq-flow-arrow">→</div>', unsafe_allow_html=True)
    with cols[8]:
        _stage_header(5, "🛠️", "Repair", state_repair)
        st.button("Repair Target", key="btn_repair", width="stretch",
                   type="primary" if nxt == STEP_REPAIR else "secondary",
                   disabled=no_target, on_click=perform,
                   args=(lambda l: l.repair_target(), STEP_REPAIR),
                   help="Run the pipeline first." if no_target else
                        "Remove the defect and rebuild the target from source.")

    if st.session_state.get("flash"):
        st.warning(st.session_state.flash)

    if st.session_state.live_chat_open:
        render_live_chat_dialog(lab, config)


def render_step_animation(lab: DemoLab, just_did: str | None) -> None:
    """Replay the action that just ran as visible steps. Every line comes from real results."""
    run = lab.last_run
    if just_did in (STEP_RUN_PIPELINE, STEP_REPAIR) and run is not None:
        if just_did == STEP_REPAIR:
            label, done = "Repairing target…", "Repair finished"
            steps = ["🛠️ Remove the stale legacy status override",
                     "🔄 Rebuild the target from source"]
        else:
            label, done = "Running pipeline…", "Pipeline finished"
            steps = [f"📥 Extract: read {run.rows_read} source records",
                     "🔄 Transform: normalise and map subscription status"]
        if run.succeeded:
            steps.append(f"📤 Load: wrote {run.rows_written} rows to the target table")
            play_steps(label, steps, f"{done} · execution {run.status} · "
                                     f"{run.rows_written} rows written")
        else:
            steps.append(f"❌ {run.error}")
            play_steps(label, steps, f"{done} · execution {run.status}", state="error")
    elif just_did == STEP_BASIC_CHECKS and lab.basic_results is not None:
        play_steps("Running basic technical checks…",
                   [f"{ICONS[c.status]} {c.name}" for c in lab.basic_results],
                   f"Basic checks {lab.basic_status} · {passed_count(lab.basic_results)}",
                   state="complete" if lab.basic_status == PASSED else "error")
    elif just_did == STEP_TARGETED_CHECKS and lab.targeted_results is not None:
        flagged = len({m.customer_id for m in lab.detected_mismatches})
        play_steps("Running business data validation…",
                   [f"{ICONS[c.status]} {c.name}" for c in lab.targeted_results],
                   f"Business validation {lab.targeted_status} · {flagged} record(s) flagged",
                   state="complete" if lab.targeted_status == PASSED else "error")


def render_status(lab: DemoLab, just_did: str | None = None) -> None:
    run = lab.last_run
    if run is None:
        pipeline_lines = ["Pipeline has not run yet", " "]
    elif run.succeeded:
        defect = "ON (deliberate)" if run.defect_injected else "OFF"
        pipeline_lines = [f"Ran without errors · {run.rows_written} rows written",
                          f"Deliberate defect in this run: {defect}"]
    else:
        pipeline_lines = ["Technical error during transformation", run.error or ""]

    basic_lines = [passed_count(lab.basic_results) if lab.basic_results is not None
                   else "Not run against current target"]

    targeted_lines = [passed_count(lab.targeted_results) if lab.targeted_results is not None
                      else "Not run against current target"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(status_card("⚙️ Pipeline Execution", lab.pipeline_status, pipeline_lines),
                unsafe_allow_html=True)
    c2.markdown(status_card("🧪 Basic Technical Checks", lab.basic_status, basic_lines),
                unsafe_allow_html=True)
    c3.markdown(status_card("🎯 Business Data Validation", lab.targeted_status, targeted_lines),
                unsafe_allow_html=True)

    statuses = (lab.pipeline_status, lab.basic_status, lab.targeted_status)
    just_validated = just_did == STEP_TARGETED_CHECKS
    if statuses == (SUCCEEDED, PASSED, FAILED):
        st.session_state.saw_failure = True
    elif statuses == (SUCCEEDED, PASSED, PASSED):
        # Celebrate only the fix of a failure the audience has seen, not a clean first run.
        celebrate = just_validated and st.session_state.get("saw_failure", False)
        banner("ok", "All checks pass on the current target data.",
               "Business values now match the source and the expiry-date rule.",
               extra=check_types_banner(PASSED),
               effect="dq-celebrate" if celebrate else "")


def _issue_metric(label: str, value: object, tone: str = "") -> str:
    cls = f"dq-metric-{tone}" if tone else ""
    return (
        f'<div class="dq-metric-label">{escape(label)}</div>'
        f'<div class="dq-metric-value {cls}">{escape(str(value))}</div>'
    )


def render_issue(lab: DemoLab) -> None:
    if lab.targeted_status != FAILED:
        return
    st.subheader("🚨 Detected issue")

    flagged_ids = sorted({m.customer_id for m in lab.detected_mismatches})
    for cid in flagged_ids:
        mismatches = [m for c in lab.targeted_results for m in c.mismatches
                      if m.customer_id == cid]
        rule_names = [c.name for c in lab.targeted_results
                      if any(m.customer_id == cid for m in c.mismatches)]
        first = mismatches[0]

        with st.container(border=True):
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.markdown(_issue_metric("Customer ID", cid), unsafe_allow_html=True)
            mc2.markdown(_issue_metric("Field", first.field), unsafe_allow_html=True)
            mc3.markdown(_issue_metric("Expected", first.expected, tone="ok"),
                         unsafe_allow_html=True)
            mc4.markdown(_issue_metric("Actual", first.actual, tone="bad"),
                         unsafe_allow_html=True)
            mc5.markdown(_issue_metric("Detection rule", f"{len(rule_names)} check(s)"),
                         unsafe_allow_html=True)
            st.caption(" · ".join(rule_names))

    with st.expander("Technical details"):
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


def render_data(lab: DemoLab) -> None:
    st.subheader("Source vs target records")
    st.caption(f"Customer {DEMO_CUSTOMER_ID} is highlighted. "
               "Status cells turn red only when a targeted check has flagged them.")
    flagged = {m.customer_id for m in lab.detected_mismatches}

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="dq-table-title dq-accent-source">📀 SOURCE SYSTEM'
                '<span class="dq-table-sub">System of record</span></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(styled_records(source_display(lab), "status", set()),
                         hide_index=True, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="dq-table-title dq-accent-target">🤖 TARGET TABLE'
                '<span class="dq-table-sub">What the AI application reads</span></div>',
                unsafe_allow_html=True,
            )
            if lab.target is None:
                st.info("Target is empty. Click **Run Pipeline** to build it.")
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
        st.caption(f"🟢 Live LLM · {config.label}")
    else:
        reason = "" if config.live_available else f" (Live LLM unavailable: {config.problem})"
        st.caption(f"🧪 Deterministic demo mode · template answers, not an LLM{reason}")

    if lab.target is None:
        st.info("Run the pipeline first so there is target data to chat about.")
        if st.button("✕ Close", key="btn_close_chat_no_target"):
            st.session_state.live_chat_open = False
            st.rerun()
        return

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


def render_footer(lab: DemoLab) -> None:
    st.divider()
    with st.expander("📜 Event log", expanded=False):
        st.markdown("\n".join(f"- {escape(e)}" for e in lab.events))


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    lab = get_lab()
    config = AssistantConfig.from_env()
    # Read once: any later rerun (widgets, chat dialog) must not replay the animation.
    just_did = st.session_state.pop("just_did", None)

    render_sidebar(lab, config)
    render_header()
    render_workflow(lab, config)
    render_step_animation(lab, just_did)
    render_status(lab, just_did)

    render_issue(lab)
    render_data(lab)
    render_check_details(lab)
    render_footer(lab)


main()
