# AI Data Quality Lab

Demo app for the DataFest Tbilisi talk **"The Pipeline Is Green. The AI Is Wrong."**

All data is synthetic and the incident is a fictional, reproducible example.

## Current status: Phase 3 (Live Chat added)

| File | Purpose |
|---|---|
| `demo_data.py` | Synthetic source customers, demo customer `1042`, fixed as-of date `2026-10-15` |
| `pipeline.py` | Source → target transformation, deliberate status defect, pipeline execution status |
| `data_quality.py` | Basic technical checks and targeted business checks, validation status |
| `demo_lab.py` | `DemoLab`: demo state and actions (reset, run, toggle defect, basic checks, targeted checks, repair, event log, next step, Live Chat) |
| `ai_assistant.py` | Resolves a customer by ID or name, sends only that one retrieved record to an LLM (or builds a labelled demo-mode answer), and handles errors |
| `app.py` | Streamlit UI. Renders `DemoLab` state, contains no business logic |
| `.env.example` | Placeholder names for API configuration (copy to `.env`) |
| `.streamlit/config.toml` | Light theme and larger font for projection |
| `main.py` | Terminal walkthrough of the full scenario |
| `tests/` | pytest suite, including headless `AppTest` runs of `app.py` |

### The scenario

- Source: customer `1042` is `active` and expires on `2026-12-31`.
- With the deliberate defect on, the transformation applies a stale "legacy override" that sets `1042` to `expired`.
- Pipeline execution: **SUCCEEDED**. Basic checks (schema, non-null IDs, unique IDs, row count): **PASSED**.
- Business data validation: **FAILED**. Both targeted checks report customer `1042`, `subscription_status`, expected `active`, actual `expired`:
  - source-to-target status reconciliation
  - business rule: status must be `active` when the expiry date is on or after the as-of date
- Repair rebuilds the target from source with the defect removed, and all checks pass.
- Live Chat, asked "Is customer 1042's subscription active?": before repair it answers from the record `subscription_status = 'expired'` (wrong). After repair the same question is answered from `'active'`.

Pipeline execution status (`PipelineRun.status`) and data quality status (`ValidationReport.status`) are separate objects.

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\ZaurTskhvaradze\PycharmProjects\AI Data Quality Lab"
python -m venv .venv          # skip if .venv already exists
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

With `uv` instead: `uv venv` then `uv pip install -r requirements.txt`.

## Configure Live Chat

The app works without any API key: Live Chat then runs in **deterministic demo mode**. That mode applies a fixed template to the retrieved record, never calls an LLM, and is always labelled "SIMULATED".

To use a live LLM, copy the example file and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

With OpenAI, put these two lines in `.env`:

```
AI_PROVIDER=openai
OPENAI_API_KEY=<your key>
```

With Anthropic, set `ANTHROPIC_API_KEY=<your key>` and leave `AI_PROVIDER=anthropic`.

`.env` is listed in `.gitignore`. The key is read only from the environment or `.env` and is never displayed. The UI shows only "API key: configured".

| Variable | Default | Meaning |
|---|---|---|
| `AI_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | none | Anthropic API key for live mode |
| `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` | none | Alternative to the API key: bearer token and URL of a gateway/proxy (see below) |
| `OPENAI_API_KEY` | none | OpenAI API key for live mode |
| `AI_MODEL` | `claude-opus-5-5` (Anthropic), `gpt-5-mini` (OpenAI) | Model ID override |
| `AI_TIMEOUT_SECONDS` | `20` | Per-request timeout (one retry) |

Variables already set in the shell take precedence over `.env`. The OpenAI SDK also honours `OPENAI_BASE_URL`.

### Using the CodeMie / Claude Code session proxy

Claude Code sessions launched through CodeMie set `ANTHROPIC_BASE_URL` (a local proxy such as `http://localhost:53693`) and `ANTHROPIC_AUTH_TOKEN`. When the app starts from that environment, it detects these variables and goes live without an API key. The mode banner then reads `anthropic / claude-opus-5-5 via localhost:<port>`.

The proxy only exists while that session is running, and its port changes between sessions. So don't copy the values into `.env`. Start the app from the session (ask Claude to start it, or open a terminal that inherits the session's environment). A plain PowerShell window does not have these variables, so the app falls back to the labelled demo mode there. For the talk itself, prefer a dedicated `ANTHROPIC_API_KEY`, or rehearse with the proxy setup you will actually use on stage.

What is sent to the model: a short system prompt ("answer only from the record; say if the record is not enough") and one record — `customer_id`, `customer_name`, `plan`, `subscription_status`, `expiry_date` — for whichever customer the message resolves to. Live Chat picks that one customer by ID or name (or reuses the one from the previous turn for follow-ups) before calling the model; other customers' records are never sent. The same record is shown in the UI under a collapsed "Sources" line.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py  # UI, opens http://localhost:8501
python main.py        # terminal walkthrough
python -m pytest -v   # tests
```

If the browser does not open automatically, go to http://localhost:8501. Stop the app with `Ctrl+C`.

## Presenting the demo

The highlighted (primary) button is always the recommended next step. Buttons that would be out of order are disabled.

| Step | Click | What the audience sees |
|---|---|---|
| A | ↺ Reset demo | All three status cards: ⏸ NOT RUN. Target empty. |
| B–C | ① Run pipeline | Pipeline execution: ✅ SUCCEEDED (deliberate defect ON) |
| D | ② Run basic checks | Basic technical checks: ✅ PASSED, "Everything looks green so far…" |
| E | ③ Run targeted checks | Business data validation: ❌ FAILED, banner "The pipeline is green. The data is wrong.", detected issue: 1042 / subscription_status / expected `active` / actual `expired`, red status cell in the target table |
| F | ④ Repair target | Target rebuilt without defect; check cards return to ⏸ NOT RUN (old results never carry over to new data) |
| G | ② then ③ | Both checks ✅ PASSED, "All checks pass on the current target data." |

**💬 Live Chat** (sidebar button, enabled once the pipeline has run) opens a free-form chat about any customer in the current target data. Ask the same question before and after repair to show the contrast live:

| When | Ask in Live Chat | What the audience sees |
|---|---|---|
| After D or E | "Is customer 1042's subscription active?" | Answer grounded in `subscription_status = 'expired'` (wrong), with the record under a collapsed "Sources" line |
| After F/G | The same question again | The same question, now answered from `active` |

A few suggested-question chips covering other customers (e.g. Giorgi Kapanadze, customer 1044) are offered when the chat is empty, to show the chat isn't limited to the one scripted customer. Each message resolves its own customer by ID or name, or falls back to whichever customer the previous turn was about for follow-ups.

The sidebar holds the presenter-only toggle for the deliberate defect (Reset turns it on; Repair turns it off). It also holds the **Assistant mode** switch (Live LLM / Deterministic demo mode) that Live Chat uses. If Live is selected but not configured, demo mode is used and labelled as such. If a live call fails (timeout, network, rate limit, exhausted quota, auth, model not found, server error), the chat bubble shows a red "LIVE LLM ERROR" and the reason, followed by a deterministic fallback answer labelled "FALLBACK · not an LLM".

### Limits

- Data quality checks reduce one data-related risk. They do not guarantee correct AI answers: an LLM can still misread or misinterpret valid data.
- Live Chat only ever retrieves and sends the one record for the customer resolved from each message (plus, for follow-ups, the customer from the previous turn) — it does not search across the whole dataset or reason over multiple customers at once.
- Live model wording varies between runs. The demo relies on the factual answer following the `subscription_status` field. The system prompt calls that field authoritative, but a model may still remark on other fields, such as the expiry date.
- Rehearse with the real key and network before the talk. Keep demo mode (and screenshots) as the backup.
