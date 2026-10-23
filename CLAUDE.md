# CLAUDE.md — AI Data Quality Lab

## Project purpose

Build a small, reliable Python + Streamlit application for a 15–20 minute DataFest Tbilisi talk.

**Working talk title:** *The Pipeline Is Green. The AI Is Wrong.*

**Subtitle:** Finding data quality failures before they reach AI applications.

The demo should make one point clear: a data pipeline can complete successfully and pass basic technical checks while still producing semantically incorrect data. An AI application that consumes that data may then give a plausible but incorrect answer. A targeted, business-aware data quality check can detect the defect before the data is released to the AI application.

This is a controlled educational demonstration, not a production system and not a claim that data validation guarantees AI correctness.

## Audience and presentation goal

The audience may include AI/ML practitioners, software engineers, data engineers, analysts, and technical product people. Explain the problem in accessible language, but make the engineering details credible.

The demo must tell a simple story:
1. Start with correct source data.
2. Run a pipeline that reports success.
3. Introduce a known defect during transformation.
4. Show that basic operational checks still pass.
5. Ask the AI assistant a question whose answer depends on the defective field.
6. Show the wrong answer and trace it to the data defect.
7. Run a targeted source-to-target/business-rule validation that catches the defect.
8. Repair or regenerate the target data, rerun checks, and ask the same question again.
9. End with the lesson: successful execution is not proof of correct data.

## Example scenario

Use a fictional subscription company and synthetic customer data only.

Example customer:
- Customer ID: `1042`
- Source subscription status: `active`
- Subscription expiry date: `2026-12-31`
- Defective transformed status: `expired`

The transformation bug deliberately maps this record's status incorrectly, despite the source record and expiry date indicating that it is active. Keep the rule and dates internally consistent. The precise example can be adjusted if a better deterministic scenario is found.

The pipeline should report that execution succeeded because the transformation ran without a technical exception. Basic checks such as schema, non-null IDs, and row counts may pass. A targeted comparison of relevant source and target business fields should flag the mismatch.

Do not claim the scenario is a real production incident. Label it as a synthetic, reproducible example.

## Required user experience

Create a clean, conference-friendly Streamlit interface named **AI Data Quality Lab**.

The audience should be able to see:
- Source records
- Transformed/target records
- Pipeline execution status
- Basic check results
- Targeted quality check results
- The AI assistant's question and answer
- A clear indication of the data defect and its downstream impact

Provide obvious controls for:
- Resetting the demo to its initial state
- Running the pipeline
- Introducing or removing the deliberate defect
- Running quality checks
- Repairing/rebuilding the target from source
- Asking the same question before and after remediation

Exact controls can be combined if that makes the interface simpler. Avoid unnecessary features and menus. The main demo sequence must be easy to operate live without editing code.

## Technical choices

- Python
- Streamlit for the UI
- Pandas and/or SQLite for the small synthetic dataset
- A real LLM API for the AI assistant, added only after the deterministic data-quality workflow works
- Synthetic, non-sensitive data only

Keep the first version small. Do not introduce Databricks, a vector database, a complex RAG framework, an autonomous agent, user authentication, or deployment infrastructure unless there is a clear demonstration need.

The AI assistant should retrieve the relevant customer record from the target dataset and answer based on that record. It does not need to train a model or search the internet. Keep the retrieved record and the model's evidence visible enough that the audience can understand why the answer was given.

## Critical demo reliability requirements

The demonstration must be reproducible:
- The data defect is deliberately injected by application logic; do not rely on an LLM spontaneously making a mistake.
- Use the same customer and same question before and after the data is fixed.
- Separate pipeline execution status from data quality status.
- Make the distinction explicit: “pipeline succeeded” can coexist with “targeted quality check failed.”
- The targeted check must detect the exact injected defect, and the app should show the expected and actual values.
- After repair, the targeted check should pass and the answer should reflect the corrected target data.
- Add a reset path so the demo can be repeated from a clean state.
- Handle missing API keys, API errors, and timeouts gracefully.
- Have a fallback mode for the presentation (for example, a clearly labelled deterministic simulated assistant response or prepared screenshots/recording) in case internet access or the LLM API fails. Never present a simulated response as a live LLM response.
- Test the full sequence repeatedly before the talk.

## AI behavior and honest claims

The model must be instructed to answer only from the retrieved record and say when the record does not contain enough information. Show the relevant record used as evidence.

Do not hard-code the wrong answer as if it were generated by the model. The wrong answer should result from the AI receiving the deliberately incorrect target record. If the model's wording varies, that is acceptable; the factual answer should be grounded in the status field.

For a reliable live presentation, it is acceptable to constrain the assistant to a simple question about the retrieved record and use a low-temperature setting where supported. Still, verify the actual behavior with the chosen model and API before relying on it in the talk.

State the limits accurately:
- Data quality checks can reduce certain data-related risks; they cannot guarantee that AI outputs are always correct.
- The demo shows one controlled failure mode, not all causes of AI failure.
- The sample data and incident are fictional.

## Suggested build phases

### Phase 1 — Deterministic data and quality checks
Build and test the source dataset, transformation, defect injection, basic checks, targeted source-to-target validation, and repair/reset behavior. Do not integrate an LLM yet.

Acceptance criteria:
- The pipeline completes successfully with and without the deliberate defect.
- Basic technical checks pass for the intended scenario.
- The targeted check fails on the exact injected status mismatch and shows expected vs actual values.
- Repairing/rebuilding the target makes the targeted check pass.
- The demo can be reset and repeated.

### Phase 2 — Streamlit interface
Present the data and results clearly with buttons for the complete demonstration sequence. Make state transitions understandable and avoid controls that can put the app into confusing states.

Acceptance criteria:
- A presenter can operate the entire demo from the UI without editing files.
- The sequence works after reset, including repeated runs.
- Pipeline status and data quality status are visually distinct.

### Phase 3 — AI integration
Add an LLM API call that receives the retrieved target record and answers a question about the subscription. Read credentials from environment variables or Streamlit secrets; never commit keys to source control.

Acceptance criteria:
- With defective target data, the assistant's factual answer reflects the defective record.
- With corrected target data, the same question reflects the corrected record.
- The UI shows the record supplied to the model.
- Missing credentials and API failures are handled gracefully.
- No API key or sensitive information appears in logs or the UI.

### Phase 4 — Presentation hardening
Rehearse the complete scenario, verify the expected results, improve the UI for projection, and prepare a backup.

Acceptance criteria:
- The full demo can be completed reliably in a few minutes.
- Reset restores a known starting state.
- The presenter has a fallback if the live API/network is unavailable.
- All demo content is synthetic and clearly identified as such.

## Suggested project structure

Keep the structure simple, for example:

```text
ai-data-quality-lab/
├── app.py
├── data_quality.py
├── demo_data.py
├── ai_assistant.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

It is fine to start with fewer files and split modules when useful. Do not create abstractions merely for abstraction's sake.

## Design guidance

- Make the interface readable on a projector.
- Use concise labels, large status indicators, and a visible sequence of events.
- Show evidence: source value, target value, expected value, actual value, and check outcome.
- Use clear wording such as “Pipeline execution: Succeeded” and “Business data validation: Failed.”
- Avoid relying only on green/red colors; include text and icons/labels for accessibility.
- Avoid overloading the screen with raw logs or too many metrics.
- Use a short event log or timeline to explain what happened.
- Make the central contrast immediately obvious.

## Development instructions for Claude

- First inspect the existing project and current files; do not overwrite working code blindly.
- Implement incrementally in the phases above.
- Prefer complete, runnable files over isolated snippets when making changes.
- Explain commands for Windows PowerShell when setup commands are needed.
- Keep dependencies minimal and record them in `requirements.txt`.
- Never commit `.env`, API keys, tokens, or other secrets. Include `.env.example` with placeholder names only.
- Do not claim a command or test passed unless it was actually run and its output verified.
- Add small tests for the deterministic transformation and quality-check logic where practical.
- Before changing architecture or adding dependencies, explain why they are needed.
- If a design decision could make the live demo less reliable, prioritize reproducibility and a graceful fallback over sophistication.

## Definition of done

The project is ready when a presenter can reset the app, run the pipeline, show that ordinary checks pass despite the injected defect, ask the AI a question and get an answer based on the bad target record, run a targeted check that catches the mismatch, repair the target, ask the same question again, and explain the lesson in under five minutes—all without editing code during the presentation.
