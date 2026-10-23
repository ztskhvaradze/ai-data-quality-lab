"""In-memory demo state and the actions a presenter can take.

The UI should drive the demo only through DemoLab so every action keeps
pipeline status, validation status, and the event log consistent.
"""

from dataclasses import dataclass, field

import pandas as pd

from ai_assistant import (
    CHAT_CONTEXT_FIELDS,
    ERROR,
    MODE_DEMO,
    OK,
    AssistantAnswer,
    AssistantConfig,
    answer_question,
    resolve_customer_id,
)
from data_quality import (
    CheckResult,
    Mismatch,
    ValidationReport,
    run_basic_checks,
    run_targeted_checks,
)
from demo_data import load_source
from pipeline import PipelineRun, rebuild_target, run_pipeline

# The demo story starts with the defect switched on.
DEFECT_ENABLED_ON_RESET = True

# Display statuses shared by the UI and tests.
NOT_RUN = "NOT RUN"
SUCCEEDED = "SUCCEEDED"
PASSED = "PASSED"
FAILED = "FAILED"

# Values returned by DemoLab.next_step().
STEP_RUN_PIPELINE = "run_pipeline"
STEP_BASIC_CHECKS = "basic_checks"
STEP_TARGETED_CHECKS = "targeted_checks"
STEP_REPAIR = "repair"
STEP_DONE = "done"


@dataclass(frozen=True)
class ChatTurn:
    """One question/answer pair in the open-ended Live Chat, about any customer."""

    answer: AssistantAnswer


@dataclass
class DemoLab:
    source: pd.DataFrame = field(default_factory=load_source)
    defect_enabled: bool = DEFECT_ENABLED_ON_RESET
    target: pd.DataFrame | None = None
    last_run: PipelineRun | None = None
    # None means "not run against the current target".
    basic_results: tuple[CheckResult, ...] | None = None
    targeted_results: tuple[CheckResult, ...] | None = None
    events: list[str] = field(default_factory=list)
    chat: list[ChatTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.events:
            self._log("Demo reset: synthetic source loaded, target empty")

    def _log(self, message: str) -> None:
        self.events.append(f"{len(self.events) + 1}. {message}")

    # --- Actions --------------------------------------------------------------

    def reset(self) -> None:
        """Restore the known starting state."""
        self.source = load_source()
        self.defect_enabled = DEFECT_ENABLED_ON_RESET
        self.target = None
        self.last_run = None
        self.basic_results = None
        self.targeted_results = None
        self.events = []
        self.chat = []
        self._log("Demo reset: synthetic source loaded, target empty")

    def set_defect(self, enabled: bool) -> None:
        """Turn the deliberate transformation defect on or off for the next run."""
        if enabled == self.defect_enabled:
            return
        self.defect_enabled = enabled
        self._log(f"Deliberate defect {'ENABLED' if enabled else 'DISABLED'} for next pipeline run")

    def run_pipeline(self) -> PipelineRun:
        run, target = run_pipeline(self.source, inject_defect=self.defect_enabled)
        defect_note = "with deliberate defect" if self.defect_enabled else "without defect"
        self._apply_run(run, target, f"Pipeline run ({defect_note})")
        return run

    def repair_target(self) -> PipelineRun:
        """Fix the transformation (disable defect) and rebuild the target from source."""
        self.defect_enabled = False
        run, target = rebuild_target(self.source)
        self._apply_run(run, target, "Repair: target rebuilt from source without defect")
        return run

    def run_basic_checks(self) -> tuple[CheckResult, ...]:
        self._require_target()
        self.basic_results = run_basic_checks(self.source, self.target)
        self._log(f"Basic technical checks: {self.basic_status}")
        return self.basic_results

    def run_targeted_checks(self) -> tuple[CheckResult, ...]:
        self._require_target()
        if self.basic_results is None:
            raise ValueError("Run the basic technical checks first.")
        if not all(c.passed for c in self.basic_results):
            raise ValueError("Basic technical checks failed. Fix the structural problems first.")
        self.targeted_results = run_targeted_checks(self.source, self.target)
        detail = ""
        if self.detected_mismatches:
            ids = sorted({m.customer_id for m in self.detected_mismatches})
            detail = f" (customer {', '.join(map(str, ids))})"
        self._log(f"Business data validation: {self.targeted_status}{detail}")
        return self.targeted_results

    def run_quality_checks(self) -> ValidationReport:
        """Run basic checks, then targeted checks if the basic checks passed."""
        self.run_basic_checks()
        if all(c.passed for c in self.basic_results):
            self.run_targeted_checks()
        return ValidationReport(self.basic_results + (self.targeted_results or ()))

    def ask_chat(
        self,
        question: str,
        mode: str = MODE_DEMO,
        config: AssistantConfig | None = None,
        llm_call=None,
    ) -> ChatTurn:
        """Ask Live Chat about any customer, grounded in the current target data.

        Each message resolves its own customer by ID or name (falling back to
        the customer discussed in the previous turn), so the record sent to
        the model always comes from the live target table.
        """
        self._require_target()
        config = config or AssistantConfig.from_env()
        last_id = self.chat[-1].answer.customer_id if self.chat else None
        directory = [(int(cid), name) for cid, name in
                     zip(self.target["customer_id"], self.target["customer_name"])]
        customer_id = resolve_customer_id(question, directory, last_id)
        answer = answer_question(question, self.get_target_record, mode, config, llm_call,
                                  customer_id=customer_id, context_fields=CHAT_CONTEXT_FIELDS)
        turn = ChatTurn(answer=answer)
        self.chat.append(turn)

        who = "live LLM" if answer.mode != MODE_DEMO else "demo mode"
        if answer.outcome == OK:
            outcome = f"answered about customer {answer.customer_id} from the target record"
        elif answer.outcome == ERROR:
            outcome = f"error ({answer.error_kind})"
        else:
            outcome = "no matching customer found"
        self._log(f"Live chat asked ({who}): {outcome}")
        return turn

    def clear_chat(self) -> None:
        """Clear the Live Chat conversation without touching the rest of the demo state."""
        self.chat = []

    # --- Derived state for display --------------------------------------------

    @property
    def last_validation(self) -> ValidationReport | None:
        """Full report, only when both check layers ran against the current target."""
        if self.basic_results is None or self.targeted_results is None:
            return None
        return ValidationReport(self.basic_results + self.targeted_results)

    @property
    def pipeline_status(self) -> str:
        return NOT_RUN if self.last_run is None else self.last_run.status

    @property
    def basic_status(self) -> str:
        if self.basic_results is None:
            return NOT_RUN
        return PASSED if all(c.passed for c in self.basic_results) else FAILED

    @property
    def targeted_status(self) -> str:
        if self.targeted_results is None:
            return NOT_RUN
        return PASSED if all(c.passed for c in self.targeted_results) else FAILED

    @property
    def detected_mismatches(self) -> tuple[Mismatch, ...]:
        """Mismatches found by the targeted checks on the current target."""
        if self.targeted_results is None:
            return ()
        return tuple(m for c in self.targeted_results for m in c.mismatches)

    def next_step(self) -> str:
        """The recommended next action in the demo sequence."""
        if self.target is None or not (self.last_run and self.last_run.succeeded):
            return STEP_RUN_PIPELINE
        if self.basic_results is None:
            return STEP_BASIC_CHECKS
        if self.basic_status == FAILED:
            return STEP_REPAIR
        if self.targeted_results is None:
            return STEP_TARGETED_CHECKS
        if self.targeted_status == FAILED:
            return STEP_REPAIR
        return STEP_DONE

    def get_target_record(self, customer_id: int) -> dict | None:
        """Return one target record as a plain dict (used by the AI assistant)."""
        if self.target is None:
            return None
        rows = self.target[self.target["customer_id"] == customer_id]
        if rows.empty:
            return None
        record = rows.iloc[0].to_dict()
        record["customer_id"] = int(record["customer_id"])
        record["expiry_date"] = record["expiry_date"].date().isoformat()
        return record

    # --- Internals ------------------------------------------------------------

    def _require_target(self) -> None:
        if self.target is None:
            raise ValueError("No target data yet. Run the pipeline first.")

    def _apply_run(self, run: PipelineRun, target: pd.DataFrame | None, label: str) -> None:
        self.last_run = run
        # Earlier check results describe old data, so they no longer apply.
        self.basic_results = None
        self.targeted_results = None
        if run.succeeded:
            self.target = target
            self._log(f"{label}: execution {run.status} ({run.rows_written} rows written)")
        else:
            self._log(f"{label}: execution {run.status} ({run.error})")
