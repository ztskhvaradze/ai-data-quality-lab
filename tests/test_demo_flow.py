"""Phase 2 tests: step-by-step DemoLab state used by the Streamlit UI."""

import pytest

from demo_data import DEMO_CUSTOMER_ID
from demo_lab import (
    FAILED,
    NOT_RUN,
    PASSED,
    STEP_BASIC_CHECKS,
    STEP_DONE,
    STEP_REPAIR,
    STEP_RUN_PIPELINE,
    STEP_TARGETED_CHECKS,
    SUCCEEDED,
    DemoLab,
)


def _statuses(lab):
    return lab.pipeline_status, lab.basic_status, lab.targeted_status


def test_initial_state_shows_nothing_as_passed():
    lab = DemoLab()
    assert _statuses(lab) == (NOT_RUN, NOT_RUN, NOT_RUN)
    assert lab.detected_mismatches == ()
    assert lab.last_validation is None
    assert lab.next_step() == STEP_RUN_PIPELINE


def test_presenter_sequence_a_to_g():
    lab = DemoLab()
    lab.reset()                                     # A

    lab.run_pipeline()                              # B, C
    assert _statuses(lab) == (SUCCEEDED, NOT_RUN, NOT_RUN)
    assert lab.next_step() == STEP_BASIC_CHECKS

    lab.run_basic_checks()                          # D
    assert _statuses(lab) == (SUCCEEDED, PASSED, NOT_RUN)
    assert lab.next_step() == STEP_TARGETED_CHECKS

    lab.run_targeted_checks()                       # E
    assert _statuses(lab) == (SUCCEEDED, PASSED, FAILED)
    assert [(m.customer_id, m.field, m.expected, m.actual) for m in lab.detected_mismatches] == [
        (DEMO_CUSTOMER_ID, "subscription_status", "active", "expired"),
        (DEMO_CUSTOMER_ID, "subscription_status", "active", "expired"),
    ]
    assert lab.next_step() == STEP_REPAIR

    lab.repair_target()                             # F
    assert _statuses(lab) == (SUCCEEDED, NOT_RUN, NOT_RUN)
    assert lab.detected_mismatches == ()
    assert lab.next_step() == STEP_BASIC_CHECKS

    lab.run_basic_checks()                          # G
    lab.run_targeted_checks()
    assert _statuses(lab) == (SUCCEEDED, PASSED, PASSED)
    assert lab.next_step() == STEP_DONE


def test_checks_before_pipeline_are_rejected_and_leave_state_unchanged():
    lab = DemoLab()
    with pytest.raises(ValueError, match="Run the pipeline first"):
        lab.run_basic_checks()
    with pytest.raises(ValueError, match="Run the pipeline first"):
        lab.run_targeted_checks()
    assert _statuses(lab) == (NOT_RUN, NOT_RUN, NOT_RUN)


def test_targeted_checks_require_basic_checks_first():
    lab = DemoLab()
    lab.run_pipeline()
    with pytest.raises(ValueError, match="basic technical checks first"):
        lab.run_targeted_checks()
    assert lab.targeted_status == NOT_RUN


def test_rerunning_pipeline_clears_old_results():
    lab = DemoLab()
    lab.run_pipeline()
    lab.run_quality_checks()
    assert lab.targeted_status == FAILED

    lab.run_pipeline()
    assert (lab.basic_status, lab.targeted_status) == (NOT_RUN, NOT_RUN)
    assert lab.detected_mismatches == ()


def test_reset_after_failed_validation_clears_everything():
    lab = DemoLab()
    lab.run_pipeline()
    lab.run_quality_checks()
    lab.reset()
    assert _statuses(lab) == (NOT_RUN, NOT_RUN, NOT_RUN)
    assert lab.detected_mismatches == ()
    assert lab.target is None
    assert lab.defect_enabled


def test_repair_disables_defect_so_rerun_stays_clean():
    lab = DemoLab()
    lab.run_pipeline()
    lab.repair_target()
    assert not lab.defect_enabled
    lab.run_pipeline()
    lab.run_quality_checks()
    assert lab.targeted_status == PASSED


def test_set_defect_logs_only_real_changes():
    lab = DemoLab()
    lab.set_defect(True)  # already on
    assert len(lab.events) == 1
    lab.set_defect(False)
    assert len(lab.events) == 2
    assert "DISABLED" in lab.events[-1]


def test_event_log_follows_sequence():
    lab = DemoLab()
    lab.run_pipeline()
    lab.run_basic_checks()
    lab.run_targeted_checks()
    assert [e.split(". ", 1)[1] for e in lab.events] == [
        "Demo reset: synthetic source loaded, target empty",
        "Pipeline run (with deliberate defect): execution SUCCEEDED (8 rows written)",
        "Basic technical checks: PASSED",
        "Business data validation: FAILED (customer 1042)",
    ]
