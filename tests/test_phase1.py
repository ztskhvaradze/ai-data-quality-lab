"""Phase 1 tests: deterministic pipeline, defect, quality checks, repair, reset."""

from datetime import date

import pandas as pd
import pytest

from data_quality import (
    check_status_consistent_with_expiry,
    run_basic_checks,
    validate,
)
from demo_data import AS_OF_DATE, DEMO_CUSTOMER_ID, load_source
from demo_lab import DemoLab
from pipeline import FAILED, SUCCEEDED, TARGET_COLUMNS, rebuild_target, run_pipeline


# --- Source data ------------------------------------------------------------

def test_source_record_for_demo_customer():
    source = load_source()
    row = source[source["customer_id"] == DEMO_CUSTOMER_ID].iloc[0]
    assert row["status"] == "active"
    assert row["expiry_date"] == "2026-12-31"


def test_source_statuses_follow_expiry_rule():
    source = load_source()
    for row in source.itertuples(index=False):
        expected = "active" if date.fromisoformat(row.expiry_date) >= AS_OF_DATE else "expired"
        assert row.status == expected, row.customer_id


def test_load_source_returns_independent_copies():
    first = load_source()
    first.loc[0, "status"] = "tampered"
    assert load_source().loc[0, "status"] != "tampered"


# --- Pipeline execution -----------------------------------------------------

@pytest.mark.parametrize("inject_defect", [False, True])
def test_pipeline_succeeds_with_and_without_defect(inject_defect):
    source = load_source()
    run, target = run_pipeline(source, inject_defect=inject_defect)
    assert run.status == SUCCEEDED
    assert run.error is None
    assert run.rows_read == run.rows_written == len(source)
    assert list(target.columns) == TARGET_COLUMNS


def test_pipeline_does_not_modify_source():
    source = load_source()
    run_pipeline(source, inject_defect=True)
    pd.testing.assert_frame_equal(source, load_source())


def test_pipeline_reports_failure_on_technical_error():
    broken = load_source().drop(columns=["status"])
    run, target = run_pipeline(broken)
    assert run.status == FAILED
    assert "status" in run.error
    assert target is None


# --- Injected defect --------------------------------------------------------

def _status(target, customer_id):
    return target.loc[target["customer_id"] == customer_id, "subscription_status"].iloc[0]


def test_defect_changes_only_customer_1042():
    source = load_source()
    _, clean = run_pipeline(source, inject_defect=False)
    _, defective = run_pipeline(source, inject_defect=True)

    assert _status(clean, DEMO_CUSTOMER_ID) == "active"
    assert _status(defective, DEMO_CUSTOMER_ID) == "expired"

    others_clean = clean[clean["customer_id"] != DEMO_CUSTOMER_ID]
    others_defective = defective[defective["customer_id"] != DEMO_CUSTOMER_ID]
    pd.testing.assert_frame_equal(others_clean, others_defective)


def test_defect_is_deterministic():
    source = load_source()
    _, first = run_pipeline(source, inject_defect=True)
    _, second = run_pipeline(source, inject_defect=True)
    pd.testing.assert_frame_equal(first, second)


# --- Basic checks pass despite the defect -----------------------------------

def test_basic_checks_pass_on_defective_target():
    source = load_source()
    _, defective = run_pipeline(source, inject_defect=True)
    results = run_basic_checks(source, defective)
    assert len(results) == 4
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_basic_checks_catch_null_id_and_row_loss():
    source = load_source()
    _, target = run_pipeline(source)
    broken = target.iloc[1:].astype({"customer_id": "Int64"}).reset_index(drop=True)
    broken.loc[0, "customer_id"] = pd.NA
    by_name = {r.name: r for r in run_basic_checks(source, broken)}
    assert not by_name["Customer IDs are not null"].passed
    assert not by_name["Row count matches source"].passed


# --- Targeted validation catches the defect ---------------------------------

def test_targeted_validation_fails_on_defect_with_expected_and_actual():
    source = load_source()
    run, defective = run_pipeline(source, inject_defect=True)
    report = validate(source, defective)

    # Pipeline status and validation status are independent.
    assert run.status == SUCCEEDED
    assert report.basic_passed
    assert not report.targeted_passed
    assert report.status == "FAILED"

    reconciliation = report.targeted_checks[0]
    assert not reconciliation.passed
    assert len(reconciliation.mismatches) == 1
    mismatch = reconciliation.mismatches[0]
    assert mismatch.customer_id == DEMO_CUSTOMER_ID
    assert mismatch.field == "subscription_status"
    assert mismatch.expected == "active"
    assert mismatch.actual == "expired"


def test_expiry_rule_check_also_flags_defect():
    _, defective = run_pipeline(load_source(), inject_defect=True)
    result = check_status_consistent_with_expiry(defective)
    assert not result.passed
    assert [(m.customer_id, m.expected, m.actual) for m in result.mismatches] == [
        (DEMO_CUSTOMER_ID, "active", "expired")
    ]
    assert "2026-12-31" in result.mismatches[0].reason


def test_all_checks_pass_on_clean_target():
    source = load_source()
    _, clean = run_pipeline(source, inject_defect=False)
    report = validate(source, clean)
    assert report.passed
    assert report.mismatches == ()


# --- Repair -----------------------------------------------------------------

def test_rebuild_target_produces_clean_data():
    source = load_source()
    run, repaired = rebuild_target(source)
    assert run.succeeded
    assert not run.defect_injected
    assert _status(repaired, DEMO_CUSTOMER_ID) == "active"
    assert validate(source, repaired).passed


# --- DemoLab end-to-end and reset -------------------------------------------

def _run_full_scenario(lab: DemoLab) -> None:
    run = lab.run_pipeline()
    assert run.status == SUCCEEDED
    assert lab.get_target_record(DEMO_CUSTOMER_ID)["subscription_status"] == "expired"

    report = lab.run_quality_checks()
    assert report.basic_passed
    assert not report.targeted_passed
    assert {(m.customer_id, m.expected, m.actual) for m in report.mismatches} == {
        (DEMO_CUSTOMER_ID, "active", "expired")
    }

    run = lab.repair_target()
    assert run.status == SUCCEEDED
    assert lab.last_validation is None  # stale result cleared after target changed
    assert lab.get_target_record(DEMO_CUSTOMER_ID)["subscription_status"] == "active"
    assert lab.run_quality_checks().passed


def test_demo_lab_full_scenario():
    lab = DemoLab()
    assert lab.defect_enabled
    assert lab.target is None
    _run_full_scenario(lab)


def test_reset_restores_initial_state_and_scenario_repeats():
    lab = DemoLab()
    for _ in range(3):
        _run_full_scenario(lab)
        lab.reset()
        assert lab.defect_enabled
        assert lab.target is None
        assert lab.last_run is None
        assert lab.last_validation is None
        assert len(lab.events) == 1
        pd.testing.assert_frame_equal(lab.source, load_source())


def test_toggle_defect_off_then_on():
    lab = DemoLab()
    lab.set_defect(False)
    lab.run_pipeline()
    assert lab.run_quality_checks().passed
    lab.set_defect(True)
    lab.run_pipeline()
    assert not lab.run_quality_checks().targeted_passed


def test_quality_checks_before_pipeline_raise_clear_error():
    with pytest.raises(ValueError, match="Run the pipeline first"):
        DemoLab().run_quality_checks()


def test_get_target_record_is_plain_json_friendly_dict():
    lab = DemoLab()
    lab.run_pipeline()
    assert lab.get_target_record(DEMO_CUSTOMER_ID) == {
        "customer_id": 1042,
        "customer_name": "Ana Gelashvili",
        "plan": "Premium",
        "subscription_status": "expired",
        "expiry_date": "2026-12-31",
    }
    assert lab.get_target_record(9999) is None
