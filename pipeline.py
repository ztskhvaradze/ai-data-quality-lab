"""Transformation pipeline: source records -> target records for the AI assistant.

The pipeline reports only *execution* status (did it run without a technical
error?). Whether the output data is correct is a separate question answered by
data_quality.py.
"""

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMNS = ["customer_id", "customer_name", "plan", "subscription_status", "expiry_date"]

# Maps source status values to target status values.
STATUS_MAPPING = {
    "active": "active",
    "expired": "expired",
}

# DELIBERATE DEMO DEFECT. Story: a stale override left over from a legacy
# billing migration forces this customer's status to "expired". It raises no
# error, keeps the schema valid, and changes no row counts.
DEFECT_STATUS_OVERRIDES = {1042: "expired"}

SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"


@dataclass(frozen=True)
class PipelineRun:
    """Outcome of one pipeline execution. Says nothing about data correctness."""

    status: str
    rows_read: int
    rows_written: int
    defect_injected: bool
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCEEDED


def transform(source: pd.DataFrame, inject_defect: bool = False) -> pd.DataFrame:
    """Build the target dataset from the source dataset.

    Steps: copy, normalise status text, map status, apply the optional
    deliberate defect, parse dates, and rename/reorder columns.
    """
    target = source.copy()

    normalised_status = target["status"].str.strip().str.lower()
    target["subscription_status"] = normalised_status.map(STATUS_MAPPING)

    if inject_defect:
        for customer_id, wrong_status in DEFECT_STATUS_OVERRIDES.items():
            target.loc[target["customer_id"] == customer_id, "subscription_status"] = wrong_status

    target["customer_id"] = target["customer_id"].astype("int64")
    target["expiry_date"] = pd.to_datetime(target["expiry_date"], format="%Y-%m-%d")

    return target[TARGET_COLUMNS].reset_index(drop=True)


def run_pipeline(source: pd.DataFrame, inject_defect: bool = False) -> tuple[PipelineRun, pd.DataFrame | None]:
    """Run the transformation and report execution status.

    Returns (run, target). target is None if the pipeline failed technically.
    """
    try:
        target = transform(source, inject_defect=inject_defect)
    except Exception as exc:  # any technical failure counts as a failed run
        run = PipelineRun(
            status=FAILED,
            rows_read=len(source),
            rows_written=0,
            defect_injected=inject_defect,
            error=f"{type(exc).__name__}: {exc}",
        )
        return run, None

    run = PipelineRun(
        status=SUCCEEDED,
        rows_read=len(source),
        rows_written=len(target),
        defect_injected=inject_defect,
    )
    return run, target


def rebuild_target(source: pd.DataFrame) -> tuple[PipelineRun, pd.DataFrame | None]:
    """Repair: regenerate the target from source with the defect removed."""
    return run_pipeline(source, inject_defect=False)
