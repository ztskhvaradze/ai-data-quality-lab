"""Data quality checks.

Two layers:
- Basic technical checks: schema, non-null IDs, unique IDs, row counts.
  These pass on the defective target, which is the point of the demo.
- Targeted business checks: source-to-target status reconciliation and an
  expiry-date business rule. These catch the injected defect.

The validation status reported here is independent of pipeline execution status.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from pandas.api import types as ptypes

from demo_data import AS_OF_DATE
from pipeline import STATUS_MAPPING, TARGET_COLUMNS

PASSED = "PASSED"
FAILED = "FAILED"

BASIC = "basic"
TARGETED = "targeted"


@dataclass(frozen=True)
class Mismatch:
    customer_id: int
    field: str
    expected: str
    actual: str
    reason: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    passed: bool
    details: str
    mismatches: tuple[Mismatch, ...] = ()

    @property
    def status(self) -> str:
        return PASSED if self.passed else FAILED


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    def _by_category(self, category: str) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.category == category)

    @property
    def basic_checks(self) -> tuple[CheckResult, ...]:
        return self._by_category(BASIC)

    @property
    def targeted_checks(self) -> tuple[CheckResult, ...]:
        return self._by_category(TARGETED)

    @property
    def basic_passed(self) -> bool:
        return all(c.passed for c in self.basic_checks)

    @property
    def targeted_passed(self) -> bool:
        return all(c.passed for c in self.targeted_checks)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def status(self) -> str:
        return PASSED if self.passed else FAILED

    @property
    def mismatches(self) -> tuple[Mismatch, ...]:
        return tuple(m for c in self.checks for m in c.mismatches)


# --- Basic technical checks -------------------------------------------------

def check_schema(target: pd.DataFrame) -> CheckResult:
    name = "Schema matches expected columns and types"
    missing = [c for c in TARGET_COLUMNS if c not in target.columns]
    extra = [c for c in target.columns if c not in TARGET_COLUMNS]
    if missing or extra:
        return CheckResult(name, BASIC, False, f"Missing columns: {missing}; unexpected columns: {extra}")

    type_problems = []
    if not ptypes.is_integer_dtype(target["customer_id"]):
        type_problems.append("customer_id is not an integer")
    for col in ("customer_name", "plan", "subscription_status"):
        if not (ptypes.is_string_dtype(target[col]) or ptypes.is_object_dtype(target[col])):
            type_problems.append(f"{col} is not text")
    if not ptypes.is_datetime64_any_dtype(target["expiry_date"]):
        type_problems.append("expiry_date is not a date")

    if type_problems:
        return CheckResult(name, BASIC, False, "; ".join(type_problems))
    return CheckResult(name, BASIC, True, f"All {len(TARGET_COLUMNS)} columns present with expected types")


def check_customer_id_not_null(target: pd.DataFrame) -> CheckResult:
    nulls = int(target["customer_id"].isna().sum())
    return CheckResult(
        "Customer IDs are not null",
        BASIC,
        nulls == 0,
        "No null customer IDs" if nulls == 0 else f"{nulls} null customer ID(s)",
    )


def check_customer_id_unique(target: pd.DataFrame) -> CheckResult:
    dupes = int(target["customer_id"].duplicated().sum())
    return CheckResult(
        "Customer IDs are unique",
        BASIC,
        dupes == 0,
        "No duplicate customer IDs" if dupes == 0 else f"{dupes} duplicate customer ID(s)",
    )


def check_row_count(source: pd.DataFrame, target: pd.DataFrame) -> CheckResult:
    ok = len(source) == len(target)
    return CheckResult(
        "Row count matches source",
        BASIC,
        ok,
        f"Source rows: {len(source)}, target rows: {len(target)}",
    )


# --- Targeted business checks -----------------------------------------------

def check_status_matches_source(source: pd.DataFrame, target: pd.DataFrame) -> CheckResult:
    """Source-to-target reconciliation of subscription status per customer."""
    name = "Subscription status matches source"
    expected = source[["customer_id", "status"]].copy()
    expected["expected_status"] = expected["status"].str.strip().str.lower().map(STATUS_MAPPING)

    joined = expected.merge(
        target[["customer_id", "subscription_status"]], on="customer_id", how="left"
    )
    bad = joined[joined["expected_status"] != joined["subscription_status"]]

    mismatches = tuple(
        Mismatch(
            customer_id=int(row.customer_id),
            field="subscription_status",
            expected=str(row.expected_status),
            actual=str(row.subscription_status),
            reason=f"Source status is '{row.status}'",
        )
        for row in bad.itertuples(index=False)
    )
    details = (
        f"All {len(joined)} customers match source"
        if not mismatches
        else f"{len(mismatches)} of {len(joined)} customers differ from source"
    )
    return CheckResult(name, TARGETED, not mismatches, details, mismatches)


def check_status_consistent_with_expiry(target: pd.DataFrame, as_of: date = AS_OF_DATE) -> CheckResult:
    """Business rule: active if expiry_date >= as_of, otherwise expired."""
    name = f"Status consistent with expiry date (as of {as_of.isoformat()})"
    as_of_ts = pd.Timestamp(as_of)
    expected_status = target["expiry_date"].ge(as_of_ts).map({True: "active", False: "expired"})
    bad = target[expected_status != target["subscription_status"]]

    mismatches = tuple(
        Mismatch(
            customer_id=int(row.customer_id),
            field="subscription_status",
            expected=str(expected_status.loc[idx]),
            actual=str(row.subscription_status),
            reason=f"Expiry date is {row.expiry_date.date().isoformat()}",
        )
        for idx, row in bad.iterrows()
    )
    details = (
        f"All {len(target)} customers follow the rule"
        if not mismatches
        else f"{len(mismatches)} of {len(target)} customers violate the rule"
    )
    return CheckResult(name, TARGETED, not mismatches, details, mismatches)


# --- Runners ----------------------------------------------------------------

def run_basic_checks(source: pd.DataFrame, target: pd.DataFrame) -> tuple[CheckResult, ...]:
    schema = check_schema(target)
    if not schema.passed:
        # Later checks depend on the expected columns existing.
        return (schema,)
    return (
        schema,
        check_customer_id_not_null(target),
        check_customer_id_unique(target),
        check_row_count(source, target),
    )


def run_targeted_checks(source: pd.DataFrame, target: pd.DataFrame) -> tuple[CheckResult, ...]:
    return (
        check_status_matches_source(source, target),
        check_status_consistent_with_expiry(target),
    )


def validate(source: pd.DataFrame, target: pd.DataFrame) -> ValidationReport:
    basic = run_basic_checks(source, target)
    if not all(c.passed for c in basic):
        return ValidationReport(basic)
    return ValidationReport(basic + run_targeted_checks(source, target))
