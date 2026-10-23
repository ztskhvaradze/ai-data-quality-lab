"""Terminal walkthrough of the Phase 1 scenario (no UI, no LLM)."""

from data_quality import ValidationReport
from demo_data import DEMO_CUSTOMER_ID
from demo_lab import DemoLab


def print_report(report: ValidationReport) -> None:
    for check in report.checks:
        print(f"  [{check.status}] ({check.category}) {check.name}: {check.details}")
        for m in check.mismatches:
            print(
                f"      customer {m.customer_id} {m.field}: "
                f"expected '{m.expected}', actual '{m.actual}' ({m.reason})"
            )


def main() -> None:
    print("AI Data Quality Lab - synthetic, reproducible example\n")
    lab = DemoLab()

    print("Step 1: run pipeline with deliberate defect")
    run = lab.run_pipeline()
    print(f"  Pipeline execution: {run.status}")
    print(f"  Target record {DEMO_CUSTOMER_ID}: {lab.get_target_record(DEMO_CUSTOMER_ID)}\n")

    print("Step 2: run quality checks")
    report = lab.run_quality_checks()
    print(f"  Basic checks: {'PASSED' if report.basic_passed else 'FAILED'}")
    print(f"  Business data validation: {'PASSED' if report.targeted_passed else 'FAILED'}")
    print_report(report)
    print()

    print("Step 3: repair target from source and re-check")
    run = lab.repair_target()
    print(f"  Pipeline execution: {run.status}")
    print(f"  Target record {DEMO_CUSTOMER_ID}: {lab.get_target_record(DEMO_CUSTOMER_ID)}")
    report = lab.run_quality_checks()
    print(f"  Business data validation: {'PASSED' if report.targeted_passed else 'FAILED'}")
    print_report(report)
    print()

    print("Event log:")
    for event in lab.events:
        print(f"  {event}")


if __name__ == "__main__":
    main()
