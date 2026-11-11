"""Synthetic source data for the AI Data Quality Lab demo.

All customers, names, and dates are fictional. The scenario is a synthetic,
reproducible example, not a real production incident.
"""

from datetime import date

import pandas as pd

# The customer whose status the deliberate defect corrupts.
DEMO_CUSTOMER_ID = 1042

# Fixed "as of" date for business rules. Using a constant instead of today's
# date keeps the demo deterministic no matter when it is presented.
AS_OF_DATE = date(2026, 10, 15)

SOURCE_COLUMNS = ["customer_id", "customer_name", "plan", "status", "expiry_date"]

# Rule baked into this data: status is "active" when expiry_date >= AS_OF_DATE,
# otherwise "expired".
_SOURCE_ROWS = [
    (1038, "Nino Beridze", "Basic", "expired", "2026-03-31"),
    (1039, "Giorgi Giorgiadze", "Premium", "active", "2027-01-31"),
    (1040, "Mariam Maisuradze", "Standard", "active", "2026-11-30"),
    (1041, "Luka Lomidze", "Basic", "expired", "2026-08-31"),
    (1042, "Ana Kapanadze", "Premium", "active", "2026-12-31"),
    (1043, "Davit Davitadze", "Standard", "active", "2027-03-31"),
    (1044, "Tamar Tamarashvili", "Premium", "expired", "2026-09-30"),
    (1045, "Irakli Kapanadze", "Basic", "active", "2026-10-31"),
]


def load_source() -> pd.DataFrame:
    """Return a fresh copy of the synthetic source dataset."""
    return pd.DataFrame(_SOURCE_ROWS, columns=SOURCE_COLUMNS)
