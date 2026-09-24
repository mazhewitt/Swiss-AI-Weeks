"""The one place that defines the Cutoff, the Horizon and the allowed labels."""

import pandas as pd

CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
HORIZON_DAYS = 90
HORIZON = pd.Timedelta(days=HORIZON_DAYS)

MERCHANT_FAMILIES = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming")
NONE_LABEL = "none"
LABELS = (*MERCHANT_FAMILIES, NONE_LABEL)

LABEL_COLUMN = "target_next_recurring_merchant"
PREDICTION_COLUMN = "predicted_next_recurring_merchant"
