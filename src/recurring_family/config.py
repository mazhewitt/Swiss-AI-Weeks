"""The one place that defines the Cutoff, the Horizon and the allowed labels."""

import pandas as pd

CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
HORIZON_DAYS = 90
HORIZON = pd.Timedelta(days=HORIZON_DAYS)

# Every history ends on the last day before the Cutoff (2025-12-31), so a Shifted Cutoff's Horizon
# must end by then. The default Shifted Cutoff is the latest one whose Horizon is fully observed.
HISTORY_END = CUTOFF
SHIFTED_CUTOFF = HISTORY_END - HORIZON  # 2025-10-03

MERCHANT_FAMILIES = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming")
NONE_LABEL = "none"
LABELS = (*MERCHANT_FAMILIES, NONE_LABEL)

LABEL_COLUMN = "target_next_recurring_merchant"
PREDICTION_COLUMN = "predicted_next_recurring_merchant"
