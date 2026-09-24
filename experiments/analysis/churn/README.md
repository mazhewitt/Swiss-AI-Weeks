# Churn analysis: why truth-`none` Clients with a live stream get a family (train only)

The scripts here reproduce every figure. They run in order `prep.py` → `features.py` → `hidden.py` → the rest, and write their pickles beside themselves; `prep.py` expects `oof_ranker.csv`, the pseudo ranker's `rf cv` output. They read train labels only, never valid.

- Group: 1,685 train Clients with at least one Active Stream, 23.3% of them truth-`none`.
- The ranker's none AUC (1 − max) is 0.69 (`stack_results.csv`).
- Classic churn signals are near chance: overdue ratio, a missed or late last payment, day of month, family, a final refund (`univariate.csv`).
- A Client-level none model on the ranker's own fields reaches AUC 0.84 (+0.021 nested tuned macro-F1). Adding churn features from stream member payments reaches 0.88 (+0.035); see `stack2_results.csv`. This became ticket 09.
- Inside the history, 3–5% of live streams stop within 90 days (8–12% for Sep–Oct Shifted Cutoffs). At the real Cutoff, 23% of live-stream Clients are `none` (`stop_hazard.csv`). So `none` is decided at the label level, and Shifted-Cutoff Pseudo-Labels can't teach it.
- "Hidden series" are amount clusters of subscription-like card payments that join no stream. In train, 99% of Clients with one of 3+ payments are truth-`none`. They add +0.06 nested (`stack2_results.csv`, variant D). But they are raw-transaction features, which are exposed to the Decoy drift of ADR 0001. Ticket 10 is parked until a label-free drift check on valid and test.
- Ceiling: setting every live-stream truth-`none` Client to `none` gives tuned macro-F1 0.686, against 0.582 today (`f1_results.csv`).
