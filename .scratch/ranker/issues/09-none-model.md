# 09: A Client-level `none` model for the Stream Ranker

**What to build:** Give the Stream Ranker its own model for `none`. Today `none` is 1 − the best Candidate Stream's score. That construction is weak: among train Clients with an Active Stream, its AUC for "truth is `none`" is 0.69. A Client-level LightGBM on per-Client aggregates of the ranker's own stream fields reaches 0.84.

The analysis is in `experiments/analysis/churn/` (scripts plus CSVs; `stack2_results.csv` is the key table). On train out-of-fold, with the decision layer fitted on 4 folds and scored on the 5th ("nested"):

| Variant | nested tuned macro-F1 |
|---|---|
| Baseline ranker with Pseudo-Labels | 0.5699 |
| A: Client-level `none` model on the ranker's own fields (min, max and the soonest stream's value) | 0.5905 (+0.021) |
| B: A plus churn features from stream member payments | 0.6045 (+0.035) |

Why: at the real Cutoff about 23% of Clients with a live stream are `none`, while inside the history only 3–5% of live streams stop within 90 days (ticket 08 found the same). So `none` depends on the Client, not on one stream's timing.

The design:

- **When the flag is on:** `train/cv --model ranker --none-model` fits, next to the ranker, a LightGBM binary model on one row per real-labelled Client that has at least one Candidate Stream. Its target is "label is `none`".
- **Combining:** P(`none`) comes from the `none` model. Each family gets its share of the ranker's family scores (its best candidate's score over the sum of the families' best scores), times 1 − P(`none`). A Client without candidates stays `none`.
- **Training rows:** Pseudo-Labelled Clients never train the `none` model (ticket 08: a Shifted Cutoff has almost no churn). They still train the ranker as before.
- **The ranker's own score as a feature:** the ranker's scores on its own training Clients are in-sample. If the `none` model uses the ranker's score (the analysis used "1 − max" as a feature), get it by cross-fitting inside `fit` (k-fold over the training Clients), never in-sample. Measure both ways (with and without the feature) on train and keep the better one.

Features: only from the stream table and from the member payments and matched refunds of detected streams (ADR 0001). Never raw-transaction counts. Group A is per-Client prim/min/max of the candidate fields. Group B is from the analysis's `ROBUST` list (`experiments/analysis/churn/stack2.py`, definitions in `features.py`): overdue, last gap ratio, missed payments, gap trend, refunds near the end, `n_ended`, `past_churn_rate`, `n_ended_recent120`, `n_families_live`, `total_payments`, `stream_first_days`, and similar. Features that need member payments may need `streams._summarise` or `_match_refunds` to return more fields. If so, bump nothing by hand: the stream cache is keyed by detector source. Don't use the `FILLER` features (description share, last-amount changes): they shift between train and valid. Don't use the raw "hidden series" features either: those are ticket 10, parked until a drift check.

**Blocked by:** 03, 05

**Status:** ready-for-agent

- [ ] `--none-model` works in `train`, `cv`, `evaluate` and `submit` (and `--decision tuned`). Save then load gives identical probabilities. Rows sum to 1. The log's model column names it (e.g. `ranker+none+pseudo+tuned`)
- [ ] Without `--none-model`, every existing output is byte-identical (ranker, blend, rules, the Day-2 noon upload script)
- [ ] Pseudo-Labelled Clients never reach the `none` model's fit (a test proves it). No label of a scored Client reaches its own features
- [ ] Every new feature is computed from stream member payments, matched refunds or the stream table only (a test: injecting Decoy Transactions leaves the `none` model's features unchanged)
- [ ] Train out-of-fold comparison with the Pseudo-Label settings of run `20260924T171108-3251a4` (train+unlabeled at 2025-10-03, weight 0.5, min_payments 4): argmax, tuned and nested tuned macro-F1 for the baseline, group A only, and A+B. Recorded in the ticket
- [ ] Only if the best variant beats the baseline's nested tuned macro-F1 by at least 0.01 on train: one run is logged on the selection set (`--decision tuned`, paired bootstrap against `20260924T171108-3251a4`), and the predictions are committed
- [ ] The full fast test suite passes
