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

**Status:** done

- [x] `--none-model` works in `train`, `cv`, `evaluate` and `submit` (and `--decision tuned`). Save then load gives identical probabilities. Rows sum to 1. The log's model column names it (e.g. `ranker+none+pseudo+tuned`)
- [x] Without `--none-model`, every existing output is byte-identical (ranker, blend, rules, the Day-2 noon upload script)
- [x] Pseudo-Labelled Clients never reach the `none` model's fit (a test proves it). No label of a scored Client reaches its own features
- [x] Every new feature is computed from stream member payments, matched refunds or the stream table only (a test: injecting Decoy Transactions leaves the `none` model's features unchanged)
- [x] Train out-of-fold comparison with the Pseudo-Label settings of run `20260924T171108-3251a4` (train+unlabeled at 2025-10-03, weight 0.5, min_payments 4): argmax, tuned and nested tuned macro-F1 for the baseline, group A only, and A+B. Recorded in the ticket
- [x] Only if the best variant beats the baseline's nested tuned macro-F1 by at least 0.01 on train: one run is logged on the selection set (`--decision tuned`, paired bootstrap against `20260924T171108-3251a4`), and the predictions are committed
- [x] The full fast test suite passes

## Outcome

- `train` and `cv --model ranker --none-model` fit the `none` model (`src/recurring_family/none_model.py`) beside the ranker; `evaluate` and `submit` load it with the saved model. The features: group A (29: the soonest-due Active Stream's value, min and max of the candidate fields, plus stream counts), group B (75: churn signals from member payments and matched refunds, `n_ended`, `past_churn_rate`, `n_ended_recent120`, activity ratios, `active_has_<family>` and so on) and `ranker_none`, cross-fitted in 5 folds over the training Clients. `detect_stream_payments` exposes the member payments and which of them a matched refund reverses; the stream table is unchanged (identical on train, valid, test and unlabeled), and `bash scripts/day2_noon_ranker_pseudo.sh` reproduces `submissions/day2_noon_ranker_pseudo.csv` byte for byte.
- The features match the analysis's `features.py` on train except where intended: the soonest-due stream's value is that stream's own (the analysis's `groupby().first()` skipped missing values), and the refund features use matched refunds, not raw refunds within 7 days.
- Train out-of-fold (`scripts/none_model_variants.py`, `experiments/none_model_variants.csv`), Pseudo-Label settings of `20260924T171108-3251a4`, 2,000 Clients; nested = decision fitted on 4 folds, applied to the 5th:

| Variant | argmax | tuned | nested tuned | nested gain |
|---|---|---|---|---|
| Baseline ranker with Pseudo-Labels | 0.5485 | 0.5819 | 0.5699 | |
| A | 0.5925 | 0.5980 | 0.5761 | +0.006 |
| A + `ranker_none` | 0.6053 | 0.6106 | 0.5958 | +0.026 |
| A+B | 0.6137 | 0.6206 | 0.6033 | +0.033 |
| **A+B + `ranker_none`** (chosen) | 0.6154 | 0.6192 | **0.6095** | **+0.040** |

  Fitting and applying the model to Active-Stream Clients only is worse for every feature set (nested 0.592 to 0.597). No point of a small grid (15 leaves, min_child 60, 400 trees, lr 0.05 with 300 trees) beat the analysis's hyperparameters (best 0.6079). `rf cv --none-model` gives exactly the script's out-of-fold rows for the chosen setup.
- The train gate (+0.01) was met, so one selection run was logged: `20260924T225229-cc7243` (`ranker+none+pseudo+tuned`), macro-F1 0.5764 against 0.5735 for `20260924T171108-3251a4`: delta +0.0029 (95% -0.0221 .. +0.0267), a tie. `none` F1 rises 0.6233 -> 0.6616, but music (0.4793 -> 0.4511) and software (0.5037 -> 0.4769) fall. The +0.040 on train does not carry over. One possible reason, not checked: the analysis that motivated the ticket and the feature set were both made on the same train Clients.
- Candidate only: `scripts/day2_final_ranker_none.sh` refits on train plus the selection set and writes `submissions/day2_final_ranker_none.csv` (valid; 85% agreement with the Day-2 12:00 upload; none 286, insurance 121, gym 107, mobile 107, cloud 106, streaming 96, music 92, software 85). Humans decide whether to upload it.
