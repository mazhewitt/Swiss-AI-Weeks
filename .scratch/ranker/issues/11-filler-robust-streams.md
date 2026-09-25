# 11: Streams that don't break on Filler Descriptions

**What to build:** Make stream detection robust to the Filler Description rate of valid and test, so that the stream features of train Clients look like those of valid and test Clients.

Ticket 09's review found a covariate shift in the stream table. Label-free, from the package's public functions:

- A classifier telling train from valid Clients reaches AUC 0.73 on the churn features. Against test it reaches 0.80.
- `max_missed_rate` averages 0.069 in train, 0.129 in valid and 0.167 in test.
- `n_ended` falls from 0.58 to 0.30, and `n_short` rises from 1.14 to 1.48.
- The top description of a stream is a Filler Description for 1.3% of streams in train and 16.7% in valid (`experiments/analysis/churn/drift_streams.csv`).

That shift is the likely reason ticket 09's +0.040 on train became a tie on selection, and it probably costs every stream-based model some of its score.

Likely mechanism, to confirm first: `streams._evidence` drops a description with no family word unless its MCC is a family's home MCC. So a stream payment that carries a Filler Description ("member plan", "subscription charge") on a non-home MCC is dropped. The stream then shows a gap: a "missed" payment, or a stream split in two. Valid and test have many more Filler Descriptions, so their streams break more often.

Fix direction: let a Filler Description payment on any MCC (never a Decoy Transaction, a shop payment or a fee) join an existing stream when its amount fits the stream's amount cluster and its date fits the stream's schedule. It must never start a stream. Other fixes are fine if the diagnosis points elsewhere.

**Blocked by:** 09

**Status:** done

- [x] A diagnosis with figures: why stream features differ between train, valid and test. Label-free: transactions and stream tables only, never a valid or test label. Recorded in a comment on the ticket
- [x] The detector change, with unit tests at the detector seam: a stream whose payments partly carry Filler Descriptions on other MCCs is detected whole; a Filler Description payment at another amount, or off the schedule, does not join; a Decoy Transaction never joins; the Decoy-injection test of ADR 0001 still passes
- [x] Label-free shift check before and after: the train-vs-valid and train-vs-test classifier AUC on the ranker's candidate fields and on ticket 09's `none`-model features, plus the means of `max_missed_rate`, `n_short` and `n_ended`, as a table in the ticket. The change should narrow the shift
- [x] Train out-of-fold (argmax, tuned, nested tuned) for the ranker with Pseudo-Labels, with and without `--none-model`, before and after the change. Pseudo-Label settings as in run `20260924T171108-3251a4`
- [x] If the shift narrows and train does not get worse by more than 0.01: one selection run each for `ranker+pseudo` and `ranker+none+pseudo` (tuned), each with a paired bootstrap against the same model before the change (`20260924T171108-3251a4` and `20260924T225229-cc7243`). Predictions committed
- [x] The Day-2 noon upload stays reproducible from its script at the commit it was made from. The script may now produce a different file on the new detector. That's expected, so say so in the ticket
- [x] The full fast test suite passes

## Comments

### Diagnosis (label-free: transactions and stream tables only)

Scripts: `experiments/analysis/filler_streams/` (`gaps.py`, `kindlib.py`, `shift11.py`). For each detected stream of 2+ payments, `gaps.py` looks for payments left out of every stream that fit the stream's amount range (± the amount tolerance) and its schedule within 4 days: in a gap where a payment is missing, or one or two periods beyond its ends. It groups them by `_evidence` kind. A control counts the same payments half a period off the schedule.

- **The gaps are real, and most are Filler Descriptions booked on another family's home MCC.** Missed payments per stream: 0.36 in train, 0.78 in valid, 1.01 in test. Payments that fill a gap, per 100 streams:

  | kind | train | valid | test | valid control (off-phase) |
  |---|---|---|---|---|
  | Filler Description on a home MCC (of another family) | 2.10 | 12.54 | 23.74 | 0.28 |
  | Filler Description dropped, no home MCC (5411 only) | 0.37 | 3.10 | 4.83 | 0.00 |
  | ambiguous, no family of it fits (`monthly plan` 90%) | 0.51 | 2.63 | 6.62 | 0.19 |
  | family word on the wrong MCC | 0.81 | 1.93 | 1.05 | 0.05 |
  | Decoy Transaction (excluded by ADR 0001) | 0.88 | 8.69 | 7.27 | 1.03 |
  | shop or fee | 0.58 | 0.56 | 0.75 | 0.66 |

  The Filler Descriptions sit on the schedule: of those on a home MCC that fit a stream's amount (within 1.5 periods of its ends, at least half a period from its payments), 83% are within 2 days of an empty slot. Their MCC is a fixed alternative MCC of the stream's family: mobile on 5734, gym on 5812, software on 5812 and 5732, cloud on 4814 and 5734, music and streaming on 5734. A family word on such an MCC ("phone contract" on 5734) already joins its family's stream, and train's stream payments carry family words far more often (below). A Filler Description there takes the MCC's family (software) and finds no stream to join. The detector's "drop" of an off-home Filler Description (the ticket's first guess) explains only about a sixth of the gap. Streams one period beyond their last payment show the same pattern (filler on a home MCC: 1.20 train, 2.21 valid, 4.33 test), so ended streams look more ended.
- **Streams split in two are rare.** Same family and amount, the second starting on schedule where the first ends: 0.06, 0.05 and 0.10 per 100 streams. Not a cause.
- **What the rate of Filler Descriptions looks like per split.** Per Client, card payments carrying a family word on a home MCC: 12.5 train, 6.8 valid, 5.0 test. Filler Descriptions on a home MCC: 1.9, 7.4, 9.3. In train the Filler Descriptions mark `none` Clients (train labels only): 2.75 per `none` Client against 0.57 for the others. After the fix, 92.5% of the train Clients who get a scheduled join are truth-`none` (0.41 joins per `none` Client, 0.009 per other), whereas 38% of valid and 52% of test Clients get one. So the gaps that the `none` model learned as churn in train are a Filler Description artefact that valid and test have everywhere. Filling them removes a train-only `none` signal: the train score could not rise.
- **Not explained by Filler Descriptions:** the lower `n_ended` of valid and test (0.30 against 0.58), and 3.3 family words per Client on a foreign MCC in valid and test that join no stream (0.8 in train; 60% of them in Clients without a stream of that family). History lengths are equal across splits (median 413–414 days). Decoy-described payments on a stream's schedule (8.7 per 100 streams in valid, control 1.0) stay excluded by ADR 0001.

### The change

`streams._client_streams`: a *stray* payment (a description that names no family: a Filler Description on another family's home MCC or on no home MCC, or an ambiguous description no family of it fits) that no stream took joins a stream of at least two payments. It must fit the stream's amount range (the amount tolerance) and an empty slot of its schedule: within `StreamParams.schedule_tolerance_days` (3, from the offset distribution above) of a whole number of periods from one of its payments, at least half a period from all of them, and at most one period beyond its first or last payment. Passes repeat, so a run of strays can extend a stream. It never starts a stream. A shop payment, fee or Decoy Transaction is still `drop`. A family word on the wrong MCC keeps its family. `_evidence` calls an off-home Filler Description `stray` (it was `drop`), and refunds ignore it as before.

After the fix, gap-filling Filler Descriptions per 100 streams fall to 0.42 (valid) and 0.60 (test), and missed payments per stream to 0.33 (train), 0.57 (valid) and 0.57 (test).

### Shift check, before and after (`shift11.py`; LightGBM train-vs-other classifier, 5-fold, 3 seeds; Clients with a stream)

| Measure | before | after |
|---|---|---|
| AUC train vs valid: ranker candidate fields (min/mean/max per Client, family counts) | 0.712 | 0.704 |
| AUC train vs valid: `none` model group A | 0.607 | 0.594 |
| AUC train vs valid: `none` model group B | 0.733 | 0.668 |
| AUC train vs valid: `none` model A+B | 0.756 | 0.700 |
| AUC train vs test: ranker candidate fields | 0.771 | 0.753 |
| AUC train vs test: group A | 0.678 | 0.618 |
| AUC train vs test: group B | 0.796 | 0.697 |
| AUC train vs test: A+B | 0.811 | 0.725 |
| mean `max_missed_rate` train / valid / test | 0.069 / 0.129 / 0.167 | 0.063 / 0.088 / 0.089 |
| mean `n_short` train / valid / test | 1.143 / 1.481 / 1.322 | 1.116 / 1.457 / 1.301 |
| mean `n_ended` train / valid / test | 0.583 / 0.304 / 0.297 | 0.583 / 0.290 / 0.281 |
| Active Streams per Client train / valid / test | 1.58 / 1.34 / 1.34 | 1.61 / 1.38 / 1.37 |
| stream payments per Client train / valid / test | 18.32 / 15.37 / 15.09 | 18.45 / 15.94 / 16.15 |

Every classifier AUC falls, most for the churn features (group B: −0.065 against valid, −0.099 against test), and `max_missed_rate` no longer drifts. `n_short` and `n_ended` hardly move: their train-to-valid and train-to-test gaps widen by at most 0.02. The top remaining drivers are `stream_first_days`, `n_ended` and `max_amount_cv`, which Filler Descriptions don't explain.

## Outcome

- **Cause:** stream payments booked with a Filler Description on another family's home MCC (or on no home MCC), which valid and test have 4–5 times as often as train, fell out of their streams. That caused the extra missed payments, and it made ended streams look more ended. The detector now lets such a stray payment join a stream whose amount and schedule it fits (`StreamParams.schedule_tolerance_days` = 3). It never starts a stream, and Decoy Transactions, shop payments and fees never join: the Decoy-injection tests of ADR 0001 pass unchanged. New tests in `tests/test_streams.py` (section "stray payments on a stream's schedule") cover the following cases. A stream partly paid with Filler Descriptions on other MCCs is detected whole and stays active. Strays join in a gap, late within the tolerance, and one period before or after the stream. Nothing joins at another amount, off the schedule, beside an existing payment or two periods out. No Decoy Transaction, shop payment, fee or other family's keyword joins. The tolerance is a parameter. Strays never start a stream, and a stream of one payment takes none. A series of strays at another amount stays out.
- **Shift:** every classifier AUC falls (table above), while `n_short` and `n_ended` hardly move. Against valid, the group B AUC falls from 0.733 to 0.668, and against test from 0.796 to 0.697. Mean `max_missed_rate` goes from 0.069 / 0.129 / 0.167 to 0.063 / 0.088 / 0.089.
- **Train out-of-fold** (`rf cv`, Pseudo-Labels of `20260924T171108-3251a4`; nested = decision fitted on 4 folds, applied to the 5th; `oof_score.py`):

  | Model | detector | argmax | tuned | nested tuned |
  |---|---|---|---|---|
  | ranker+pseudo | before | 0.5485 | 0.5819 | 0.5699 |
  | ranker+pseudo | after | 0.5422 | 0.5810 | 0.5651 (−0.0048) |
  | ranker+none+pseudo | before | 0.6154 | 0.6192 | 0.6095 |
  | ranker+none+pseudo | after | 0.6081 | 0.6198 | 0.6082 (−0.0013) |

  Train gets slightly worse, within the −0.01 gate. That's expected: in train the stray payments mark `none` Clients, so filling their gaps removes a train-only signal.
- **Selection** (one run each, `--decision tuned`; paired bootstrap by `rf compare` against the same model on the old detector):

  | Run | Model | macro-F1 | vs | delta (95%) | verdict |
  |---|---|---|---|---|---|
  | `20260925T001614-c904bc` | ranker+pseudo+tuned | 0.5770 | `20260924T171108-3251a4` (0.5735) | +0.0035 (−0.0145 .. +0.0230) | tie |
  | `20260925T001755-e309a3` | ranker+none+pseudo+tuned | 0.5871 | `20260924T225229-cc7243` (0.5764) | +0.0107 (−0.0105 .. +0.0332) | tie |

  Both are the best of their kind on selection, but neither gain is significant. For ranker+none+pseudo, `none` F1 rises 0.6616 → 0.6889, streaming 0.5630 → 0.5793 and music 0.4511 → 0.5036. Gym falls 0.5952 → 0.5632. The train-to-selection gap of the `none` model narrows: nested tuned on train minus selection is 0.033 before and 0.021 after. Per-Client predictions are in `experiments/runs/`, and the log rows are in `experiments/log.csv`. Its automatic comparison is against the best logged run, so the named comparisons are in the conclusion column.
- **Day-2 submissions:** `scripts/day2_final_ranker_none_v2.sh` refits ranker+none+pseudo on train plus the selection set with the new detector. It writes `submissions/day2_final_ranker_none_v2.csv` (valid per `rf submit --check`): none 237, insurance 130, cloud 114, mobile 113, software 109, streaming 108, gym 100, music 89. It agrees with v1 (`day2_final_ranker_none.csv`) on 89.5% of Clients and with the Day-2 12:00 upload on 85.8%. Humans decide whether to upload it. The v1 script and file are unchanged. The Day-2 12:00 upload stays reproducible from `scripts/day2_noon_ranker_pseudo.sh` at the commit it was made from: rerun at `64310b0`, it gives `submissions/day2_noon_ranker_pseudo.csv` byte for byte. On the new detector the same script writes a different file (92.1% agreement), as expected. The same holds for `scripts/day2_final_ranker_none.sh`.
- **Tests:** `uv run pytest -q`: 405 passed (6 slow deselected). `test_every_stream_parameter_is_part_of_the_cache_key` now includes `schedule_tolerance_days`.
- **Not done / open:** the lower `n_ended` of valid and test, and the family words on foreign MCCs that join no stream (3.3 per Client in valid and test, 0.8 in train), are not Filler Description effects and remain. The test set shifts further from train than valid on the old detector. On the new one, `max_missed_rate` is equal for valid and test (0.088 and 0.089), but the classifier still tells test apart a little more easily (A+B 0.725 against 0.700).

## Fix round 1

- **The old detector is still reachable.** `StreamParams.join_strays` (default `True`) switches the Stray Payment join off, and it is part of the stream cache key. `--param` now parses booleans (`true`/`false`, `1`/`0`, `yes`/`no`); before, `bool("false")` was `True`. With `join_strays=false`, the stream table and member payments of train, valid, test and unlabeled match the pre-ticket detector (`64310b0`) exactly (same hashes). The committed milestone submissions are made with it:
  - `scripts/day2_noon_milestone.sh`, `scripts/milestone2_rule.py` and `SOLUTION.md` now refit the rule with `--param join_strays=false`;
  - so do the slow tests `test_gated_rules_reproduce_the_milestone2_submission` and `test_day2_noon_submission_is_reproduced_by_its_committed_commands`. The current code reproduces `milestone2_rules_none_gate_v2.csv` and `day2_noon_rules_gate4.csv` byte for byte. The committed `day2_noon_rules_gate4.md` is unchanged, and it still cites the commands as run at the time.
- **Period floor:** `StreamParams.stray_min_period_days` (7) means a stream with a shorter period takes no Stray Payments. Without it, a 2-payment stream with a 3-day period chains up every stray 3 days apart; a unit test covers this. On real data the floor changes one unlabeled stream (U003064, period 6.99 days: 7 payments without the floor, 6 with it) and nothing in train, valid or test. `bash scripts/day2_final_ranker_none_v2.sh` still writes `submissions/day2_final_ranker_none_v2.csv` byte for byte.
- **Pseudo-Labels change on the new detector** (Shifted Cutoff 2025-10-03, min_payments 4):
  - train: 38 of 2,000 labels change, `none` share 0.1535 → 0.1445 (18 `none` → a family, 20 a family → another family, 0 a family → `none`);
  - unlabeled: 220 of 10,000 change, 0.1409 → 0.1293 (116 `none` → a family, 104 a family → another family, 0 a family → `none`).
- **Fidelity check on the new detector:** `20260925T011208-e56725` (default search space). It passes and chooses the same setting as before (min_payments 3, min_payments_before 0, churn 0.2):
  - the chosen setting's none share is 0.3005 against the real 0.2985 (was 0.3035);
  - the gated rule scores 0.5402 on the Pseudo-Labels against 0.5245 on the real labels (was 0.5433 against 0.5331).

  The setting the ranker runs use (min_payments 4, no churn) still fails: none share 0.1445 against 0.2985, rule macro-F1 gap +0.1107. The two ticket-11 selection rows (`c904bc`, `e309a3`) cite fidelity check `add1e5` in their `fidelity` column. That check was made on the old detector; the one on the new detector is `e56725`. The slow fidelity test is re-pinned to the new figures, with the reason stated in the test.
- **ADR 0001** has an amendment on Stray Payments, and **CONTEXT.md** defines "Stray Payment". The code keeps its names.
- **Tests:** the fast suite (`uv run pytest -q`) and `uv run pytest -q -m slow`. Two slow tests were skipped because they run `rf evaluate` on the selection set: `test_e1_real_data::test_e1_macro_f1_on_the_valid_selection_set_is_near_the_reference` and `test_ranker_real_data::test_the_ranker_trains_and_scores_the_selection_set_within_the_evaluation_budget`. Results: slow 4 passed; fast 415 passed (6 slow deselected).
- **Follow-ups (not done here):**
  - Cap how many Stray Payments a stream may take. This would change train outputs: 22 train streams have at least as many strays as payments of their own, against 3 each in valid and test.
  - Review the 3 streams that turn from active to inactive and the 3 that flip between music and streaming on the new detector.
