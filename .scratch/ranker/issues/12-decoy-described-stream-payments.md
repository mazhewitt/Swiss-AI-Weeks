# 12: Stream payments that carry a Decoy description

**What to build:** Let a payment with a Decoy description ("digital order", "merchant charge", ...) join an existing Recurring Stream, but only under the Stray Payment rules from ticket 11:

- the payment fits the stream's amount range;
- it fits an empty slot on the stream's schedule, within `schedule_tolerance_days`;
- the stream has at least two payments and a period of at least `stray_min_period_days`;
- it never starts a stream.

Ticket 11's diagnosis (`experiments/analysis/filler_streams/`, the ticket's Comments) found 8.7 Decoy-described payments per 100 valid streams that fill a gap on the schedule, and 7.3 in test. The off-schedule control is 1.0 in valid. In train the on-schedule rate is 0.88, the same as its control. So in valid and test, some real stream payments carry a Decoy description, while in train almost none do. They break valid and test streams just as Filler Descriptions did.

ADR 0001's concern is that Decoy Transactions drift (0.16 per Client in train, 3.1 in valid, 4.5 in test) and must not create features. A random Decoy lands on an empty slot at the stream's amount at about the control rate. So the join must be judged by the gap between the on-schedule rate and the control rate in each split, and the ADR amended with those figures.

**Blocked by:** 11

**Status:** done

- [x] Figures per split (label-free): Decoy-described on-schedule joins against the off-schedule control (the expected false-join rate), before the change
- [x] Behind `StreamParams.join_decoys` (default on only if the gate below passes, and in the cache key): Decoy-described payments join under the Stray Payment rules. Unit tests: a Decoy on an empty slot at the stream's amount joins; off schedule, at another amount or on a stream shorter than two payments it does not; it never starts a stream. The ADR 0001 Decoy-injection test is updated to inject off-schedule or off-amount Decoys and still passes, plus a test that random Decoys rarely join
- [x] The label-free shift check as in ticket 11 (classifier AUC train vs valid and train vs test on the candidate fields and the `none` model's features; mean `max_missed_rate`, `n_short`, `n_ended`), before and after
- [x] Train out-of-fold (argmax, tuned, nested tuned) for ranker+pseudo and ranker+none+pseudo, before and after
- [x] Gate: the shift narrows and train gets worse by no more than 0.01 nested. Then one selection run of ranker+none+pseudo (tuned), with `rf compare` against `20260925T001755-e309a3` (0.5871), predictions committed, and `submissions/day2_final_ranker_none_v3.csv` from a v3 script (refit on train plus selection, `rf submit --check`). Otherwise leave `join_decoys` off by default and record the result
- [x] ADR 0001 amended with the figures; `--param join_decoys=false` reproduces the ticket 11 outputs; slow tests (except those that run `rf evaluate`) and the fast suite pass

## Comments

### Before: Decoy-described payments on a stream's schedule (label-free)

`experiments/analysis/filler_streams/gaps.py` (tolerance 4 days) on the ticket-11 detector (`8623d2e`). Per 100 streams of two or more payments (train 5,341, valid 2,129, test 2,009), Decoy-described payments left out of every stream at a stream's amount:

| position | train | valid | test |
|---|---|---|---|
| in a gap, on the schedule | 0.88 | 8.64 | 6.77 |
| in a gap, half a period off (control) | 0.06 | 0.94 | 1.44 |
| one or two periods after the last payment | 0.39 | 3.52 | 4.08 |
| one or two periods before the first payment | 0.41 | 4.70 | 5.57 |

On-schedule against control: 15x in train, 9x in valid and 5x in test. Train has few Decoy Transactions (0.16 per Client), so both its rates are small. The control is the expected false-join rate.

### The change

`StreamParams.join_decoys` (in the cache key; on by default at 73ec6a5, off since the Decision below): a payment with a Decoy description (a Decoy word and no shop or fee word, `streams._decoy_described`) that no stream took joins under the Stray Payment rules (`_join_on_schedule`), in a second pass after the stray payments have taken their slots. It needs `join_strays` too, so `join_strays=false` still gives the detector before ticket 11 and the milestone reproductions are unchanged. It never starts a stream. `_evidence` is unchanged (a Decoy description is still `drop`, so refunds ignore it).

Decoy-described payments that join, per 100 streams (`experiments/analysis/filler_streams/decoy_joins.py`):

| | train | valid | test |
|---|---|---|---|
| joined | 1.54 | 14.70 | 13.29 |
| of which in a gap | 0.95 | 8.88 | 7.62 |
| of which at an end | 0.58 | 5.82 | 5.67 |
| expected false joins in a gap (control) | 0.06 | 0.94 | 1.44 |
| share of Clients with one | 3.9% | 19.4% | 16.5% |

Unlabeled has no Decoy Transactions, so its streams and Pseudo-Labels are unchanged. After the change, `gaps.py` finds 0.07 / 1.03 / 1.34 Decoy-described payments per 100 streams in a gap against controls of 0.04 / 0.70 / 0.80. Missed payments per stream: 0.33 / 0.57 / 0.57 before, 0.32 / 0.49 / 0.50 after.

`--param join_decoys=false` reproduces ticket 11: the stream table and member payments of train, valid, test and unlabeled have the same hashes as at `8623d2e`.

### Shift check, before and after (`shift11.py`)

| Measure | before (ticket 11) | after |
|---|---|---|
| AUC train vs valid: ranker candidate fields | 0.704 | 0.683 |
| AUC train vs valid: `none` model group A | 0.594 | 0.570 |
| AUC train vs valid: `none` model group B | 0.668 | 0.635 |
| AUC train vs valid: `none` model A+B | 0.700 | 0.653 |
| AUC train vs test: ranker candidate fields | 0.753 | 0.741 |
| AUC train vs test: group A | 0.618 | 0.598 |
| AUC train vs test: group B | 0.697 | 0.663 |
| AUC train vs test: A+B | 0.725 | 0.694 |
| mean `max_missed_rate` train / valid / test | 0.063 / 0.088 / 0.089 | 0.062 / 0.082 / 0.080 |
| mean `n_short` train / valid / test | 1.116 / 1.457 / 1.301 | 1.103 / 1.342 / 1.208 |
| mean `n_ended` train / valid / test | 0.583 / 0.290 / 0.281 | 0.590 / 0.342 / 0.325 |
| Active Streams per Client train / valid / test | 1.61 / 1.38 / 1.37 | 1.61 / 1.44 / 1.42 |
| stream payments per Client train / valid / test | 18.45 / 15.94 / 16.15 | 18.49 / 16.26 / 16.42 |

Every AUC falls, and the `n_short` and `n_ended` gaps that ticket 11 left alone narrow too (train-to-valid `n_ended` gap 0.293 → 0.248, `n_short` 0.341 → 0.239). The shift narrows.

### Train out-of-fold (`rf cv`, Pseudo-Labels train+unlabeled@2025-10-03, weight 0.5, min_payments 4; `oof_score.py`)

| Model | detector | argmax | tuned | nested tuned |
|---|---|---|---|---|
| ranker+pseudo | ticket 11 | 0.5422 | 0.5810 | 0.5651 |
| ranker+pseudo | ticket 12 | 0.5456 | 0.5805 | 0.5643 (−0.0008) |
| ranker+none+pseudo | ticket 11 | 0.6081 | 0.6198 | 0.6082 |
| ranker+none+pseudo | ticket 12 | 0.6094 | 0.6179 | 0.6005 (−0.0077) |

The ticket-11 rows reproduce that ticket's "after" figures exactly.

### Gate

Passed: the shift narrows, and nested tuned for ranker+none+pseudo falls by 0.0077 (limit 0.01). `join_decoys` was set on by default for the selection run and v3 (73ec6a5), then turned off (Decision below).

## Outcome

- **Detector:** `StreamParams.join_decoys` (in the cache key; default off, see the Decision below) lets a Decoy-described payment join an existing stream under the Stray Payment rules, after the strays. It never starts a stream, and shop payments and fees never join. It needs `join_strays`, so `join_strays=false` still restores the detector before ticket 11 and the milestone reproductions don't change. `--param join_decoys=false` reproduces ticket 11's stream tables and member payments for train, valid, test and unlabeled exactly (same hashes as at `8623d2e`).
- **Tests** (`tests/test_streams.py`, section "Decoy-described payments on a stream's schedule"): a Decoy joins on an empty slot at the stream's amount (in a gap, 2 days late, at 0.9 of the amount tolerance, one period before or after). It does not join off the schedule, half a period off, beside a payment, two periods out, at another amount, or as a shop payment or fee with a Decoy word. It never starts a stream or joins a single-payment stream. It needs `join_strays`, and a stray payment takes a slot before a Decoy does. Random Decoys at test-like rates at the exact stream amount join rarely (under 10%, above 0), never at another amount, and never without `join_decoys`. The ADR 0001 Decoy-injection tests (stream table and Pseudo-Labels) run with `join_decoys` off and on. The shadow test now injects Decoys half a period off the schedule and at another amount on an empty slot. It checks that a Decoy at the stream's amount on that slot joins only with `join_decoys`. The `none` model's Decoy test (`tests/test_ranker_none_model.py::with_decoys`) now places its Decoys half a period off each Client's stream. `test_every_stream_parameter_is_part_of_the_cache_key` covers `join_decoys`.
- **Figures:** see the Comments. The expected false-join rate is the off-schedule control: 0.06 / 0.94 / 1.44 per 100 streams in gaps (train / valid / test), against 0.95 / 8.88 / 7.62 actual gap joins.
- **Gate:** passed. The shift narrows on every measure, and nested tuned for ranker+none+pseudo falls by 0.0077 (0.6082 → 0.6005).
- **Selection** (one run, `--decision tuned`): `20260925T014822-cd37c4` ranker+none+pseudo+tuned scores macro-F1 0.5884. `rf compare --run 20260925T014822-cd37c4 --run 20260925T001755-e309a3`: delta +0.0012 (95% −0.0187 .. +0.0205), a tie. The winner is `cd37c4` on ties by simplicity. Per family against `e309a3`: gym 0.5632 → 0.5862, cloud 0.5778 → 0.5882, mobile 0.6299 → 0.6395, music 0.5036 → 0.5079, insurance 0.6301 → 0.6259, streaming 0.5793 → 0.5714, software 0.5241 → 0.5039, none 0.6889 → 0.6838. The train-to-selection gap narrows again: nested tuned on train minus selection is 0.012 (0.021 on ticket 11). Predictions are in `experiments/runs/20260925T014822-cd37c4.csv`, and the comparison is in the log's conclusion column.
- **Day-2 submissions:** `scripts/day2_final_ranker_none_v3.sh` refits ranker+none+pseudo on train plus the selection set on the ticket-12 detector. It writes `submissions/day2_final_ranker_none_v3.csv` (valid per `rf submit --check`): none 245, insurance 123, cloud 116, software 110, streaming 106, gym 104, mobile 102, music 94. It agrees with v2 on 95.0% of Clients, with v1 on 88.1%, and with the Day-2 12:00 upload on 83.8%. Humans decide whether to upload it. The v1 and v2 scripts and files are unchanged. The v3 file is reproducible from its script at `73ec6a5` only, where `join_decoys` defaulted on. The ranker commands take no stream parameters, so after the default went back to off the script runs v2's commands on v2's detector, as its header says.
- **Pseudo-Label fidelity:** with `join_decoys` on, the slow fidelity test kept the same choice (min_payments 3 / before 0 / churn 0.2), and its figures moved to: real rule macro-F1 0.5245 → 0.5236, chosen none share 0.3005 → 0.3000, rule macro-F1 on it 0.5402 → 0.5380. That test was re-pinned at `73ec6a5`. With the default back to off it is pinned to the ticket-11 figures again. No fidelity run was logged. Unlabeled Pseudo-Labels don't depend on the flag, since unlabeled has no Decoys.
- **Risk:** in test about 1 in 5 of the gap joins is expected to be a real Decoy Transaction (control 1.44 against 7.62). A Decoy that joins at a stream's end can extend it by one period, and so make an ended stream look active. Active Streams per Client rise from 1.38 to 1.44 in valid and from 1.37 to 1.42 in test, while train stays at 1.61. The selection tie shows no harm, but none F1 fell slightly (0.6889 → 0.6838).
- **Test runs at `73ec6a5`** (join_decoys on): `uv run pytest -q`: 443 passed (6 slow deselected). `uv run pytest -q -m slow`: 4 passed. Two slow tests were skipped because they run `rf evaluate` on the selection set: `test_e1_real_data::test_e1_macro_f1_on_the_valid_selection_set_is_near_the_reference` and `test_ranker_real_data::test_the_ranker_trains_and_scores_the_selection_set_within_the_evaluation_budget`.

## Decision: `join_decoys` defaults to off

Made by the orchestrator after the gate passed. The code stays, and `StreamParams.join_decoys` defaults to `False`, so the default detector is ticket 11's again. ADR 0001's rule against Decoy drift wins at a tie. Reasons:

- the selection gain is a tie: +0.0012 (95% −0.0187 .. +0.0205);
- train nested tuned gets worse by 0.0077;
- about 1 in 5 of the gap joins in test, the split with the most Decoys, is expected to be a Decoy Transaction;
- a Decoy that joins at a stream's end can turn an ended stream active.

What changed:

- With the defaults, the stream table and member payments of train, valid, test and unlabeled have the same hashes as at `8623d2e` (checked).
- The slow fidelity test is pinned to the ticket-11 figures again (0.5245 / 0.3005 / 0.5402).
- The `join_decoys` unit tests pass the flag explicitly and stay.
- The changed `with_decoys` in `tests/test_ranker_none_model.py` stays: it passes, and it fails on a mutant that feeds the `none` model a raw transaction count.
- The v3 script's header says it reproduces `submissions/day2_final_ranker_none_v3.csv` only at `73ec6a5`.
- ADR 0001 records the figures, the gate and this decision.

**Follow-up (not done):** let a Decoy-described payment fill a gap between a stream's payments, but never extend a stream beyond its first or last payment. Then re-run the shift check, train out-of-fold and gate.

**Tests after the decision:** `test_every_stream_parameter_is_part_of_the_cache_key` now sets `join_decoys=true`, the non-default value. `uv run pytest -q`: 443 passed (6 slow deselected). `uv run pytest -q -m slow`: 4 passed, with the same two tests that run `rf evaluate` on the selection set skipped.
