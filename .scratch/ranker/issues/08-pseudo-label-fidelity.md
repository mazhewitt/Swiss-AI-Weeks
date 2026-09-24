# 08: Pseudo-Labels that pass the fidelity check

**What to build:** Find a Pseudo-Label setup that passes the fidelity check from ticket 04, and retrain the Stream Ranker on it. Today's check fails (run `20260924T162515-d6d36f`). At `min_payments` 4 the Pseudo-Label `none` share is 0.15 against a real 0.30. At 7 the share matches, but the rule scores 0.78 against Pseudo-Labels versus 0.53 on real labels, so the Pseudo-Label task is much easier than the real one. The ranker trained on the current Pseudo-Labels only ties the one trained without them (+0.002 on selection).

First, find out why the tasks differ. Then change the labeller and the check's search space to match. Ideas from ticket 04:

- ignore a Horizon payment of a stream that starts only inside the Horizon;
- a looser or stricter Horizon match;
- a Shifted Cutoff earlier than 2025-10-03, or several Shifted Cutoffs pooled (each Horizon must be fully observed).

Any other labeller change that makes the Pseudo-Label task look like the real one is also in scope, for example modelling how real Recurring Streams stop. The real labels are the ground truth for what "looks like" means. Every choice is made on train only.

**Blocked by:** 04, 05

**Status:** done

- [x] A written diagnosis of why the Pseudo-Label task is easier than the real one, with figures from train (a comment on this ticket)
- [x] The labeller and the fidelity check support the new options. Every new option has a unit test at the labeller seam, and the cache key includes every new parameter
- [x] The fidelity check over the new search space is logged. Report PASS or FAIL honestly, and don't loosen the tolerances
- [x] If a setup passes or clearly narrows both gaps: the ranker is cross-validated on train with it and compared with the current Pseudo-Label ranker on train out-of-fold (tuned macro-F1, argmax macro-F1). Only if the train result is better, one run is logged on the selection set with a paired bootstrap against run `20260924T171108-3251a4` (0.5735), and the predictions are committed
- [x] The default Pseudo-Label settings (`PSEUDO_MIN_PAYMENTS`, the CLI defaults) change only when the new setup wins on train. Otherwise they stay as they are, and the ticket records the result
- [x] The full fast test suite passes

## Comments

**Diagnosis (train only; real labels are train's, Pseudo-Labels at Shifted Cutoff 2025-10-03, min_payments 4 unless stated).** Real: `none` share 0.2985, milestone-2 rule macro-F1 0.5331. Pseudo-Labels: 0.1535 and 0.6453.

The gap is churn at the Cutoff: the real Horizon has Clients whose streams all stop at the Cutoff, and no Horizon inside the known history does.

- Error profile of the rule, as shares of the 2,000 train Clients:

  | | real | Pseudo-Labels |
  |---|---|---|
  | rule predicts a family, label `none` | 0.148 | 0.004 |
  | rule predicts `none`, label a family | 0.072 | 0.190 |
  | rule predicts a family, label another family | 0.239 | 0.165 |
  | coverage (label family among the surviving streams) | 0.763 | 0.789 |
  | selection accuracy given coverage | 0.728 | 0.735 |

  Picking among streams is equally hard in both tasks. The difference is the first row: at the real Cutoff 19% of the rule's family predictions are `none` Clients, but inside the history almost none are.
- `none` share by the Client's longest surviving stream (payments at the Cutoff):

  | payments | 0 | 3 | 4 | 5 | 6 | 7-8 | 9-10 | 11-14 | 15+ |
  |---|---|---|---|---|---|---|---|---|---|
  | real | 0.65 | 0.73 | 0.73 | 0.37 | 0.28 | 0.12 | 0.15 | 0.20 | 0.18 |
  | Pseudo-Labels | 0.61 | 0.23 | 0.16 | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

  Long, healthy streams end in `none` for about 17% of real Clients (1,356 Clients with 7 or more payments) and for none of the Pseudo-Labelled ones. Young streams (3-4 payments) end in `none` 73% of the time really, against 16-23%.
- Clients with exactly one surviving stream: the real label is that stream's family for 50% (726 Clients; 31.5% `none`). Inside the history the stream pays again in the Horizon for 91% at 2025-10-03, 99.6% at 2025-07-05 and 99.1% at 2025-04-06. Of the 3,473 full-history streams with 4 or more payments, 79% still pay in December 2025 and only 7% stop before October.

Hypotheses:

- **(a) streams that start inside the Horizon: minor, and in the other direction.** They carry 0.7% of Pseudo-Labels at min_payments 4 (4.5% at 3, 10% at 2). Real family labels in no stream detected at the Cutoff: 6.2%, against 3.0% for Pseudo-Labels. Ignoring them (min_payments_before 1) makes the task easier, not harder: rule 0.6493 at min_payments 4.
- **(b) streams stop at the real Cutoff: the main cause** (figures above). Only a labeller that churns Clients at the Shifted Cutoff can reproduce it: the history has no such stops to observe.
- **(c) the detector's own families cancel out: minor.** Coverage and selection accuracy are within 0.03, and music/streaming swaps are 30 real against 18 Pseudo-Labelled.
- **(d) seasonality: rejected.** Without `none`, the family mix of the Oct-Dec Horizon matches the real Jan-Mar mix within 0.01 per family, and the detector knows only biweekly and monthly periods.
- **(e) earlier Shifted Cutoffs change the task for other reasons.** Median history before the Cutoff: 13.6 months real, 10.7 at 2025-10-03, 7.7 at 2025-07-05. At 2025-07-05 some settings pass the check, but by coincidence: coverage is 0.56, 19% of family labels are in no stream yet, the rule says `none` to 46% of family-labelled Clients, and "rule family, label `none`" is 0.000. At 2025-04-06 and 2025-01-06 the rule collapses (macro-F1 at most 0.29 and 0.10), because most streams have not started.

**Labeller options (`streams.LabellerParams`, all in the Pseudo-Label cache key).**

- `churn` P: a share P of Clients have all their streams stop at the Shifted Cutoff and get `none`. Which Clients is a fixed hash draw per Client and Shifted Cutoff. So tables are reproducible, a larger share churns a superset of Clients, and a Client pooled at two Shifted Cutoffs churns at each independently. The share estimated directly from train is about 0.17 (long-stream Clients, above).
- `min_payments_before` K: a stream's Horizon payment counts only if the stream has K payments before the Shifted Cutoff. 1 ignores streams that start inside the Horizon; 3 roughly means "already an Active Stream".
- CLI: `rf pseudo-labels --min-payments-before K --churn P`. The fidelity check searches `--candidates` (min_payments, 2-10) × `--before-candidates` (0-3) × `--churn-candidates` (0, 0.1, 0.15, 0.2, 0.25, 0.3). That is 198 settings once repeats are dropped; ties go to the simplest setting. The ranker takes `--pseudo-min-payments-before` and `--pseudo-churn`, which are recorded in the metadata and in two new log columns.
- Pooled Shifted Cutoffs already work (`--pseudo` repeated). They were not pursued, because the earlier Horizons are less like the real one (above).

## Outcome

**Fidelity check: PASS** at the default Shifted Cutoff (run `20260924T214629-add1e5`). The tolerances are unchanged. Real: `none` share 0.2985, rule macro-F1 0.5331.

| Shifted Cutoff | min_payments | min_payments_before | churn | none share | rule macro-F1 | gaps | |
|---|---|---|---|---|---|---|---|
| 2025-10-03 | 4 | 0 | 0 | 0.1535 | 0.6453 | -0.145 / +0.112 | fail (ticket 04's setup) |
| 2025-10-03 | 4 | 1 | 0 | 0.1565 | 0.6493 | -0.142 / +0.116 | fail (new streams ignored) |
| 2025-10-03 | **3** | **0** | **0.2** | **0.3035** | **0.5433** | **+0.005 / +0.010** | **pass (chosen)** |
| 2025-10-03 | 4 | 0 | 0.2 | 0.3320 | 0.5835 | +0.034 / +0.050 | fail (just) |
| 2025-07-05 | 3 | 1 | 0.1 | 0.3150 | 0.5158 | +0.017 / -0.017 | pass (logged `20260924T214603-38acdf`; a coincidental match, see (e)) |

At 2025-10-03, 9 of the 198 settings pass, and every one has churn 0.15-0.25. No setting without churn passes there. The chosen churn (0.2) is close to the 0.17 estimated directly from long-stream Clients.

**Ranker on train out of fold** (5 folds; sources train + unlabeled at 2025-10-03, weight 0.5; tuned = the decision layer fitted on the same out-of-fold rows; nested = fitted on 4 folds, applied to the 5th):

| Pseudo-Labels | argmax | tuned | nested tuned |
|---|---|---|---|
| min_payments 4, no churn (current) | 0.5485 | 0.5819 | 0.5699 |
| min_payments 3, churn 0.2 (check's choice) | 0.5222 (-0.026, 95% -0.043..-0.010) | 0.5916 (+0.010, -0.006..+0.025) | 0.5738 (+0.004, -0.012..+0.020) |
| min_payments 4, churn 0.2 | 0.5365 (-0.012) | 0.5932 (+0.011) | 0.5734 (+0.004) |

Churn raises every Client's `none` probability, so argmax says `none` too often, and the tuned `none` threshold undoes it. On the tuned decision, which `evaluate --decision tuned` scores, the check's choice was higher on train, so it got the one selection run.

**Selection set, run `20260924T215400-3a4f12`** (`ranker+pseudo+tuned`, min_payments 3, churn 0.2): macro-F1 0.5666. Against `20260924T171108-3251a4` (0.5735): delta -0.0069 (95% -0.0322 .. +0.0183), a **tie**. Predictions are in `experiments/runs/`.

**Conclusion.** The Pseudo-Label task now looks like the real one on both of the check's figures, and the diagnosis explains why it did not before: the real Cutoff has churn that no Horizon inside the history has. However, Pseudo-Labels that match the real task did not make the ranker better. The train gain is within noise and argmax got worse, and on the selection set it ties (slightly below) the current setup. The defaults (`PSEUDO_MIN_PAYMENTS` 4, no churn, no minimum before) stay as they are. The churn is random by construction (the real churn looks unpredictable from the history: a flat 12-20% `none` across long streams). So it mainly recalibrates the `none` probability, and the tuned decision layer already did that. The real task's predictable part that Pseudo-Labels still miss is young streams (3-4 payments) that end in `none` 73% of the time. A stream-age-dependent stop model is the next idea, if Pseudo-Labels get another round.
