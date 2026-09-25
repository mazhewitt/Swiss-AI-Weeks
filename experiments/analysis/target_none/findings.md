# Ticket 18: a `none` signal of the target domain's own? (capped diagnostic)

Scripts: `features.py` (the two feature blocks), `diagnostic.py` (the pre-registered test),
`posthoc_drift.py` (context only, not pre-registered). Numbers: `results.json`. Caches go in `cache/`, which
git ignores.

**Footing.** 700 selection Clients (205 `none`, 495 family), inside `data.training_run(with_selection=True)`.
Selection labels are read only through the sanctioned `load_labels("selection")` guard. The sealed holdout's
labels and transactions are never used. Features come from history before the Cutoff. LightGBM uses the v2
`none` model's fixed parameters (`n_jobs=1`, no tuning), repeated 5x5 stratified CV (seeds 0-4), with the
out-of-fold scores averaged over the repeats. CIs are DeLong 95%.

**Blocks.**
- **base**: the v2 `none` model's 104 stream features.
- **raw** (130 features): ticket 16's kitchen sink with all description-word bags dropped (the MCC bags stay), plus:
  - Decoy count, Decoy share of card payments, and the last-90-day Decoy share normalised by the Client's own last-90-day activity share;
  - Decoys within 6% (log-amount) of an Active Stream's amount;
  - currency, hour, type and MCC mix;
  - refund counts, windows and timing.

## Pre-registered measures

| Block | `none` AUC | DeLong 95% CI | per-repeat AUCs |
|---|---|---|---|
| raw | **0.807** | [0.772, 0.843] | 0.790 - 0.810 |
| base | 0.821 | [0.787, 0.856] | 0.812 - 0.824 |
| base+raw | **0.873** | [0.845, 0.902] | 0.854 - 0.879 |

- **Paired DeLong, base+raw against base:** +0.052, 95% CI [+0.024, +0.080], z = 3.69, p = 0.0002.
- **Adversarial AUC of the top 10 raw features** (label-free, 5-fold):

  | Pair | AUC |
  |---|---|
  | train vs test | **0.9998** |
  | selection vs test | 0.802 |
  | train vs selection | 0.9985 |

**Gate:** the raw AUC is 0.807, which is at least 0.70, and its CI excludes 0.5. **The gate passes. Recommend a
build (a Daumé / `is_target` stack) as a new ticket.**

## What carries it

The top 10 raw features by gain are listed below. The single-feature AUC on selection comes first, then the
single-feature selection-vs-test drift AUC.

| Feature | `none` AUC | Direction | Drift AUC |
|---|---|---|---|
| `decoy_share_of_card` | 0.764 | high -> `none` (0.336 vs 0.248) | **0.782** |
| `type_share_refund` | 0.666 | high -> `none` | 0.521 |
| `n_desc` (distinct descriptions) | 0.683 | low -> `none` | 0.557 |
| `refund_type_gap_median` | 0.597 | low -> `none` | 0.505 |
| `refund_type_first_days` | 0.555 | low -> `none` | 0.518 |
| `refund_type_share_of_card` | 0.667 | high -> `none` | 0.521 |
| `last_family_named_days` | 0.627 | high -> `none` | 0.545 |
| `decoy_rate90_minus_before` | 0.618 | high -> `none` | 0.526 |
| `cur_entropy` | 0.550 | low -> `none` | 0.501 |
| `refund_type_amount_mean` | 0.570 | low -> `none` | 0.505 |

- **The lead is the Decoy share of card payments.** It is normalised by the Client's own card activity, so it is
  not the retracted, activity-proportional "Decoy rise". Train barely has Decoys (mean share 0.014), which is
  why train labels could never teach it. It is a target-domain marker.
- **The lead also shifts between selection and test.** The mean share is 0.273 on selection and 0.394 on test.
  That shift is almost all of the 0.80 selection-vs-test adversarial AUC: the other nine features each sit at
  0.50-0.56. A build must not use the Decoy share as an absolute level. Rank it within each domain, or check
  that the higher test level is not simply more `none` Clients. Test has no labels, so that cannot be told
  apart here.
- **Refund share and timing form a second group, and they do not drift** (all about 0.52 selection vs test, and
  about 0.52 train vs test).
- **Post hoc (not pre-registered):** raw with all 15 Decoy features removed still scores 0.762 [0.724, 0.800],
  which is -0.045 against full raw (p < 0.001). So the signal is not one feature.

## Limits

- This is a `none` AUC, not the macro-F1 that is scored. Ticket 13's ceiling said fixing every `none` error on
  selection would take v2 from 0.587 to about 0.70. A +0.05 `none` AUC is a fraction of that.
- The base `none` AUC here (0.82) is below its train figure in ticket 16 (0.89). The selection Clients are
  harder for the stream features.
- Everything is learned from 700 labelled Clients. A build must be judged by nested CV on selection before any
  test upload, because this diagnostic is not a macro-F1 comparison.
