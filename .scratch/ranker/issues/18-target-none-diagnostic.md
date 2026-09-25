# 18: Is there a `none` signal of the target domain's own? (capped diagnostic)

Status: done

**Why:** possibility 3 from the research round (ticket 17). The train `none` markers do not transfer (ticket 16). The only open question is whether valid/test carry their own `none` marker, one that only valid labels can teach. The Decoy-rise lead was retracted: it is proportional to activity. This ticket is the one bounded check. **Timebox: 2 hours. No build in this ticket.**

## Pre-registered test (fixed before any number)

- **Data:** the 700 selection Clients only, inside `data.training_run(with_selection=True)`. The labels are the selection labels. The sealed holdout is never read, and test labels do not exist.
- **Two feature blocks, per Client, from history before the Cutoff:**
  - **base:** the v2 `none` model's stream features;
  - **raw:** ticket 16's kitchen sink **without description words or description bags**, plus:
    - Decoy counts, and their last-90-day share normalised by the Client's own activity;
    - Decoys within 6% of an Active Stream's amount;
    - currency, hour and MCC mix;
    - refund counts and timing.
- **Model:** LightGBM, `n_jobs=1`, small fixed parameters (no tuning), binary `none` against family. Repeated 5×5 stratified CV (seeds 0–4); average the out-of-fold scores across repeats.
- **Measures:**
  1. **the gate:** standalone `none` AUC of **raw**, with a DeLong 95% CI;
  2. `none` AUC of base, of base+raw, and a paired DeLong test of base+raw against base;
  3. the train-vs-test adversarial AUC of the top 10 raw features (label-free), as context.
- **Gate:** raw AUC ≥ 0.70 with the CI excluding 0.5 → recommend a build (a Daumé / `is_target` stack) as a new ticket. Otherwise, close the target-domain `none` route in writing.

## Result

Scripts, numbers and notes: `experiments/analysis/target_none/`. The test ran as pre-registered on the 700
selection Clients (205 `none`) inside `data.training_run(with_selection=True)`. The sealed holdout was not read.

| Block | `none` AUC | DeLong 95% CI |
|---|---|---|
| raw (130 features) | **0.807** | [0.772, 0.843] |
| base (104 stream features) | 0.821 | [0.787, 0.856] |
| base+raw | 0.873 | [0.845, 0.902] |

- **Paired DeLong, base+raw against base:** +0.052, CI [+0.024, +0.080], p = 0.0002.
- **Adversarial AUC of the top 10 raw features** (label-free):

  | Pair | AUC |
  |---|---|
  | train vs test | 0.9998 |
  | selection vs test | 0.802 |
  | train vs selection | 0.9985 |

- **Gate: passes.** Raw 0.807 is at least 0.70, and the CI excludes 0.5. **Recommend a build (a Daumé /
  `is_target` stack) as a new ticket.**
- **The lead feature** is the Decoy share of card payments (single-feature AUC 0.76; high means `none`). Train
  has almost no Decoys (share 0.014), so only valid labels can teach it.
- **The lead also drifts from selection to test** (mean 0.27 against 0.39, single-feature AUC 0.78). It is
  nearly all of the 0.80 selection-vs-test adversarial AUC. The other top features each sit at 0.50–0.56. A
  build must use it relative within each domain, not as an absolute level.
- **Refund share and timing** form a second group that does not drift.
- **Post hoc:** raw without its 15 Decoy features still scores 0.762 [0.724, 0.800].
- **Caveat:** this is a `none` AUC, not macro-F1. The build ticket must show a nested-CV macro-F1 gain on
  selection before any upload.
