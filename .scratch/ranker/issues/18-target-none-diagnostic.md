# 18: Is there a `none` signal of the target domain's own? (capped diagnostic)

Status: ready-for-agent

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

(to fill)
