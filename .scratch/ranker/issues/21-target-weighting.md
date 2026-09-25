# 21: Weight the valid-like Clients up in the refit (pre-registered)

Status: ready-for-agent

**Why:** train and valid/test differ, and in the final refit 2,000 train Clients outweigh 700 selection Clients about 3 to 1. In Santander's second-place solution, the winning move was to train on the data that matched the target. Here, the lever is weighting the target-domain labelled Clients up. **Deadline: the final file by 17:10 CEST.** If it isn't ready by then, the hail mary stays.

## Candidate

- **Hail mary A with the valid Clients at weight 3.**
  - P = ½ P(v2) + ½ P(hard Survival Race), as in `hailmary.py` configuration A: the same factories, seeds and Pseudo-Labels.
  - The labelled valid Clients count **3 times** in the model fits. They are implemented as **duplicates**: two extra copies of each labelled valid Client, with prefixed ids, as in the hail mary's `self:` mechanism. For tree models this approximates weight 3.
- **The decision layer (E3) is shared.** It is fitted on the out-of-fold probabilities of the **unweighted** fit, exactly as in configuration A. Duplicates therefore never enter a cross-validation split, which avoids leakage between copies.
- **Reference:** the same thing at weight 1, which is configuration A.

## Rehearsal (cross-fitted over two halves of the selection set)

- Split the 700 selection Clients into two fixed halves (seed 0, stratified by label).
- For each half h:
  - fit v2 and the race on train + the other half (weight 1, and separately weight 3);
  - fit E3 on the out-of-fold probabilities of the weight-1 fit on train + the other half;
  - predict half h.
- Each selection Client is predicted exactly once per weight, by models that never saw its label.
- Freeze both prediction sets before any selection label is read in `data.scoring()`. The fitting labels are read only inside `data.training_run(with_selection=True)`, restricted to the other half.
- **Score:** pooled macro-F1 over 700 for w = 3 and for w = 1, plus a paired bootstrap (2,000 resamples, seed 0).

## Upload rule (fixed before any number)

The 17:30 upload becomes the **final w = 3 file** only if all three hold:
- w = 3 beats w = 1 on the pooled rehearsal by more than 0 (point estimate);
- the w = 3 final file disagrees with v2's file on at least 10% of test Clients;
- it is ready and passes `rf submit --check` by 17:10.

Otherwise the hail mary stays. The final w = 3 file is fitted on train plus all 700 selection Clients at weight 3, with E3 fitted on the weight-1 out-of-fold probabilities, and predicts test. The claims rule (0.03) is unchanged; the result is reported as a tie unless it clears 0.03.

## Rules

- The sealed holdout is never read. Code goes in `experiments/analysis/target_weight/`.
