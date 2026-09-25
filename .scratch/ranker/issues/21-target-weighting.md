# 21: Weight the valid-like Clients up in the refit (pre-registered)

Status: done

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

## Results (rehearsal over the 700 selection Clients, cross-fitted over two halves)

Code: `experiments/analysis/target_weight/target_weight.py` (imports the hail mary's factories, `decision_and_prior` and `target`); caches under `artifacts/target_weight/`; numbers in `experiments/analysis/target_weight/results.json`. Both prediction sets (`predictions/rehearsal_w{1,3}.csv`) were committed before the selection labels were read for scoring.

- Halves: 350 / 350, `StratifiedKFold(2, shuffle, seed 0)` over the selection Clients sorted by id (ids only stored, in `artifacts/target_weight/halves.csv`).

| Weight | Pooled macro-F1 (700) |
|---|---|
| w = 1 (configuration A) | 0.5971 |
| **w = 3** | **0.6054** |

- **Paired bootstrap (w = 3 − w = 1, 2,000 resamples, seed 0):** +0.0082 (95% −0.0052 .. +0.0227). The two agree on 95.4% of the selection Clients.
- Per label, w = 3 gains most on cloud (+0.024), software (+0.024), insurance (+0.017) and none (+0.011); it loses on mobile (−0.013).

**Final files (train + all 700 selection Clients):**
- **Check:** the final w = 1 test probabilities reproduce `artifacts/hailmary/final` configuration A exactly (max probability difference 0.0, 100% label agreement).
- The w = 3 file disagrees with v2's file (`day2_final_ranker_none_v2.csv`) on **10.3%** of test Clients (w = 1: 10.2%), and with the w = 1 file on 4.1%.
- Label mix of `submissions/day2_final_target_weight.csv`: none 217, cloud 135, mobile 125, gym 116, insurance 113, software 104, streaming 99, music 91.
- `uv run rf submit --check` passed at 15:31:56 CEST.

**The upload rule, applied as written:**
1. w = 3 beats w = 1 on the pooled rehearsal: +0.0082 > 0. **Holds.**
2. The w = 3 file disagrees with v2's on ≥ 10% of test Clients: 10.3%. **Holds** (narrowly).
3. Ready and checked by 17:10: 15:31:56. **Holds.**

**Verdict:** the 17:30 upload is `submissions/day2_final_target_weight.csv`. Under the claims rule (0.03) it is a **tie** with configuration A.

**Implementation notes:**
- Duplicates are `dup1:<id>` / `dup2:<id>` with their transaction rows copied under the new ids. The stream memo is keyed on content including the id, so copies are detected separately; the Pseudo-Labels' ids (`<split>@<date>:`) cannot collide.
- v2's `none` model trains on a ranker `none` feature cross-fitted over 5 folds of Clients. Left as is, a copy could land in another fold than its original and see its own label. The ranker is therefore subclassed with the folds drawn over base Clients (a Client and its copies share a fold). With no copies the folds are identical to the original, which the w = 1 check confirms.
- E3 uses only the weight-1 out-of-fold probabilities: in the rehearsal those of train + the other half; in the final the hail mary's `artifacts/hailmary/final/oof_{v2,surv}.csv`, reused. No duplicate enters any cross-validation split.
- Selection labels: for fitting, read only inside `data.training_run(with_selection=True)`, the held-out half dropped on load (the stratified split itself reads all 700 labels once, inside a training run, as the spec's stratification requires); for scoring, once, in `data.scoring()`. The sealed holdout was never read.
- The final script is `scripts/day2_final_target_weight.sh`.
