# 22: Train the final file on all 1,000 valid Clients (the holdout unsealed)

Status: ready-for-agent

**Why:** the user decided at 15:40 CEST to unseal the 300-Client holdout for training. Its one job, checking v2 (0.596), is done. Every final file so far has been fitted on train plus 700 selection Clients. The only signal specific to valid/test can be learned only from valid labels, so 300 more of them (+43%) is the last lever left.

**What the unsealing costs:** after this, no held-out valid data is left. Nothing fitted with the holdout can be validated. This ticket therefore changes only the **data** of ticket 21's final fit, not its method, which ticket 21's cross-fitted rehearsal validated.

## Design (pre-registered)

- **Exactly ticket 21's final fit**, with the fit set = train plus **all 1,000** valid Clients (selection plus holdout) instead of train plus 700:
  - hail mary configuration A (v2 plus the hard race, the same factories, Pseudo-Labels and seeds);
  - the valid Clients at weight 3, as two duplicates each (`GroupedRanker` for v2's internal `none` cross-fit);
  - E3 fitted on the **weight-1** 5-fold out-of-fold probabilities of train plus all 1,000 valid Clients (3,000 Clients, no duplicates in any split).
- **Label access:** the holdout labels are read through an explicit new guard mode (for example `data.training_run(with_selection=True, with_holdout=True)`), not by bypassing the guard. The mode is covered by a unit test. Existing modes behave exactly as before.

## Upload rule (fixed before any number)

The 17:30 upload becomes this file (`submissions/day2_final_unsealed.csv`) only if all of these hold:
1. it passes `rf submit --check` by **17:10 CEST**;
2. it disagrees with v2's file on at least 10% of test Clients (the pool rule);
3. **sanity:**
   - its test `none` count is within ±40 of ticket 21's file (217);
   - it agrees with ticket 21's file on at least 85% of test Clients.

   These bounds guard against a broken fit, since there is no score to check.

Otherwise, ticket 21's file stays. There is no claim, because there is no held-out score. The write-up reports it as "fitted on all labelled valid Clients, unvalidated beyond ticket 21's rehearsal".

## Acceptance

- [x] The guard mode plus its test; `uv run pytest -q` passes
- [x] `experiments/analysis/target_weight/` extended, or `experiments/analysis/unsealed/`, plus `scripts/day2_final_unsealed.sh`, which reproduces the file byte for byte
- [x] The rule applied and recorded here
- [ ] A critic (leakage and regression); blocking findings fixed

## Result (15:52 CEST, rule applied as written)

**Verdict: the rule fails on the `none` sanity bound, so ticket 21's file (`submissions/day2_final_target_weight.csv`) stays the 17:30 upload.** The unsealed file is written and valid, but it is not uploaded.

| Condition | Value | Bound | Holds |
|---|---|---|---|
| 1. passes `rf submit --check` by 17:10 CEST | checked 15:52:31 CEST | by 17:10 | yes |
| 2. disagreement with v2's file | 11.6% | at least 10% | yes |
| 3a. test `none` count | 265 (ticket 21: 217, +48) | 177 to 257 | **no** |
| 3b. agreement with ticket 21's file | 93.4% | at least 85% | yes |

Label mix of the unsealed file: none 265, cloud 122, mobile 122, gym 108, insurance 104, streaming 98, software 95, music 86.

Nothing was scored: no holdout label was read outside `data.training_run(with_selection=True, with_holdout=True)`. Numbers in `experiments/analysis/target_weight/unsealed/results.json`.

**Implementation.** `data.training_run(with_selection=True, with_holdout=True)` is the new guard mode (holdout and full valid labels readable in training; `with_holdout` without `with_selection` is a `ValueError`); the existing modes are unchanged and their refusals are re-tested in `tests/test_honest_evaluation.py`. `target_weight.py` gains the `unsealed` domain (train plus all 1,000 valid Clients; the valid Clients are the weighted ones), the `oof unsealed` and `fit unsealed 3` stages, and the `unsealed` and `unsealed-checked` stages that write the file and apply this rule. E3 is fitted on the average of the weight-1 5-fold out-of-fold probabilities of the 3,000 Clients. `scripts/day2_final_unsealed.sh` clears stale outputs, runs the four fits in parallel, waits on each pid, writes the file, runs `rf submit --check` and records the verdict. A second full run of the script (16:02 to 16:04 CEST) rewrote the file byte for byte (`cmp` identical, and the four probability files too); `results.json` differs only in its two timestamps. `uv run pytest -q`: 547 passed.

## Decision: the user overrode the sanity rule (16:10 CEST)

The pre-registered rule failed on one sanity bound: the file predicts 265 test `none` against a bound of 177–257. The bound was a guard against a broken fit, since this file cannot be scored, and 265 (26.5%) is closer to the true `none` share on selection (29.3%) than ticket 21's 217. The other checks all held:
- it agrees with ticket 21's file on 93.4% of test Clients;
- it disagrees with v2 on 11.6%;
- the families shrink evenly.

The user chose to override the bound, so **`submissions/day2_final_unsealed.csv` is the 17:30 upload**, provided the leakage critic finds nothing blocking. This is recorded as an override of our own pre-registered rule. The file is unvalidated beyond ticket 21's rehearsal of the method.

## Review

One critic (leakage and regression). **None blocking.**
- **The guard mode:**
  - only `training_run(with_selection=True, with_holdout=True)` opens the holdout, and only for training;
  - `with_holdout` alone raises;
  - the permission ends with its block;
  - scoring and checkpoint are unchanged;
  - 56 guard tests pass.
- **The `unsealed` domain:**
  - the out-of-fold files have 3,000 unique ids and no duplicates;
  - `GroupedRanker` keeps copies in their original's fold;
  - test Clients contribute transactions only;
  - no holdout label scores anything.
- **Regression:** the h0, h1 and final paths are unchanged, ticket 21's outputs are untouched, and both files pass `rf submit --check`.
- **Diagnosis of the +48 `none`:** it comes from the **decision layer**, not the models.
  - Mean test P(none) barely moves: 0.324 → 0.328.
  - E3 refitted with the 300 extra labels favours `none` by about 2–3 grid steps.
  - Crossing each fit's weights with the other's probabilities: the weights account for about +41 to +44 `none`, the models for +4 to +7.
- **Open (non-blocking):**
  - no test for `pseudo_labelling()` nested inside the unsealed mode (safe: that check runs first);
  - `unsealed-checked` relies on the script's `set -e` for the submit check.

## Test leaderboard

**0.60659**, against v2's 0.6059 (+0.0007): a tie. The file disagrees with v2 on 11.6% of test Clients, yet the score barely moved: the changes it made cancel out on test. Final best upload: 0.6066.
