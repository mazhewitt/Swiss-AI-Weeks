# 19: Build — a target-domain `none` stacker on the hail-mary average

Status: ready-for-agent

**Why:** ticket 18's gate passed. On the 700 selection Clients, raw transaction features predict `none` with AUC 0.807, and adding them to the stream features lifts the `none` AUC from 0.821 to 0.873 (paired DeLong +0.052, p = 0.0002). The lead feature is the Decoy share of card payments. Train has almost no Decoys, so only valid labels can teach it, and it drifts from selection to test (mean 0.27 against 0.39). **Deadline: final file by 16:45 CEST.**

## Design (pre-registered; no other variants)

- **Base probabilities: B = hail mary A**, the ½ v2 + ½ hard Survival Race average, fitted on **train only**. Its input to the stacker then has the same distribution on selection and on test.
  - Selection: `artifacts/hailmary/rehearsal/target_{v2,surv}.csv`.
  - Test: the same train-only models, predicting the test Clients. This is a new fit on train only, with the same code, seeds and folds as `hailmary.py`'s rehearsal fits.
- **Raw features:** ticket 18's raw block (`experiments/analysis/target_none/features.py`, the same 130 features), each converted to its **percentile rank within its own domain**: within the 700 selection Clients for selection, within the 1,000 test Clients for test. This removes level shifts such as the Decoy share's.
- **The stacker:** LightGBM binary `none` model on [logit P_B(none)] plus the 130 ranked raw features. It uses ticket 18's fixed parameters, with `n_jobs=1` and no tuning.
- **The adjusted probabilities:** P'(none) = the stacker's output, and each family's probability is P_B(family) × (1 − P'(none)) / (1 − P_B(none)). Guard the division: if P_B(none) = 1, split the family mass equally.
- **The decision:** D, the batch expected-macro-F1 decision (ticket 17, `decision.fit_expected`), picked label-free on the batch it predicts.

## Rehearsal (selection labels, nested; no label read outside the CV folds)

- Repeated 5×5 stratified CV over the 700 selection Clients, seeds 0–4. For each outer fold:
  - fit the stacker on the other four folds' Clients;
  - predict the held-out Clients' P'(none).
- Average P'(none) over the repeats, giving out-of-fold P' for all 700.
- **Candidates**, each scored once after every prediction is frozen:
  - **B+E3:** hail mary A as it is (0.5983; a check);
  - **B+D:** B with the batch decision, label-free;
  - **N:** the stack, P' with D picked on all 700 out-of-fold P' (label-free).
- **Scoring:** macro-F1; paired bootstrap against v2 (2,000 resamples, seed 0) and against B+E3; the paired SE.
- **Optimism:** the same 700 labels chose this feature block (ticket 18), so N's score is optimistic. Record this in the results.

## Final

- Fit the stacker on all 700 selection Clients.
- Apply it to the test Clients: the train-only B on test, plus the test-ranked raw features.
- Pick D on the 1,000 test Clients' P' (label-free).
- Write `submissions/day2_final_none_stack.csv`, with a script `scripts/day2_final_none_stack.sh`.

## Upload rule (fixed before any number)

- The ticket-17 pool rule, unchanged: a positive d against v2, at least 10% test disagreement with v2's file, ranked by d̃ = d − 1.7·SE.
- The pool is extended with N and B+D.
- The top-ranked candidate is the 17:30 upload. The claims rule (0.03) is unchanged.

## Acceptance

- [ ] `experiments/analysis/none_stack/` (script, frozen predictions, `results.json`)
- [ ] The rehearsal table and the rule applied, recorded here
- [ ] The final file plus its script, validated with `rf submit --check`
- [ ] Three critics; blocking findings fixed
- [ ] The sealed holdout is never read
