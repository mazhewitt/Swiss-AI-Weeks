# 19: Build — a target-domain `none` stacker on the hail-mary average

Status: done

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

- [x] `experiments/analysis/none_stack/` (script, frozen predictions, `results.json`)
- [x] The rehearsal table and the rule applied, recorded here
- [x] The final file plus its script, validated with `rf submit --check` (a clean end-to-end run of the script reproduced it byte for byte, sha256 d0ba3119…; `predictions/` and `results.json` unchanged)
- [x] Critics (two: spec and test validity merged, plus leakage and regression); none blocking
- [x] The sealed holdout is never read

## Result

Scripts, frozen predictions and numbers: `experiments/analysis/none_stack/` (`base_test.py`, `stacking.py`,
`none_stack.py`, `predictions/`, `results.json`); tests in `tests/test_none_stack.py`. Ran as pre-registered.

**Train-only B on test.** `base_test.py` fits v2 and the hard race on train only with `hailmary.py`'s own
factories and seeds, and predicts selection and test from the same fitted model. Its selection probabilities are
**byte-identical** to `artifacts/hailmary/rehearsal/target_{v2,surv}.csv`, and B+E3 reproduces hail mary A's
committed selection predictions (700/700) and ticket 17's row (0.5983, d +0.0111, SE 0.0110).

**Labels.** Selection labels were read once for the stacker (the binary `none` target), inside
`data.training_run(with_selection=True)`, for the 5x5 nested CV and the final fit; and once in `data.scoring()`,
after `predictions/` was committed. The sealed holdout was never read.

**Rehearsal** (700 selection Clients; paired bootstrap 2,000 resamples, seed 0; SE against v2):

| Candidate | macro-F1 | d vs v2 [95%] | d vs B+E3 [95%] | SE | d̃ = d − 1.7·SE | test disagreement with v2 | F1 `none` |
|---|---|---|---|---|---|---|---|
| B+E3 (hail mary A) | 0.5983 | +0.0111 [−0.0099, +0.0327] | — | 0.0110 | −0.0076 | 0.104 | 0.668 |
| B+D | 0.5956 | +0.0085 [−0.0131, +0.0301] | −0.0026 [−0.0149, +0.0097] | 0.0111 | −0.0103 | 0.102 | 0.661 |
| N | 0.5974 | +0.0103 [−0.0132, +0.0341] | −0.0008 [−0.0200, +0.0185] | 0.0120 | −0.0101 | 0.166 | 0.693 |

- The stacker lifts the out-of-fold `none` AUC from 0.878 (B) to 0.893 (P'), and N's `none` F1 from 0.668 to
  0.693, but the families lose about as much. Net: N is 0.0008 below B+E3.
- **Optimism:** the same 700 labels chose the raw block (ticket 18), so N's 0.5974 is optimistic; it does not
  beat B+E3 even so.
- P_B(none) = 1 on 31 selection and 54 test Clients (no detected streams); the guard splits their family mass
  equally, and all of them stay `none` under N.

**Upload rule.** The pool is ticket 17's (hail mary A, S+D, soft race, with its values) plus N and B+D (both
qualify: d > 0, disagreement ≥ 10%). By d̃: hail mary A −0.0076, N −0.0101, B+D −0.0103, S+D −0.0126, soft race
−0.0142. **The top is hail mary A: the 17:30 upload stays `submissions/day2_final_hailmary.csv`.**

**Test** (N's file, written anyway: `submissions/day2_final_none_stack.csv`, from
`scripts/day2_final_none_stack.sh`, `rf submit --check` valid). Label mix:

| File | cloud | gym | insurance | mobile | music | software | streaming | `none` |
|---|---|---|---|---|---|---|---|---|
| N (train-only B + stack, D on test) | 109 | 106 | 102 | 118 | 92 | 106 | 97 | **270** |
| B+D (train-only B, D on test) | 114 | 111 | 108 | 121 | 96 | 112 | 106 | 232 |
| B+E3 (train-only B) | 112 | 110 | 121 | 122 | 94 | 95 | 99 | 247 |
| hail mary file (committed, train+selection refit) | 128 | 117 | 114 | 126 | 95 | 104 | 102 | 214 |

- N's `none` share on test is 27.0%, against the hail mary's 21.4%. The selection `none` rate is 29.3%
  (205/700) and N predicts 173/700 (24.7%) there. Mean P'(none) is 0.280 on selection (out of fold) and 0.293 on
  test: the within-domain ranks remove the Decoy share's level shift, so there is no large `none` jump. Part of
  the gap to 214 is the train-only base (B+E3 on test already gives 247).

## Review

- **Spec and test validity (one critic):** none blocking. The design conforms to the pre-registration (d991616), the upload rule was applied with ticket 17's pool values and the same resamples, and there is no leakage path in the nested CV. Fixed:
  - the final script now builds the hail mary's git-ignored rehearsal caches first, so it runs from a fresh clone;
  - the stacker-input test now asserts that the raw block arrives ranked.
- **Open (non-blocking):**
  - **Extra optimism:** N's selection P' averages 5 models fitted on 560 Clients each, while test gets one model fitted on 700. This may flatter N's selection score slightly. It changes no choice, since N ranks below B+E3.
  - **Untested rebuild path:** if ticket 18's git-ignored raw caches are missing, `none_stack.py` rebuilds them from `hm.target` and the sample submission rather than ticket 18's inputs. By reading the code these give the same rows, and shape and column asserts guard them, but the path has not been run.
- **Leakage and regression:** see below.
