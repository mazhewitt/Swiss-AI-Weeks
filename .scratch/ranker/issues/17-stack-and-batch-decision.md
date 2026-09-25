# 17: Pre-registered stack and a batch expected-macro-F1 decision

Status: ready-for-human

**Why:** another team reached 0.68 on test. A research round (web sources plus train-only simulation) found:
- to reach 0.68 through `none` we would need a new `none` signal with AUC ≈ 0.97 stacked on the race: a near-oracle;
- no behavioural feature does that, and neither does anything else we have;
- the realistic best is 0.62–0.635 on test.

This ticket runs the two cheapest candidates for the 17:30 upload.

**Upload versus claim (agreed with the user):**
- **Claims:** the 0.03 paired-bootstrap tie rule stays for any "better" claim in the write-up.
- **Uploads:** only the best upload counts, and v2's 0.606 is locked in. The upload goes to the candidate with the best chance of beating v2, by the rule below.

## 1. The stack (S)

- P = ½ P(v2) + ½ P(soft Survival Race: `SurvivalModel(order="monthly-slot", soft=True)`, the hard race's fit, as in `scripts/day2_final_survival_soft.sh`).
- The decision layer is tuned E3, fitted on the 5-fold out-of-fold probabilities of the same average. This is hail-mary configuration A with the hard race replaced by the soft one: the same code path (`experiments/analysis/hailmary/hailmary.py`), folds and seeds.
- Nothing is fitted on the 700 selection Clients: no weight search, no EM, no self-training.
- Rehearsal: fit on train, predict the 700 selection Clients. Final: fit on train plus selection, predict test.

## 2. The batch expected-macro-F1 decision (D)

For a set of Client probabilities and any decision reached by E3's search over E3's grid (the same parameters, grid and coordinate ascent as `decision_layer`), compute the **expected** macro-F1 on that batch:
- the probabilities stand in as soft labels;
- for each label c, E[TP_c] = Σ over Clients predicted c of p_ic, and E[true_c] = Σ_i p_ic;
- F1_c = 2·E[TP_c] / (#predicted c + E[true_c]);
- take the mean over the 8 labels.

Pick the grid decision with the highest expected macro-F1. No labels are used.

- **Gate A (train labels only; run first):**
  - on S's train out-of-fold probabilities, per outer fold, pick D on the held-out fold's own probabilities and score it on that fold's labels; pool the folds;
  - compare with nested E3 on the same folds;
  - report the reliability of P(none) by decile.
  - **Pass if realised D ≥ nested E3 − 0.01.** If it fails, D is not built further and is not uploaded.
- **Rehearsal (if Gate A passes):** pick D on the 700 selection Clients' S probabilities, without labels. Freeze D's and E3's predictions before any selection label is read.
- **Final:** pick D on the 1,000 test Clients' S probabilities.

## Scoring (selection labels are read only by the final refit, as training labels, and once in `data.scoring()` after every selection prediction is frozen)

- For S, S+D, and the existing pool: v2 0.5871 (run e309a3), the soft race 0.5987 (run 5456a7), and hail mary A 0.5983:
  - macro-F1;
  - paired bootstrap against v2: 2,000 resamples, seed 0, the same resamples as `submissions/day2_final_compare_soft.md`;
  - the paired standard error.
- Post hoc, as explanation only: the calibration of S's P(none) on selection (reliability by decile, Platt slope).

## Upload rule (fixed before any number above)

1. **The pool:** candidates with a positive selection point estimate against v2, and whose test file disagrees with v2's on at least 10% of test Clients.
2. **Ranking:** rank the pool by the shrunk delta d̃ = d − 1.7·SE_paired, where SE_paired is the standard deviation (ddof=1) of the 2,000 paired bootstrap deltas. This is the winner's-curse correction for about 15 things tried.
3. **The 17:30 upload** is the top-ranked candidate. If the pool is empty, the committed hail-mary file stays.
4. **Claims** follow the 0.03 rule only. None of these is expected to pass it.

## Acceptance

- [x] Stages in `experiments/analysis/stack17/` (or as extensions of `hailmary.py`), with results in `results.json`.
- [x] Gate A result, the rehearsal table and the rule applied as written, all recorded in this ticket.
- [x] If the pool's top is new (it is not: nothing to write): `submissions/day2_final_<name>.csv` plus `scripts/day2_final_<name>.sh`, validated with `rf submit --check`.
- [x] Three critics (spec, test validity, leakage and regression); blocking findings fixed (none were blocking).
- [x] The sealed holdout is never read.

## Gate A (train out-of-fold, S's probabilities; train labels only)

Per outer fold (the `rf cv` folds), D is picked on the held-out fold's own S probabilities and scored on that
fold's labels; nested E3 is fitted on the other four folds' out-of-fold probabilities. Predictions pooled over
the 2,000 train Clients.

| Fold | n | Nested E3 | D | D's expected macro-F1 |
|---|---|---|---|---|
| 0 | 400 | 0.6287 | 0.6375 | 0.6027 |
| 1 | 400 | 0.6539 | 0.6475 | 0.5976 |
| 2 | 400 | 0.6111 | 0.6371 | 0.5893 |
| 3 | 400 | 0.6369 | 0.6389 | 0.6049 |
| 4 | 400 | 0.6565 | 0.6635 | 0.5985 |
| **Pooled** | 2,000 | **0.6384** | **0.6454** | |

**Pass:** D − nested E3 = +0.0070 ≥ −0.01. `none` share: true 0.2985, nested E3 0.271, D 0.240.

Reliability of S's out-of-fold P(none) by decile (mean P(none) → observed `none` rate): 0.031→0.000,
0.057→0.000, 0.085→0.045, 0.117→0.075, 0.158→0.110, 0.214→0.180, 0.298→0.305, 0.469→0.585, 0.745→0.795,
0.927→0.890. Under-confident at the low end (it predicts some `none` where the bottom fifth has none at all), and
under-confident in deciles 8–9 too. D's expected macro-F1 (~0.60) sits well below its realised one (~0.645), so the soft labels are pessimistic
in level, but their ranking of decisions was good enough to pass.

## Rehearsal result (selection, 700 Clients; fitted on train only)

Checks (asserted in `score`): v2 alone and the soft race alone reproduce runs e309a3 and 5456a7 with 100%
agreement; the recomputed hail mary A reproduces its committed predictions (100%). In the final, the single-model
refits reproduce `day2_final_ranker_none_v2.csv` and `day2_final_survival_soft.csv` exactly.

Paired bootstrap against v2's run e309a3: 2,000 resamples, seed 0 (the resamples of
`submissions/day2_final_compare_soft.md`); SE is the standard deviation of the paired deltas over them. Test
disagreement is against `submissions/day2_final_ranker_none_v2.csv`.

| Candidate | Macro-F1 | d vs v2 (95%) | SE paired | d̃ = d − 1.7 SE | Test disagreement | Pool |
|---|---|---|---|---|---|---|
| v2 (run e309a3) | 0.5871 | — | — | — | — | no (reference) |
| Soft race (run 5456a7) | 0.5987 | +0.0116 (−0.0178 .. +0.0415) | 0.0152 | −0.0142 | 0.165 | yes |
| **Hail mary A** | **0.5983** | +0.0111 (−0.0099 .. +0.0327) | 0.0110 | **−0.0076** | 0.104 | **yes (top)** |
| S (E3) | 0.5923 | +0.0052 (−0.0183 .. +0.0292) | 0.0121 | −0.0154 | 0.099 | no (< 10%) |
| S+D | 0.5949 | +0.0078 (−0.0155 .. +0.0311) | 0.0120 | −0.0126 | 0.100 | yes |

Selection decisions: S's E3 (fitted on train out-of-fold) uses a `none` threshold of 0.52; D on the 700
selection Clients picked no threshold, `none` weight 0.84 and family weights 1.0–1.19 (expected macro-F1
0.6175 vs 0.6158 for argmax).

**The rule, applied as written:**
1. **Pool:** hail mary A, S+D and the soft race (positive d, test disagreement ≥ 10%). S is out: d > 0 but it
   disagrees with v2 on 99 of 1,000 test Clients (9.9%).
2. **Ranking by d̃:** hail mary A −0.0076, S+D −0.0126, the soft race −0.0142.
3. **The 17:30 upload is the committed hail-mary file,** `submissions/day2_final_hailmary.csv`. Nothing new is
   written.
4. **Claims:** every delta is under 0.03, so all are ties with v2.

**Reading:**
- **The soft race does not stack better than the hard race.** S (0.5923) is below both the soft race alone
  (0.5987) and hail mary A (0.5983). The soft race moves only the split among detected families, and v2 already
  covers most of that; averaging gives back part of the soft race's own gain.
- **D helps S a little, as Gate A said** (+0.0026 on selection, +0.0070 on train), by predicting less `none`
  than E3 at the level the batch's own probabilities expect. Still a tie.
- Post hoc, S's P(none) on selection: mean 0.305 against a `none` share of 0.293; Platt slope 0.53 (intercept
  −0.48). The slope is mostly an artefact: 31 Clients have P(none) exactly 1.0, and the 1e-6 clip puts them at
  logit 13.8, where they set the slope (critic's toy check: 0.60 at this clip, 1.02 without them). By decile, the
  top is over-confident (0.946 → 0.814) and the bottom three are under-confident (0.03–0.08 → 0.014). The level
  is right. Explanation only; nothing was fitted on it.

**Note on the hail-mary entry:** the spec lists hail mary A at 0.5983, so the pool uses A's selection
predictions. The committed hail-mary file, whose test disagreement is used, is the self-training configuration
(survival only, q 0.6, 0.5980 on selection) that ticket 14's rule chose, not A itself. The two are a tie
(−0.0003) and the rule's outcome is the committed file either way.

## Review

Three critics (spec, test validity, and leakage/regression). **None blocking.**

- **Spec:** every element conforms; the 10% cut is exact (99 and 100 of 1,000 Clients). None of the three stated deviations changes the file.
- **Leakage:**
  - the committed code reproduces all five frozen selection prediction files from train labels alone;
  - D on test reads no labels;
  - the pre-registered rule is unchanged between the pre-registration and the scoring commit.
- **Regression:**
  - `decision.fit()` is identical before and after the refactor, over 25 cases;
  - full pytest: 530 passed;
  - `rf submit --check` passes on the hail-mary file.
- **Fixed in this round:**
  - wording: E3's search, when selection labels are read, the SE definition, and the calibration reading (the Platt slope is driven by the P(none) = 1 Clients);
  - a hand-computed test with mixed predictions (it catches the one surviving formula mutant);
  - the final stage now asserts the single-model reproduction instead of only recording it.
- **Open (non-blocking):**
  - `data.valid_split` reads the valid label file, holdout rows included, without the guard, to stratify the split (older code). Only membership comes out; the cached `artifacts/valid_split.csv` could be used instead.
  - The hail-mary pool entry pairs A's selection score with the committed self-training file's test disagreement. They tie (−0.0003), and either reading keeps the same file.
