# 17: Pre-registered stack and a batch expected-macro-F1 decision

Status: ready-for-agent

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

For a set of Client probabilities and any decision in E3's grid (the same parameters and grid as `decision_layer`), compute the **expected** macro-F1 on that batch:
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

## Scoring (selection labels read once, in `data.scoring()`, after every prediction above is frozen)

- For S, S+D, and the existing pool: v2 0.5871 (run e309a3), the soft race 0.5987 (run 5456a7), and hail mary A 0.5983:
  - macro-F1;
  - paired bootstrap against v2: 2,000 resamples, seed 0, the same resamples as `submissions/day2_final_compare_soft.md`;
  - the paired standard error.
- Post hoc, as explanation only: the calibration of S's P(none) on selection (reliability by decile, Platt slope).

## Upload rule (fixed before any number above)

1. **The pool:** candidates with a positive selection point estimate against v2, and whose test file disagrees with v2's on at least 10% of test Clients.
2. **Ranking:** rank the pool by the shrunk delta d̃ = d − 1.7·SE_paired. This is the winner's-curse correction for about 15 things tried.
3. **The 17:30 upload** is the top-ranked candidate. If the pool is empty, the committed hail-mary file stays.
4. **Claims** follow the 0.03 rule only. None of these is expected to pass it.

## Acceptance

- [ ] Stages in `experiments/analysis/stack17/` (or as extensions of `hailmary.py`), with results in `results.json`.
- [ ] Gate A result, the rehearsal table and the rule applied as written, all recorded in this ticket.
- [ ] If the pool's top is new: `submissions/day2_final_<name>.csv` plus `scripts/day2_final_<name>.sh`, validated with `rf submit --check`.
- [ ] Three critics (spec, test validity, leakage and regression); blocking findings fixed.
- [ ] The sealed holdout is never read.
