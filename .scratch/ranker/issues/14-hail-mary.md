# 14: Hail mary — average, label-shift EM and self-training on the target Clients

Status: done

**Why:** the last upload slot (17:30). v2 locks in 0.6059 on the leaderboard, and only the best upload counts, so the last slot should go to a candidate that differs a lot from v2, for a real reason. Train gains have shrunk on valid and test every time (the `none` model, the Survival Race). That points to a shift between train and valid/test. Two standard answers to a domain shift use the target Clients' transactions, never their labels:

- **Label-shift EM** (Saerens, Latinne and Decaestecker, 2002): re-estimate the label mix on the target Clients from the model's own probabilities, and reweight each Client's probabilities by target prior / train prior.
- **Self-training:** label the target Clients the model is most sure of with its own predictions, refit with them, and predict again.

Both run on top of the **average** of v2's and the Survival Race's probabilities. The two models disagree on 16–17% of Clients, and each is best on different families.

**Rehearsal (valid is a shifted domain too):**
- Fit on train only. The target is the 700 selection Clients' transactions. The 300 sealed-holdout Clients' transactions are dropped on load and never used.
- Score each configuration on the selection labels, read only after every prediction is made.

**Final:** the same code, fitted on train plus selection. The target is the 1,000 test Clients.

## Configurations (pre-registered; no others)

- **A. Average:** P = ½ P(v2) + ½ P(Survival Race, `unprojected-last`).
  - The decision layer (tuned E3) is fitted on the 5-fold out-of-fold probabilities of the same average: train in the rehearsal, train plus selection in the final.
- **A+EM:** A, with label-shift EM on the target Clients.
  - Source prior: the mean out-of-fold probability.
  - Iterate until the prior moves by less than 1e-6, at most 1,000 times.
- **Self-training (6):**
  - Take the target Clients' labels from A (or from A+EM, if EM passes its rule below). Confidence is the probability of the decided label.
  - The top q of target Clients by confidence become labelled Clients (weight 1): q ∈ {0.3, 0.6, 1.0}.
  - Refit either the Survival Race only (v2 kept as it is) or both models.
  - **Cross-fitted over the target:** it is split into two fixed halves (seed 0). Each half is predicted by models whose self-labels come from the other half only, so no Client's own self-label reaches its prediction.
  - The average, EM (if used) and the decision layer are then as in A.

## Upload rule (fixed before any hail-mary number)

1. EM is used from here on only if A+EM beats A on selection by at least +0.01.
2. The hail-mary file is the self-training configuration with the best selection macro-F1, if that score is at least 0.5771 (v2's 0.5871 − 0.01, i.e. not clearly worse).
3. If none qualifies, the file is A (with EM if rule 1 passed), if A scores at least 0.5871 on selection.
4. If neither qualifies, report it, and the user decides.

Choosing the best of 6 on selection flatters that one score, so it is reported as optimistic. The sealed holdout is not read (standing rule).

## Checks

- The rehearsal's single-model probabilities should reproduce v2's selection run `20260925T001755-e309a3` (0.5871) and the Survival Race's `20260925T080905-fd1dc5` (0.5903). They are the same code, folds and seeds, so the match should be exact, and the score stage asserts it.
- Label guards: fits run inside `data.training_run`, predictions inside `data.predicting`, and selection labels are read only inside `data.scoring`. Self-labels come from predictions only.
- Paired bootstrap against v2's selection predictions.

## Acceptance

- [x] `experiments/analysis/hailmary/hailmary.py` (stages cached under `artifacts/hailmary/`) and `results.json`
- [x] Rehearsal table in this ticket; the rule applied as written
- [x] `submissions/day2_final_hailmary.csv` plus `scripts/day2_final_hailmary.sh`, validated
- [x] Three critics (spec, test validity, leakage and regression); blocking findings fixed

## Rehearsal result (selection, 700 Clients; fitted on train only)

| Configuration | Macro-F1 | vs v2 (paired bootstrap, 95%) | Agrees with v2 |
|---|---|---|---|
| v2 alone (check) | 0.5871 | = run e309a3, 100% agreement | 1.000 |
| Survival Race alone (check) | 0.5903 | = run fd1dc5, 100% agreement | 0.829 |
| **A: average** | **0.5983** | +0.0111 (−0.0099 .. +0.0327) | 0.891 |
| A+EM | 0.5991 | +0.0119 (−0.0114 .. +0.0359) | 0.874 |
| Self-train survival, q 0.3 | 0.5911 | +0.0040 | 0.890 |
| **Self-train survival, q 0.6 (chosen)** | **0.5980** | +0.0109 (−0.0101 .. +0.0325) | 0.891 |
| Self-train survival, q 1.0 | 0.5917 | +0.0045 | 0.884 |
| Self-train both, q 0.3 | 0.5912 | +0.0041 | 0.891 |
| Self-train both, q 0.6 | 0.5906 | +0.0035 | 0.883 |
| Self-train both, q 1.0 | 0.5928 | +0.0057 | 0.876 |

**The rule, applied as written:**
- **Rule 1:** EM adds +0.0008, below +0.01, so EM is off.
- **Rule 2:** the best self-training configuration is survival-only, q 0.6, at 0.5980 ≥ 0.5771, so it is the file.
- It is −0.0003 against the plain average (95% −0.0081 .. +0.0074): the same model in effect. The critics noted that rule 2 can choose a configuration scoring below A, and here it does, by a tie.

**Reading:**
- **The gain is the average,** +0.011 over v2 on selection, a tie under our 0.03 rule.
- **Neither transductive method helps.** EM ties, and self-training ties or loses (5 of 6 below A). The shift between train and valid is not mainly a label-mix shift, and the models' own confident labels on the target teach them nothing new.
- 0.5980 is the best of 6 configurations on selection, so it is optimistic. Rule 1 is also a choice made on selection.

**Final file:** `submissions/day2_final_hailmary.csv` (valid). It is fitted on train plus selection, with test self-labels cross-fitted over two halves (296 and 304 self-labels).
- It agrees with v2's file on 89.6% of test Clients and with the Survival Race's on 91.6%.
- Label mix: none 214, cloud 128, mobile 126, gym 117, insurance 114, software 104, streaming 102, music 95.

## Review

Three critics reviewed the code (spec, test validity, and leakage/regression), reading it without running any fits.

- **Leakage:** none found. Rehearsal fits read train labels only. Selection labels are read once, in `data.scoring()`. No sealed-holdout history is used. The `self:` ids cannot collide with real or Pseudo-Label ids, and the stream cache is keyed on content.
- **Blocking, both fixed before the final rescore:**
  1. Rules 1–3 compared rounded floats, so a gain of exactly +0.01 could fail. They now compare raw scores with an epsilon.
  2. `score` could write a chosen configuration from an incomplete set of self-training runs. It now writes `complete: false` and no choice, and `submit` refuses to run.
- **Follow-ups done:**
  - every prediction, under both EM tags, is made before the labels are read;
  - the single-model checks assert 100% agreement with their logged runs;
  - NaN and length guards;
  - no spaces in file names;
  - a paired bootstrap of the choice against A.
- **Follow-ups open (non-blocking):**
  - EM's target mix and the top-q cut-off use the whole target, own half included. This is population-level only and involves no labels.
  - With EM on, self-trained models would be reweighted by the base models' source prior.
  - At q 0.6 the final's self-labels are 600 of 1,000 test Clients (about 300 per fit, beside 2,700 real ones), against 420 of 700 (about 210 per fit, beside 2,000) in the rehearsal: a similar, not identical, share.
  - The `both` refits repeat the survival-only refits.
  - No unit tests for `label_shift_em`, `self_labels` and `halves`; the validity critic checked them on toy data.
