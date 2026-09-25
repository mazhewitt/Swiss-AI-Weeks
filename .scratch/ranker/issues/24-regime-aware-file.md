# 24: A regime-aware file, built with everything we learned (after the competition)

Status: done

**Why:** the user wants one more test file, built with everything we learned, to see how far the approach goes. It cannot be used in the competition. The post-mortem (ticket 23) showed:
- valid/test descriptions and MCCs are noisy (53% and 15% of test's schedule payments);
- the noise breaks our streams apart;
- corrupting train to test's noise level reproduces our train→test gap.

## Candidates (fixed before any number)

All candidates are hail mary configuration A (v2 plus the hard race; E3 fitted on the weight-1 out-of-fold probabilities), **with the valid Clients at weight 3** (ticket 21's method, including `GroupedRanker`).

- **B (baseline):** exactly ticket 21. Its rehearsal score is 0.6054.
- **R1, train in the test regime:**
  - train's transactions **and** the unlabeled split's (both are clean-regime) are corrupted to test's measured noise;
  - ticket 23's `noise_params.json`, dose 1×, seed 0, the same per-payment draws;
  - valid is left as it is (already noisy);
  - Pseudo-Labels are rebuilt from the corrupted data.
- **R2, a noise-robust stream detector:** an **opt-in** option in `src/recurring_family/streams.py`, off by default, with unit tests. Existing behaviour must be byte-identical when it is off.
  - Candidate streams come from **amount and periodicity alone**, ignoring the description: the detector's existing amount and schedule tolerances, over subscription-like card payments, **including Filler Descriptions**.
  - A stream's family is the **majority vote** of its payments' evidence: a family word counts 1, the family's home MCC counts 0.5, and Filler or Decoy counts 0.
  - A stream with no family evidence is dropped.
  - Decoy-described payments may join only on schedule, as with `join_decoys`.
- **R1+R2:** both together.

## Rehearsal (as ticket 21: two halves of the 700 selection Clients)

- For each half, fit on train (as the candidate transforms it) plus the other half at weight 3, and predict this half.
- Freeze all predictions before scoring. Score each candidate with pooled macro-F1 and a paired bootstrap against B.
- **Choosing is best of 4 on the same 700 labels, so the chosen score is optimistic.** Report it that way.

## Final file

- The best candidate on the rehearsal (B if none beats it), fitted on train (transformed) plus **all 1,000** valid Clients at weight 3 (ticket 22's unsealed domain), predicting test.
- Written to `submissions/day2_postmortem_regime.csv` with its script, validated with `rf submit --check`.
- Not uploaded to the competition.
- Also report its agreement with the uploaded file (`day2_final_unsealed.csv`, test 0.6066) and its label mix.

## Rules

- Test labels do not exist to us.
- The only labels read are train's plus valid's, through the guard modes (including ticket 22's unsealed mode for the final).
- Code goes in `experiments/analysis/regime_aware/`.

## Result

Code and numbers are in `experiments/analysis/regime_aware/`: `regime_aware.py`, the frozen rehearsal predictions in `predictions/`, and `results.json`. The robust detector is `StreamParams(robust=True)` in `src/recurring_family/streams.py`, with tests in `tests/test_robust_streams.py`. The final file comes from `scripts/day2_postmortem_regime.sh`.

**Rehearsal** (the 700 selection Clients in ticket 21's two halves; each half predicted by a fit on train, as the candidate transforms it, plus the other half at weight 3). All four prediction sets were frozen and committed (e6e54f1, 4ce9c48) before one scoring pass in `data.scoring()`. The paired bootstrap is against B, with a 95% interval.

| Candidate | Pooled macro-F1 | vs B (95%) | Agrees with B | `none` predicted |
|---|---|---|---|---|
| **B** (ticket 21) | **0.6054** | — | — | 169 |
| R1: train in the test regime | 0.5880 | −0.017 (−0.036 .. −0.001) | 0.923 | 152 |
| R2: robust detector | 0.5995 | −0.006 (−0.036 .. +0.024) | 0.779 | 158 |
| R1+R2 | 0.5962 | −0.009 (−0.040 .. +0.021) | 0.777 | 188 |

- **No candidate beat B, so the choice is B.** Taking the best of 4 on the same 700 labels would make a winner's score optimistic. Here the winner is the baseline, so no optimism was selected in.
- **R1 is worse than B, and the interval just excludes zero.** Training on train corrupted to test's noise gives up clean signal and does not make up for it on valid-regime Clients.
  - Valid is already noisy and sits in the fit at weight 3, so the model already sees the test regime.
  - Train's Pseudo-Label `none` share rises from 0.145 (clean) to 0.183.
- **R2 is level with B within noise, but not better.** It changes 22% of labels.
  - It helps music (+0.026), software (+0.032) and `none` (+0.015).
  - It loses on cloud (−0.046) and insurance (−0.034).
  - The robust detector does recover valid and test streams: Active Streams per Client go from 1.38 to 1.63 on valid, and from 1.37 to 1.61 on test, near train's 1.61.
  - But it also turns many lone subscription-like payments into single-payment streams (56% of streams on valid, against 30% by default). The ranker and the race then have more low-value candidates to rank.
- R1+R2 does not stack.

**Final file:** `submissions/day2_postmortem_regime.csv`, B fitted on train plus all 1,000 valid Clients at weight 3 (ticket 22's unsealed domain, `training_run(with_selection=True, with_holdout=True)`), with E3 on the weight-1 out-of-fold probabilities of the 3,000 labelled Clients.
- Label mix: `none` 265, cloud 122, mobile 122, gym 108, insurance 104, streaming 98, software 95, music 86.
- **It agrees with the uploaded `day2_final_unsealed.csv` (test 0.6066) on 100% of Clients; the two files are byte-identical.** This is expected: B's final fit is ticket 22's fit. It was rerun here independently, and all four probability files match ticket 22's byte for byte.
- `rf submit --check` passes. A second run of the script reproduced the file byte for byte. The file was not uploaded.

**Checks:**
- R1's corrupted train equals ticket 23's dose-1 table (content hash `a7708bc9684a`). Unlabeled gets the same procedure with its own seed-0 generator (92,025 descriptions and 22,206 MCCs replaced).
- Pseudo-Labels are rebuilt from the DataFrames, bypassing the caches keyed on size and mtime. For B's settings, the bypass reproduces the rows `rf` pools, exactly.
- The robust detector is off by default. With it off:
  - the stream tables and member payments of train, valid, test and unlabeled, at the Cutoff and the Shifted Cutoff, hash identically to the previous code;
  - the hail mary's rehearsal v2 and race target probabilities are reproduced byte for byte.
- The option is threaded through `RankerModel`, `SurvivalModel` (including the `none` model's features and cross-fit, and save/load) and this ticket's factories. Their defaults are unchanged.
- Full suite: 561 passed.

**Interpretation decisions (not in the spec):**
- R2 uses the robust detector everywhere a stream table is made: the fit, the target, the `none` model and the Pseudo-Labels.
- The vote gives no weight to a description that names two groups.
- Ties go to the group first by name, as the detector's ambiguous vote does.
- A single-payment cluster with any evidence (even a home MCC's 0.5) is kept as a stream, as the default keeps a lone anchor.
- A Decoy joins after the vote and adds no evidence.
- Periodicity only splits an amount cluster that looks biweekly into two parts that each look monthly (canonical periods). A stricter per-gap schedule test was ruled out: on train, 32% of consecutive stream gaps are more than 3 days off the period.

**Caveats:**
- There is one corruption seed.
- The rehearsal has 700 Clients, and every R2 interval spans zero.
