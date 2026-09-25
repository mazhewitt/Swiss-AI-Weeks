# 24: A regime-aware file, built with everything we learned (after the competition)

Status: ready-for-agent

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
