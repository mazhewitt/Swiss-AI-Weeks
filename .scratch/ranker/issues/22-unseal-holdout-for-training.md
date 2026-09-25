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

- [ ] The guard mode plus its test; `uv run pytest -q` passes
- [ ] `experiments/analysis/target_weight/` extended, or `experiments/analysis/unsealed/`, plus `scripts/day2_final_unsealed.sh`, which reproduces the file byte for byte
- [ ] The rule applied and recorded here
- [ ] A critic (leakage and regression); blocking findings fixed
