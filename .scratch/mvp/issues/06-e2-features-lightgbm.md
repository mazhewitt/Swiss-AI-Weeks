# 06: E2 features and LightGBM

**What to build:** The team can train and evaluate a LightGBM model that learns the Next Recurring Family, and especially the `none` boundary, from stream-derived features. Each Client gets one feature row built only from the stream table (ADR 0001). The row covers the top three Active Streams, a block per Merchant Family, and the `none` signals.

**Blocked by:** 02, 03

**Status:** ready-for-agent

- [ ] `train --model lgbm` builds one feature row per Client from the stream table only, including Clients with no streams
- [ ] Features include the top three Active Stream slots (family, MCC, amount, period, regularity, days to next payment), a per-family block, and the `none` signals (longest active stream length, earliest active stream start, share of family-specific descriptions)
- [ ] The feature schema is identical across train, valid, test and unlabeled; unseen descriptions encode as unknown
- [ ] LightGBM multiclass with balanced class weights and default hyperparameters; out-of-fold probabilities are saved via the harness from 02
- [ ] A fixture with easily separable families reaches macro-F1 ≥ 0.95
- [ ] A fixture with a leakage trap fails unless any label-derived feature is fitted out-of-fold
- [ ] Tested through the CLI against fixture data (Seam 1)
