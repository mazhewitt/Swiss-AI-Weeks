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

## Comments

- From ticket 08 (stream detection hardening): build stream-derived features from Active Streams or from streams with 3+ payments, not from raw stream counts. Raw counts (all streams, including 1–2 payment ones) shift between train and valid/test, because description noise on non-home MCCs creates short fragments at different rates per split. Per-stream `refund_rate` is now at most 1 (each refund reverses one payment of its own stream), and music and streaming streams at nearby amounts on MCC 5812 are no longer merged.
