# 06: E2 features and LightGBM

**What to build:** The team can train and evaluate a LightGBM model that learns the Next Recurring Family, and especially the `none` boundary, from stream-derived features. Each Client gets one feature row built only from the stream table (ADR 0001). The row covers the top three Active Streams, a block per Merchant Family, and the `none` signals.

**Blocked by:** 02, 03

**Status:** resolved

- [x] `train --model lgbm` builds one feature row per Client from the stream table only, including Clients with no streams
- [x] Features include the top three Active Stream slots (family, MCC, amount, period, regularity, days to next payment), a per-family block, and the `none` signals (longest active stream length, earliest active stream start, share of family-specific descriptions)
- [x] The feature schema is identical across train, valid, test and unlabeled; unseen descriptions encode as unknown
- [x] LightGBM multiclass with balanced class weights and default hyperparameters; out-of-fold probabilities are saved via the harness from 02
- [x] A fixture with easily separable families reaches macro-F1 ≥ 0.95
- [x] A fixture with a leakage trap fails unless any label-derived feature is fitted out-of-fold
- [x] Tested through the CLI against fixture data (Seam 1)

## Comments

- From ticket 08 (stream detection hardening): build stream-derived features from Active Streams or from streams with 3+ payments, not from raw stream counts. Raw counts (all streams, including 1–2 payment ones) shift between train and valid/test, because description noise on non-home MCCs creates short fragments at different rates per split. Per-stream `refund_rate` is now at most 1 (each refund reverses one payment of its own stream), and music and streaming streams at nearby amounts on MCC 5812 are no longer merged.
- Resolved. Merged into main as 596ebcc ("Merge ticket 06: e2-features-lightgbm"). Fast suite (`uv run pytest -m "not slow"`): 196 passed. Every acceptance box has a CLI-level test in `tests/test_lgbm_pipeline.py` (Seam 1).
- Experiment (run 20260924T143300-7c79fe, `rf train --model lgbm` then `rf evaluate --model lgbm --split selection`, argmax decision): macro-F1 0.4722 on the valid selection set (700 Clients), against 0.0566 for the prior baseline: delta +0.4155 (95% CI +0.3722 .. +0.4531), verdict improvement. Per-family F1: cloud 0.5714, gym 0.5059, insurance 0.4806, mobile 0.4878, music 0.3607, software 0.4065, streaming 0.3802, none 0.5842. The log row and per-Client predictions are committed in 87077ad (`experiments/log.csv`, `experiments/runs/20260924T143300-7c79fe.csv`).
