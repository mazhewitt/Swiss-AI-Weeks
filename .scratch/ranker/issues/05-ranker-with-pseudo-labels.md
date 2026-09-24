# 05: Ranker trained with Pseudo-Labels

**What to build:** `train` and `cv` accept Pseudo-Label sources (splits and Shifted Cutoffs) and a weight (default 0.5). Pseudo-Labelled Clients are pooled with the real-labelled Clients as weighted training rows. Pseudo rows never appear in a validation fold or in the decision layer's fit, so out-of-fold scores and the tuned `none` threshold reflect real labels only. A train Client may appear both with its real label and as a Pseudo-Labelled Client at a Shifted Cutoff.

**Blocked by:** 03, 04

**Status:** done

- [x] `train --model ranker` with Pseudo-Label sources fits, and records the sources, Shifted Cutoffs and weight in the model metadata
- [x] Out-of-fold output contains only real-labelled Clients
- [x] The decision layer is fitted on real-labelled out-of-fold rows only
- [x] If the latest fidelity check failed, training still runs but the log row says so
- [x] The log row records the sources, Shifted Cutoffs, weight and number of training Clients
- [x] One run is logged on the selection set against the ranker trained on real labels only, with predictions committed

## Resolution

- `train`/`cv --model ranker --pseudo SPLIT[:YYYY-MM-DD]` (repeatable), `--pseudo-weight` (default 0.5), `--pseudo-min-payments` (default 4). Pseudo rows join every fold's fit but are never scored; the decision layer is fitted on real-labelled out-of-fold rows only. Tests: `tests/test_ranker_pseudo_labels.py`.
- Model metadata (`artifacts/ranker.meta.json`) and the evaluate log row record sources, Shifted Cutoffs, weight, min_payments, training Clients and the latest fidelity check (`fail 20260924T162515-d6d36f` today).
- Fixed on the way: a stream table whose streams all have one payment (possible at a Shifted Cutoff) failed to build.
- Selection run 20260924T171108-3251a4 (`ranker+pseudo+tuned`, train + unlabeled at 2025-10-03): 0.5735, a tie with the real-label ranker+tuned 20260924T154434-ee87f3 (0.5714), delta +0.0021 (95% -0.0272 .. +0.0306).
