# 05: Ranker trained with Pseudo-Labels

**What to build:** `train` and `cv` accept Pseudo-Label sources (splits and Shifted Cutoffs) and a weight (default 0.5). Pseudo-Labelled Clients are pooled with the real-labelled Clients as weighted training rows. Pseudo rows never appear in a validation fold or in the decision layer's fit, so out-of-fold scores and the tuned `none` threshold reflect real labels only. A train Client may appear both with its real label and as a Pseudo-Labelled Client at a Shifted Cutoff.

**Blocked by:** 03, 04

**Status:** ready-for-agent

- [ ] `train --model ranker` with Pseudo-Label sources fits, and records the sources, Shifted Cutoffs and weight in the model metadata
- [ ] Out-of-fold output contains only real-labelled Clients
- [ ] The decision layer is fitted on real-labelled out-of-fold rows only
- [ ] If the latest fidelity check failed, training still runs but the log row says so
- [ ] The log row records the sources, Shifted Cutoffs, weight and number of training Clients
- [ ] One run is logged on the selection set against the ranker trained on real labels only, with predictions committed
