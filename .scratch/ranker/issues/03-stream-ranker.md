# 03: Stream Ranker on real labels

**What to build:** A new `ranker` model that works through `train`, `cv`, `evaluate` and `submit`. Every detected Recurring Stream is a Candidate Stream (not only Active Streams due within the Horizon). A LightGBM binary model scores how likely each candidate is to carry the Client's Next Recurring Family. Candidate features: family, days to next payment, rank by that, payment count, days since first and last payment, period, gap regularity, amount stability, amount, refund rate, plus Client-level context (number of streams, number of Active Streams, longest Active Stream, earliest Active Stream start). No description-share or label-derived features (ADR 0001). Per-Client probabilities: each family's score is its highest candidate score, `none` is one minus the highest candidate score, and the row is normalised to sum to 1. The tuned decision layer (E3) works on top.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `train --model ranker` fits on train; `predict_proba` returns exactly the allowed labels, rows sum to 1, one row per requested Client
- [ ] A Client with no Candidate Streams gets a `none`-dominated row, not an error
- [ ] Two streams in the same family both feed that family's probability
- [ ] `cv --model ranker` folds by Client, so no Client's candidates straddle folds
- [ ] `train --model ranker --decision tuned` fits the decision layer on out-of-fold probabilities
- [ ] Save then load gives identical probabilities
- [ ] `submit --model ranker` writes a submission that passes validation
- [ ] A slow, marked real-data test trains and scores within the evaluation-time budget
- [ ] One run is logged on the selection set (paired bootstrap against the previous best), with predictions committed
