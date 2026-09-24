# 03: Stream Ranker on real labels

**What to build:** A new `ranker` model that works through `train`, `cv`, `evaluate` and `submit`. Every detected Recurring Stream is a Candidate Stream (not only Active Streams due within the Horizon). A LightGBM binary model scores how likely each candidate is to carry the Client's Next Recurring Family. Candidate features: family, days to next payment, rank by that, payment count, days since first and last payment, period, gap regularity, amount stability, amount, refund rate, plus Client-level context (number of streams, number of Active Streams, longest Active Stream, earliest Active Stream start). No description-share or label-derived features (ADR 0001). Per-Client probabilities: each family's score is its highest candidate score, `none` is one minus the highest candidate score, and the row is normalised to sum to 1. The tuned decision layer (E3) works on top.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `train --model ranker` fits on train; `predict_proba` returns exactly the allowed labels, rows sum to 1, one row per requested Client
- [x] A Client with no Candidate Streams gets a `none`-dominated row, not an error
- [x] Two streams in the same family both feed that family's probability
- [x] `cv --model ranker` folds by Client, so no Client's candidates straddle folds
- [x] `train --model ranker --decision tuned` fits the decision layer on out-of-fold probabilities
- [x] Save then load gives identical probabilities
- [x] `submit --model ranker` writes a submission that passes validation
- [x] A slow, marked real-data test trains and scores within the evaluation-time budget
- [x] One run is logged on the selection set (paired bootstrap against the previous best), with predictions committed

## Resolution

- `src/recurring_family/ranker.py`: `RankerModel` (registered as `ranker`), `candidates` (one row per Candidate Stream, stream-table features only) and `client_proba` (family = best candidate score, `none` = 1 - best score, normalised). Streams are detected once per Client history in-process, so the tuned decision's 5-fold CV does not re-detect; LightGBM runs single-threaded (the default thread count made each fit 30x slower here).
- Tests: `tests/test_ranker.py` (fast, Seam 1 plus the probability rule), `tests/test_ranker_real_data.py` (slow: `train --decision tuned` + `evaluate` on the selection set in about 60 s against a 120 s budget).
- Selection run `20260924T154434-ee87f3` (`ranker+tuned`): macro-F1 0.5714. Logged comparison on this branch: +0.0993 over E2 (improvement). Against ticket 01's rule runs (branch `ranker/01-rules-baseline-none-gate`, not yet merged here), same 700 Clients: +0.0350 over E1 rules 0.5365 (95% +0.0011 .. +0.0676, improvement); +0.0224 over rules+gate4 0.5490 (95% -0.0111 .. +0.0565, tie).
- Train out-of-fold exploration (no selection-set reads): LightGBM parameter variants were flat (0.627-0.636 tuned). Extra stream-table features outside this ticket's list (days since last payment over period, rank among Active Streams, days behind the soonest Active Stream, streams in the same family) lifted OOF to 0.5778 argmax / 0.6495 tuned; left out to keep to the spec's feature list, and a candidate for the self-improvement loop.
