# 07: E2 + E3 milestone-2 path and evaluation time budget

**What to build:** The team can run the full milestone-2 candidate, LightGBM with the tuned decision layer. It is judged against an explicit acceptance gate, a submission model can be refit on train plus the selection set, and a full evaluation run fits the time budget that a later self-improvement loop needs.

**Blocked by:** 04, 05, 06

**Status:** wontfix (superseded by .scratch/ranker/issues/06)

- [ ] `train --model lgbm --decision tuned` works end to end, with E3 fitted on E2's out-of-fold probabilities
- [ ] An acceptance gate reports pass/fail: beats E1 on the selection set and no Merchant Family loses more than 0.05 F1 against E1
- [ ] Submission for this model refits on train plus the selection set and passes the validator (the file itself is generated only after the human confirms)
- [ ] A slow, marked test: a full evaluation run on the real data (stream detection cached, E2 cross-validation, E3 fit, scoring) finishes in about two minutes or less
