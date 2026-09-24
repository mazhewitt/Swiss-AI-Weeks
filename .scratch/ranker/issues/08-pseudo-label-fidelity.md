# 08: Pseudo-Labels that pass the fidelity check

**What to build:** Find a Pseudo-Label setup that passes the fidelity check from ticket 04, and retrain the Stream Ranker on it. Today's check fails (run `20260924T162515-d6d36f`). At `min_payments` 4 the Pseudo-Label `none` share is 0.15 against a real 0.30. At 7 the share matches, but the rule scores 0.78 against Pseudo-Labels versus 0.53 on real labels, so the Pseudo-Label task is much easier than the real one. The ranker trained on the current Pseudo-Labels only ties the one trained without them (+0.002 on selection).

First, find out why the tasks differ. Then change the labeller and the check's search space to match. Ideas from ticket 04:

- ignore a Horizon payment of a stream that starts only inside the Horizon;
- a looser or stricter Horizon match;
- a Shifted Cutoff earlier than 2025-10-03, or several Shifted Cutoffs pooled (each Horizon must be fully observed).

Any other labeller change that makes the Pseudo-Label task look like the real one is also in scope, for example modelling how real Recurring Streams stop. The real labels are the ground truth for what "looks like" means. Every choice is made on train only.

**Blocked by:** 04, 05

**Status:** ready-for-agent

- [ ] A written diagnosis of why the Pseudo-Label task is easier than the real one, with figures from train (a comment on this ticket)
- [ ] The labeller and the fidelity check support the new options. Every new option has a unit test at the labeller seam, and the cache key includes every new parameter
- [ ] The fidelity check over the new search space is logged. Report PASS or FAIL honestly, and don't loosen the tolerances
- [ ] If a setup passes or clearly narrows both gaps: the ranker is cross-validated on train with it and compared with the current Pseudo-Label ranker on train out-of-fold (tuned macro-F1, argmax macro-F1). Only if the train result is better, one run is logged on the selection set with a paired bootstrap against run `20260924T171108-3251a4` (0.5735), and the predictions are committed
- [ ] The default Pseudo-Label settings (`PSEUDO_MIN_PAYMENTS`, the CLI defaults) change only when the new setup wins on train. Otherwise they stay as they are, and the ticket records the result
- [ ] The full fast test suite passes
