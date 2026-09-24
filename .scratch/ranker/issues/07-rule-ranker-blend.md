# 07: Blend of the gated rule and the Stream Ranker

**What to build:** A `blend` model that averages the milestone-2 rule's probabilities (one-hot, with the `none`-gate) and the Stream Ranker's, as `rule_weight * rule + (1 - rule_weight) * ranker`. It works through `train`, `cv`, `evaluate` and `submit` like any model. The rule takes `--none-gate`, `--ordering` and `--param`, and the ranker takes `--pseudo` sources. The tuned decision layer (E3) works on top. The rule weight is chosen on train out-of-fold probabilities only, never on the selection set. Motivation: the error analysis after ticket 06 found that the rule and the Pseudo-Label ranker disagree on 139 selection Clients where exactly one of them is right (61 rule, 78 ranker).

**Blocked by:** 01, 03, 05

**Status:** done

- [x] `train --model blend --rule-weight W` fits both parts; `predict_proba` returns the allowed labels, rows sum to 1, one row per requested Client
- [x] Rule weight 1 reproduces the rule's predictions and rule weight 0 the ranker's, for the same settings
- [x] `--none-gate`, `--ordering`, `--param` reach the blend's rule, and `--pseudo`, `--pseudo-weight`, `--pseudo-min-payments` reach its ranker, in `train` and in `cv`
- [x] Save then load gives identical probabilities, and the rule weight and gate survive the round trip
- [x] The log's model column names the blend's settings (e.g. `blend+gate4+w0.3+pseudo+tuned`)
- [x] A rule-weight sweep on train out-of-fold probabilities picks the weight, and the result is recorded
- [x] One run is logged on the selection set (paired bootstrap against the previous best), with predictions committed

## Outcome

- Rule-weight sweep on train out-of-fold probabilities (`scripts/blend_weight_sweep.py`, `experiments/blend_weight_sweep.csv`): tuned macro-F1 is 0.5819 at weight 0 (the ranker alone), 0.5843 at 0.1 (the best), 0.5825 at 0.2, falling to 0.5331 at 1 (the rule alone). The gain over the ranker is +0.002.
- Selection set (run `20260924T195225-27531a`, `blend+gate4+w0.1+pseudo+tuned`): macro-F1 0.5726. Against the previous best, the ranker with Pseudo-Labels at 0.5735, the delta is -0.0009 (95% -0.0206 .. +0.0182): a tie.
- Conclusion: no rule weight turns the 61/78 disagreement into a gain, on train or on selection. A likely reason, not verified, is that the rule's only signal (the order by projected payment date) is already a ranker feature, so where the two disagree the rule is right no more often than chance. The blend stays a logged comparison; the Day-2 12:00 upload is unchanged.
