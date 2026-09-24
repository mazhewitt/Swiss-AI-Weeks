# 02: Honest evaluation: sealed holdout, leakage guards, bootstrap, cross-validation

**What to build:** Every number the team sees is trustworthy. The valid Clients are split into a selection set and a sealed holdout before any tuning. Guards make it impossible to train on valid labels, or to read sealed-holdout labels outside a checkpoint that the human starts. Each result reports whether it beat the previous best by more than bootstrap noise. A cross-validation harness produces out-of-fold probabilities for any model.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Valid is split once into a selection set (~700 Clients) and a sealed holdout (~300) with a fixed seed, stratified by label; the split is identical across runs, disjoint, and covers all of valid
- [ ] A training run that reads valid labels fails loudly
- [ ] Reading sealed-holdout labels outside checkpoint mode fails loudly; checkpoint mode (an explicit CLI flag the human uses) succeeds and writes its own row type to the experiment log
- [ ] Evaluate scores the selection set by default
- [ ] Each result is compared to the previous best with a paired bootstrap; the log records the delta and a verdict (improvement / tie / worse), with deltas under ~0.03 treated as ties
- [ ] A 5-fold stratified cross-validation harness works for any model implementing the model interface; it saves out-of-fold probabilities for later use
- [ ] Submission models can be refit on train plus the selection set
- [ ] All behaviour is tested through the CLI against fixture data (Seam 1)
