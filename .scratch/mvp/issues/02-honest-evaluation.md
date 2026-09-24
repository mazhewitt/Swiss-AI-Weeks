# 02: Honest evaluation: sealed holdout, leakage guards, bootstrap, cross-validation

**What to build:** Every number the team sees is trustworthy. The valid Clients are split into a selection set and a sealed holdout before any tuning. Guards make it impossible to train on valid labels, or to read sealed-holdout labels outside a checkpoint that the human starts. Each result reports whether it beat the previous best by more than bootstrap noise. A cross-validation harness produces out-of-fold probabilities for any model.

**Blocked by:** 01

**Status:** resolved

- [x] Valid is split once into a selection set (~700 Clients) and a sealed holdout (~300) with a fixed seed, stratified by label; the split is identical across runs, disjoint, and covers all of valid
- [x] A training run that reads valid labels fails loudly
- [x] Reading sealed-holdout labels outside checkpoint mode fails loudly; checkpoint mode (an explicit CLI flag the human uses) succeeds and writes its own row type to the experiment log
- [x] Evaluate scores the selection set by default
- [x] Each result is compared to the previous best with a paired bootstrap; the log records the delta and a verdict (improvement / tie / worse), with deltas under ~0.03 treated as ties
- [x] A 5-fold stratified cross-validation harness works for any model implementing the model interface; it saves out-of-fold probabilities for later use
- [x] Submission models can be refit on train plus the selection set
- [x] All behaviour is tested through the CLI against fixture data (Seam 1)

## Comments

- Merged into main as f8eadd5 (`Merge ticket 02: honest-evaluation`), no conflicts.
- Tests: `uv run pytest -m "not slow"` -> 133 passed (2 sklearn small-fixture warnings). No tests carry the slow marker yet.
- Experiment: `rf split` -> 700 selection / 300 sealed holdout Clients. Prior model trained on train (2000 Clients) and evaluated on the selection set: macro-F1 0.0566 (none 0.4530, every family 0.0000). Verdict `first`: the only earlier log row is on the old `valid` split, so there is no comparable previous best on `selection` and no bootstrap comparison yet. Run 20260924T134044-2e6cd3; predictions committed at experiments/runs/20260924T134044-2e6cd3.csv (commit 4c12c52). The earlier walking-skeleton row had no saved predictions, so there was nothing to move from artifacts/.
