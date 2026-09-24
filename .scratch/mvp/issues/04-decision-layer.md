# 04: E3 decision layer for any probability model

**What to build:** For any model that produces probabilities, the team can switch on a decision layer that turns them into labels in a way that maximises macro-F1, instead of taking the argmax. It fits one weight per class plus a `none` threshold on out-of-fold probabilities. It is proven on the prior model and on fixture probabilities, so it is ready before LightGBM exists.

**Blocked by:** 02

**Status:** resolved

- [x] `--decision tuned` fits per-class weights and a `none` threshold by grid search on out-of-fold probabilities only (never on the evaluated Clients)
- [x] Uniform weights with no threshold reproduce argmax exactly
- [x] A `none` threshold of 0 yields no `none` predictions; a threshold of 1 yields all `none`
- [x] On its own fitting data, the tuned decisions never have lower macro-F1 than argmax
- [x] Fitting never reads valid or sealed-holdout labels (the guards from 02 stay green)
- [x] Tested through the CLI against fixture data (Seam 1)

## Comments

- Merged into main as 4b3b2a0 ("Merge ticket 04: decision-layer"); clean auto-merge (cli.py, SOLUTION.md).
- Tests after merge: `uv run pytest -m "not slow"` -> 178 passed.
- Experiment (run 20260924T140433-16f903, `rf train --model prior --decision tuned` then `rf evaluate --model prior --split selection --decision tuned`): selection macro-F1 0.0566 (none F1 0.4530, every family 0.0000); delta +0.0000 vs the prior baseline 20260924T134044-2e6cd3 -> tie. Expected: the prior model gives every Client the same probabilities, so the out-of-fold fit (0.0575 argmax -> 0.0575 tuned) keeps uniform weights and no none threshold. The decision layer earns its keep only on a model whose probabilities vary by Client (E1/E2).
