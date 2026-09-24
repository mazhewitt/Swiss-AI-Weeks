# 04: E3 decision layer for any probability model

**What to build:** For any model that produces probabilities, the team can switch on a decision layer that turns them into labels in a way that maximises macro-F1, instead of taking the argmax. It fits one weight per class plus a `none` threshold on out-of-fold probabilities. It is proven on the prior model and on fixture probabilities, so it is ready before LightGBM exists.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] `--decision tuned` fits per-class weights and a `none` threshold by grid search on out-of-fold probabilities only (never on the evaluated Clients)
- [ ] Uniform weights with no threshold reproduce argmax exactly
- [ ] A `none` threshold of 0 yields no `none` predictions; a threshold of 1 yields all `none`
- [ ] On its own fitting data, the tuned decisions never have lower macro-F1 than argmax
- [ ] Fitting never reads valid or sealed-holdout labels (the guards from 02 stay green)
- [ ] Tested through the CLI against fixture data (Seam 1)
