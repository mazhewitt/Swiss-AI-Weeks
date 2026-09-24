# 04: `rf pseudo-labels` command and fidelity check

**What to build:** `rf pseudo-labels --split <split> --cutoff <date>` writes a Pseudo-Label table for any split (train, valid, test, unlabeled) and prints the label distribution. It never reads a label file. Pseudo-Label tables are cached by split, Shifted Cutoff, detector version and labeller parameters (default Shifted Cutoff: 2025-10-03 UTC). A fidelity check on train compares the Pseudo-Label `none` share with the real share, and the milestone-2 rule's macro-F1 against Pseudo-Labels at the Shifted Cutoff with its real train score. It passes when both are within 0.05. The check chooses the labeller's minimum-payments setting, and the result is logged.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] The command writes a table with one row per Client and prints the per-label counts and shares
- [ ] Running it on valid and test does not trigger any label guard
- [ ] Inputs at the Shifted Cutoff use only transactions before it
- [ ] A second run with the same parameters loads from the cache; changing any parameter misses the cache
- [ ] The fidelity check prints both comparisons, the tolerances and PASS or FAIL, and appends a row to the experiment log
- [ ] A slow, marked real-data test runs the fidelity check on train and records the chosen minimum-payments setting
