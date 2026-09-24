# 04: `rf pseudo-labels` command and fidelity check

**What to build:** `rf pseudo-labels --split <split> --cutoff <date>` writes a Pseudo-Label table for any split (train, valid, test, unlabeled) and prints the label distribution. It never reads a label file. Pseudo-Label tables are cached by split, Shifted Cutoff, detector version and labeller parameters (default Shifted Cutoff: 2025-10-03 UTC). A fidelity check on train compares the Pseudo-Label `none` share with the real share, and the milestone-2 rule's macro-F1 against Pseudo-Labels at the Shifted Cutoff with its real train score. It passes when both are within 0.05. The check chooses the labeller's minimum-payments setting, and the result is logged.

**Blocked by:** 01, 02

**Status:** done

- [x] The command writes a table with one row per Client and prints the per-label counts and shares
- [x] Running it on valid and test does not trigger any label guard
- [x] Inputs at the Shifted Cutoff use only transactions before it
- [x] A second run with the same parameters loads from the cache; changing any parameter misses the cache
- [x] The fidelity check prints both comparisons, the tolerances and PASS or FAIL, and appends a row to the experiment log
- [x] A slow, marked real-data test runs the fidelity check on train and records the chosen minimum-payments setting

## Comments

**Implementation (ranker/04-pseudo-labels-command).**

- `rf pseudo-labels --split S [--cutoff YYYY-MM-DD] [--min-payments N] [--param NAME=VALUE] [--out CSV]` writes `artifacts/pseudo_labels/<split>-<cutoff>-min<N>.csv` with the columns of a label file (`client_id, cutoff_date, target_next_recurring_merchant`), one row per Client in the split's transactions. It prints all eight labels' counts and shares. The default `--min-payments` is `pseudo.PSEUDO_MIN_PAYMENTS` (4), the fidelity check's choice. The labeller function's own default stays 2.
- Labelling runs inside `data.pseudo_labelling()`, which refuses every label read (train's too). Tests also delete the label files and stub the reader.
- Cache: `streams.cached_pseudo_labels` in `artifacts/pseudo_labels/cache/`, keyed like the stream cache (raw file, detector source, family table) plus the Shifted Cutoff, Horizon, `min_payments` and stream parameters. Settings not yet cached are labelled together in one detection pass (`streams.pseudo_label_sweep`), so the 9-setting sweep costs one pass (about 33 s on train).
- Inputs at a Shifted Cutoff: `RulesModel` now takes a `cutoff` (saved with the model) and `proba_from_streams`. The fidelity check runs the rule on the cached train stream table detected at the Shifted Cutoff, which holds only transactions before it. A Seam 2 test rewrites every payment from the Shifted Cutoff on and gets the same stream table and rule output.
- Fidelity check: `rf pseudo-labels --split train --fidelity [--candidates 2,3,...]` (default 2-10). It picks the setting whose worse gap relative to its tolerance is smallest (ties go to the smaller setting), writes that table and appends a `fidelity` row (model `rules+gate4`, split `train`). The row's macro-F1 and per-label F1 are the rule against Pseudo-Labels. `delta` is that minus the real score, `verdict` is `pass`/`fail`, the parameters are in `change` and the figures in `conclusion`. Ticket 05 can read the latest `fidelity` row's verdict.

**Result on real train (logged 2026-09-24, run 20260924T162515-d6d36f): FAIL.** Real: `none` share 0.2985, rule macro-F1 0.5331.

| min_payments | none share | rule macro-F1 vs Pseudo-Labels |
|---|---|---|
| 2 | 0.0825 | 0.5498 |
| 3 | 0.1165 | 0.5958 |
| 4 (chosen) | 0.1535 | 0.6453 |
| 5 | 0.1955 | 0.6926 |
| 6 | 0.2560 | 0.7673 |
| 7 | 0.2995 | 0.7844 |
| 8 | 0.3470 | 0.7993 |
| 9 | 0.4110 | 0.7305 |
| 10 | 0.5015 | 0.6572 |

No setting passes both. The `none` share matches the real one at 7 payments, but there the rule scores 0.78 against Pseudo-Labels, far above its 0.53 on real labels. Pseudo-Labels come from the same detector the rule uses, so the Pseudo-Label task is easier than the real one. Per the spec, the ranker still ships on real labels, Pseudo-Label training stays possible with the failed check noted (ticket 05), and the gap is the self-improvement loop's first investigation. Ideas to try: a labeller that ignores a Horizon payment in a stream that started only inside the Horizon, a looser Horizon match, or a Shifted Cutoff earlier than 2025-10-03.
