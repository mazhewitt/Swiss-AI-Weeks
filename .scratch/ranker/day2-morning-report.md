# Day 2 morning report (overnight work, 23:18–06:30 CEST)

## Upload plan

The final rank is the **best** macro-F1 across all milestones, so the two remaining uploads should be two different strong candidates.

| Slot | File | Why |
|---|---|---|
| **Milestone 3, before 12:00** | `submissions/day2_final_ranker_none_v2.csv` | Best candidate from the default code: 0.5871 on selection, +0.014 over the noon file (a tie under our 0.03 rule, but every piece of evidence points the same way, below). Uploading it at noon locks it in even if something goes wrong at 17:30. |
| **Milestone 4, before 17:30** | `submissions/day2_noon_ranker_pseudo.csv` (the hedge), unless something better comes up today | The ranker without the `none` model. Its signals and failure modes differ, so it covers the risk that the `none` model does worse on test than on selection. |

If you already uploaded the noon file last night, replacing it with v2 before 12:00 is the change to make.

**Why v2:**

- The Client-level `none` model gains +0.040 on train out-of-fold (nested) and was reviewed for leakage by three critics.
- Its gain shrank on selection because the stream features drift between train and valid/test. Ticket 11 found the cause (Filler Descriptions on other families' home MCCs) and fixed it in the detector, and that narrowed the drift. For the `none` model's features, the train-vs-test classifier AUC fell from 0.81 to 0.73, and `max_missed_rate` in test from 0.167 to 0.089.
- v2 on the fixed detector then scores the highest of any default-code run on selection.
- `rf compare` (`submissions/day2_final_compare.md`) calls all five candidates ties and, by its tie rule, keeps the simplest: the noon run. That rule is for choosing one file. With the best of several milestones counting, a second slot costs nothing.

## Holdout checkpoint (human-only, once, for the finalist)

Run this before the 17:30 upload. It trains on train only, then scores the sealed 300-Client holdout:

```sh
uv run rf train --model ranker --none-model --decision tuned \
    --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 --pseudo-weight 0.5 --pseudo-min-payments 4
uv run rf evaluate --model ranker --decision tuned --checkpoint --change "Day-2 final: v2 finalist holdout checkpoint"
bash scripts/day2_final_ranker_none_v2.sh   # refit on train+selection; rewrites the v2 file byte for byte
```

A single holdout number can't choose between tied models: its 95% interval is about ±0.04. It is a sanity check. Expect roughly 0.55–0.62. A much lower score would suggest something is broken and would favour the hedge.

## What happened overnight

| Ticket | Result |
|---|---|
| D: seeds and capacity (train only) | Seed averaging adds nothing. The tuned decision layer is worth +0.014 when fitted and scored on different folds. A larger ranker gains on train alone, but not once the `none` model is in. |
| A: churn analysis | GO: modelling `none` per Client lifts the `none` AUC from 0.69 to 0.84–0.88. `experiments/analysis/churn/` |
| 08: Pseudo-Label fidelity | The check PASSES with uniform churn 0.2, but that's weak evidence (it matches marginals, not who churns). The ranker ties on selection (0.5666), so the defaults are unchanged. Critics caught a bug where the fidelity lookup took the last check whatever its settings; fixed. |
| 09: Client-level `none` model | Train +0.040 nested. Selection 0.5764, a tie. |
| 10: hidden series | Closed as wontfix. Ticket 11 showed the signal is a train-only artefact (it appears in 38–52% of valid and test Clients). |
| 11: Stray Payments join streams | A detector fix for Filler Descriptions on other families' home MCCs. Drift narrowed; selection 0.5871 with the `none` model. `--param join_strays=false` reproduces the old detector, and the milestone files reproduce byte for byte. Critics also caught a latent `--param` bug: "false" was read as True. |
| 12: Decoy-described stream payments | Selection 0.5884 (+0.001, a tie), train −0.008, and about 1 in 5 of the test joins would be real Decoys. Built but **off by default** (ADR 0001). v3 is reproducible at commit 73ec6a5 only. |

Every ticket went through an implementer, three adversarial critics (one for ticket 12, whose change is off by default), fix rounds and a serial merge. Follow-up tests from the critics' mutation runs are merged. Fast suite: 457 passed. Slow suite: passes, except the two tests that run `rf evaluate` on the selection set, which were not run overnight.

## Decisions for you

1. **Fast-forward `main` to `main-rotk33`?** The form asks for the repo link, and `main` is still at 7c8cf60. Everything tonight is on `main-rotk33`. I haven't pushed to `main` without your OK.
2. **The upload plan above:** v2 at noon, the noon file as the 17:30 hedge.
3. **Classifier denial:** at 23:40 the auto-mode classifier blocked my ad-hoc drift script over valid, test and unlabelled transactions as "PII data handling". I didn't work around it. Ticket 11's diagnosis later answered the same question with the package's own stream tables.

## Known follow-ups (not blocking)

- A stream can take an unlimited number of Stray Payments. 22 train streams have at least as many strays as their own payments; valid and test have 3 each. A cap would change outputs, so it's left until after the deadline.
- Same-day repeated payments crash `_summarise` (ZeroDivisionError). This predates tonight and never happens in the real data.
- `rf cv --none-model` overwrites `artifacts/oof/ranker.csv`, and `artifacts/ranker.json` is shared between setups.
