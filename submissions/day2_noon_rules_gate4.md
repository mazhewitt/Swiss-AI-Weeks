# Day-2 12:00 milestone: candidate comparison

Scored on the 700 selection set Clients from each run's committed predictions (`experiments/runs/<run>.csv`). Paired bootstrap over Clients: 2000 resamples (seed 0), the same resamples for every candidate. A delta under 0.03 is a tie; otherwise the paired 95% interval must exclude 0. Candidates are listed simplest first; on a tie the simpler candidate is preferred.

| # | Candidate | Run | Macro-F1 | 95% interval |
|---|---|---|---|---|
| 1 | rules+gate4: Milestone-2 rule: E1 plus the none-gate (none when the longest surviving stream has <= 4 payments) | 20260924T151704-615cf5 | 0.5490 | 0.5106 .. 0.5848 |
| 2 | ranker+tuned: Stream Ranker: LightGBM binary over every Candidate Stream (stream-table features only), per-Client max/none aggregation, tuned E3 decision | 20260924T154434-ee87f3 | 0.5714 | 0.5315 .. 0.6071 |
| 3 | ranker+pseudo+tuned: Stream Ranker with Pseudo-Labels: real train labels pooled with train and unlabeled Clients Pseudo-Labelled at Shifted Cutoff 2025-10-03 (min_payments 4, weight 0.5); tuned E3 decision fitted on real-labelled out-of-fold rows only | 20260924T171108-3251a4 | 0.5735 | 0.5344 .. 0.6075 |

Paired comparisons (the more complex candidate minus the simpler one):

| Candidate | Against | Delta | 95% interval | Verdict |
|---|---|---|---|---|
| ranker+tuned | rules+gate4 | +0.0224 | -0.0111 .. +0.0565 | tie |
| ranker+pseudo+tuned | rules+gate4 | +0.0245 | -0.0072 .. +0.0575 | tie |
| ranker+pseudo+tuned | ranker+tuned | +0.0021 | -0.0272 .. +0.0306 | tie |

**Winner: 20260924T151704-615cf5 (rules+gate4)**: it is the simplest candidate that no other candidate beats; 20260924T171108-3251a4 scores higher (0.5735 vs 0.5490) but only by a tie, so the simpler candidate is kept.

## Submission

`submissions/day2_noon_rules_gate4.csv`: the winner, the milestone-2 rule (soonest Active Stream due within the
Horizon, `none` when the Client's longest such stream has at most 4 payments), refit on train plus
the selection set and applied to the 1,000 test Clients. The rule learns nothing from labels, so the
refit only checks the labels are allowed; its test predictions equal the milestone-2 submission's.

Made by `bash scripts/day2_noon_milestone.sh`, which runs:

    uv run rf compare --run 20260924T151704-615cf5 --run 20260924T154434-ee87f3 --run 20260924T171108-3251a4 \
        --out submissions/day2_noon_rules_gate4.md --title "Day-2 12:00 milestone: candidate comparison"
    uv run rf train --model rules --none-gate --with-selection
    uv run rf submit --model rules --name day2_noon_rules_gate4
    uv run rf submit --check submissions/day2_noon_rules_gate4.csv

The sealed holdout was not read: the comparison uses the selection-set runs above and the refit
uses train plus the selection set only.
