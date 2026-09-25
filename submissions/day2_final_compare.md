# Candidate comparison

Scored on the 700 selection set Clients from each run's committed predictions (`experiments/runs/<run>.csv`). Paired bootstrap over Clients: 2000 resamples (seed 0), the same resamples for every candidate. A delta under 0.03 is a tie; otherwise the paired 95% interval must exclude 0. Candidates are listed simplest first; on a tie the simpler candidate is preferred.

| # | Candidate | Run | Macro-F1 | 95% interval |
|---|---|---|---|---|
| 1 | ranker+pseudo+tuned: Stream Ranker with Pseudo-Labels: real train labels pooled with train and unlabeled Clients Pseudo-Labelled at Shifted Cutoff 2025-10-03 (min_payments 4, weight 0.5); tuned E3 decision fitted on real-labelled out-of-fold rows only | 20260924T171108-3251a4 | 0.5735 | 0.5344 .. 0.6075 |
| 2 | ranker+none+pseudo+tuned: Stream Ranker with a Client-level none model (--none-model): LightGBM binary on the real-labelled Clients with Candidate Streams, features from the stream table and stream member payments and matched refunds (group A candidate-field aggregates, group B churn features) plus the cross-fitted ranker none; families keep their share of the ranker's scores. Pseudo-Labels train+unlabeled@2025-10-03 (min_payments 4, weight 0.5) train the ranker only; tuned E3 decision | 20260924T225229-cc7243 | 0.5764 | 0.5355 .. 0.6120 |
| 3 | ranker+pseudo+tuned: Stream Ranker with Pseudo-Labels (train+unlabeled@2025-10-03, weight 0.5, min_payments 4) on the ticket-11 detector: a stray payment (a description naming no family, left out of every stream: a Filler Description on another family's home MCC or on no home MCC, or an unresolved ambiguous one) joins a stream whose amount and schedule it fits (schedule tolerance 3 days); tuned E3 decision | 20260925T001614-c904bc | 0.5770 | 0.5384 .. 0.6124 |
| 4 | ranker+none+pseudo+tuned: Stream Ranker with its Client-level none model (--none-model) and Pseudo-Labels (train+unlabeled@2025-10-03, weight 0.5, min_payments 4) on the ticket-11 detector: a stray payment (a description naming no family, left out of every stream: a Filler Description on another family's home MCC or on no home MCC, or an unresolved ambiguous one) joins a stream whose amount and schedule it fits (schedule tolerance 3 days); tuned E3 decision | 20260925T001755-e309a3 | 0.5871 | 0.5479 .. 0.6214 |
| 5 | ranker+none+pseudo+tuned: Stream Ranker with its Client-level none model (--none-model) and Pseudo-Labels (train+unlabeled@2025-10-03, weight 0.5, min_payments 4) on the ticket-12 detector: a Decoy-described payment (never a shop payment or fee) left out of every stream joins one under the Stray Payment rules (its amount range, an empty slot of its schedule within 3 days, a stream of 2+ payments with a period of 7+ days; never starts one), after the stray payments; tuned E3 decision | 20260925T014822-cd37c4 | 0.5884 | 0.5482 .. 0.6222 |

Paired comparisons (the more complex candidate minus the simpler one):

| Candidate | Against | Delta | 95% interval | Verdict |
|---|---|---|---|---|
| ranker+none+pseudo+tuned | ranker+pseudo+tuned | +0.0029 | -0.0221 .. +0.0267 | tie |
| ranker+pseudo+tuned | ranker+pseudo+tuned | +0.0035 | -0.0145 .. +0.0230 | tie |
| ranker+none+pseudo+tuned | ranker+pseudo+tuned | +0.0136 | -0.0125 .. +0.0398 | tie |
| ranker+none+pseudo+tuned | ranker+pseudo+tuned | +0.0149 | -0.0113 .. +0.0405 | tie |
| ranker+pseudo+tuned | ranker+none+pseudo+tuned | +0.0006 | -0.0264 .. +0.0294 | tie |
| ranker+none+pseudo+tuned | ranker+none+pseudo+tuned | +0.0107 | -0.0105 .. +0.0332 | tie |
| ranker+none+pseudo+tuned | ranker+none+pseudo+tuned | +0.0120 | -0.0114 .. +0.0363 | tie |
| ranker+none+pseudo+tuned | ranker+pseudo+tuned | +0.0101 | -0.0136 .. +0.0333 | tie |
| ranker+none+pseudo+tuned | ranker+pseudo+tuned | +0.0113 | -0.0127 .. +0.0346 | tie |
| ranker+none+pseudo+tuned | ranker+none+pseudo+tuned | +0.0012 | -0.0187 .. +0.0205 | tie |

**Winner: 20260924T171108-3251a4 (ranker+pseudo+tuned)**: it is the simplest candidate that no other candidate beats; 20260925T014822-cd37c4 scores higher (0.5884 vs 0.5735) but only by a tie, so the simpler candidate is kept.
