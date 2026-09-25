# Candidate comparison

Scored on the 700 selection set Clients from each run's committed predictions (`experiments/runs/<run>.csv`). Paired bootstrap over Clients: 2000 resamples (seed 0), the same resamples for every candidate. A delta under 0.03 is a tie; otherwise the paired 95% interval must exclude 0. Candidates are listed simplest first; on a tie the simpler candidate is preferred.

| # | Candidate | Run | Macro-F1 | 95% interval |
|---|---|---|---|---|
| 1 | survival+monthly-slot+soft+tuned: Ticket 14: soft race order. The Survival Race averaged over uncertain payment dates (each stream's date jittered by its own schedule spread, sigma = max(3.6, 1.4 x gap_mad_days) days, 6.2 for a single-payment stream, calibrated on train plus unlabeled history); monthly-slot base order, the hard race's fit; selection run | 20260925T102231-5456a7 | 0.5987 | 0.5582 .. 0.6339 |
| 2 | survival+tuned: Ticket 13: Survival Race (S-rank), one model for none and family; selection run | 20260925T080905-fd1dc5 | 0.5903 | 0.5501 .. 0.6260 |
| 3 | survival+monthly-slot+tuned: Ticket 13 fix round 2: Survival Race with monthly-slot race order (V2); selection run | 20260925T085645-c70788 | 0.5862 | 0.5461 .. 0.6219 |
| 4 | ranker+none+pseudo+tuned: Stream Ranker with its Client-level none model (--none-model) and Pseudo-Labels (train+unlabeled@2025-10-03, weight 0.5, min_payments 4) on the ticket-11 detector: a stray payment (a description naming no family, left out of every stream: a Filler Description on another family's home MCC or on no home MCC, or an unresolved ambiguous one) joins a stream whose amount and schedule it fits (schedule tolerance 3 days); tuned E3 decision | 20260925T001755-e309a3 | 0.5871 | 0.5479 .. 0.6214 |

Paired comparisons (the more complex candidate minus the simpler one):

| Candidate | Against | Delta | 95% interval | Verdict |
|---|---|---|---|---|
| survival+tuned | survival+monthly-slot+soft+tuned | -0.0084 | -0.0335 .. +0.0156 | tie |
| survival+monthly-slot+tuned | survival+monthly-slot+soft+tuned | -0.0125 | -0.0342 .. +0.0086 | tie |
| ranker+none+pseudo+tuned | survival+monthly-slot+soft+tuned | -0.0116 | -0.0415 .. +0.0178 | tie |
| survival+monthly-slot+tuned | survival+tuned | -0.0041 | -0.0281 .. +0.0188 | tie |
| ranker+none+pseudo+tuned | survival+tuned | -0.0032 | -0.0292 .. +0.0227 | tie |
| ranker+none+pseudo+tuned | survival+monthly-slot+tuned | +0.0009 | -0.0287 .. +0.0308 | tie |

**Winner: 20260925T102231-5456a7 (survival+monthly-slot+soft+tuned)**: it is the simplest candidate that no other candidate beats.
