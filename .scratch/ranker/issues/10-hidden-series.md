# 10: Hidden series as a `none` signal (parked)

**What to build:** Add the "hidden series" features (`experiments/analysis/churn/hidden.py`, `simulate_fillers.py`) to the `none` model from ticket 09. These are amount clusters of subscription-like card payments that join no Recurring Stream. In train, 99% of Clients with such a series of 3+ payments are truth-`none`. On train out-of-fold they add about +0.025 nested tuned macro-F1 on top of ticket 09's features.

**Why parked:** these are raw-transaction features, and ADR 0001 rules those out because Decoy Transactions drift: 0.16 per Client in train, 3.1 in valid, 4.5 in test. Before any build, a label-free drift check (transactions only) must compare the marker's rate per Client across train, valid, test and unlabeled. A simulation on train suggests the scattered-MCC variant (no MCC carries 75% of the cluster) keeps precision 0.97 under valid's Filler rate. The overnight session could not run the check, so a human decides whether to run it. If the rates match, amend ADR 0001 and build this ticket.

**Blocked by:** 09, and a human decision on the drift check

**Status:** wontfix

## Resolution

Ticket 11's label-free diagnosis answered the drift question: the hidden series are Filler Descriptions booked on another family's home MCC. In train they mark `none` Clients (92.5% of train Clients with such a payment on a stream's schedule are truth-`none`). In valid and test they are part of real streams: 38% of valid and 52% of test Clients have one. A `none` feature built on them would learn a train-only artefact. Ticket 11 lets them join their stream instead.
