# Model features come only from Recurring Streams, never raw transactions

Decoy Transactions appear about 0.16 times per Client in train, 3.1 in valid, 4.5 in test and 0 in the unlabelled set. So any feature counted over raw transactions (description frequencies, per-MCC spend, transaction counts) would shift between training and test. We therefore build every model feature from detected Recurring Streams. A payment only joins a stream when its description and MCC point to one Merchant Family and its amount fits the stream's amount cluster. This makes the features insensitive to the decoy rate by construction.

## Considered Options

- **Raw transaction features plus synthetic decoys injected into train at test-like rates.** Rejected as the primary approach because it depends on matching an unknown test decoy profile. It is kept as a robustness check if the gap between train and valid scores grows.
- **Ignore the shift.** Rejected: train looks nothing like test on this axis.

## Consequences

A detector test must show that stream output is unchanged when Decoy Transactions are injected into a history. Any new feature that reads transactions directly needs to justify itself against this ADR.

## Amendment (ticket 11): Stray Payments

A Stray Payment can now join an existing stream when its amount fits the stream's amount cluster and its date fits the stream's schedule. A Stray Payment is a payment whose description names no Merchant Family, such as a Filler Description on another family's home MCC or on no home MCC. Valid and test book many stream payments this way, and without the join their streams showed missed payments that train's streams did not. A Stray Payment never starts a stream. Decoy Transactions, shop payments and fees are still dropped before any join, so the guarantee of this ADR holds: the Decoy-injection tests pass unchanged. `StreamParams(join_strays=False)` restores the earlier detector, which the committed milestone submissions were made with.

## Amendment (ticket 12): Decoy-described stream payments

Valid and test also book some real stream payments with a Decoy description ("digital order", "merchant charge"). Before this change, per 100 streams of two or more payments, Decoy-described payments that fill a missed slot of a stream at its amount (within 4 days) number 0.88 in train, 8.64 in valid and 6.77 in test. The off-schedule control, the same count half a period off the schedule, is 0.06, 0.94 and 1.44. A random Decoy Transaction lands on an empty slot at a stream's amount at about the control rate, so most of the on-schedule ones in valid and test are stream payments.

`StreamParams.join_decoys` lets a Decoy-described payment join a stream under the Stray Payment rules. The payment must fit the stream's amount range and an empty slot of its schedule within `schedule_tolerance_days`. The stream needs at least two payments and a period of at least `stray_min_period_days`. The Stray Payments take their slots first. It never starts a stream, and shop payments and fees still never join. It also needs `join_strays`.

With it on, 1.5 (train), 14.7 (valid) and 13.3 (test) Decoy-described payments per 100 streams join, 0.95, 8.9 and 7.6 of them in a gap. The expected false joins are about the control rate: 0.06, 0.94 and 1.44 per 100 streams in gaps. So about 1 in 9 of the gap joins in valid and 1 in 5 in test are Decoy Transactions. Unlabeled has no Decoy Transactions. Results with it on:

- the train-vs-valid and train-vs-test shift of the stream features narrows: classifier AUC on the `none` model's features falls from 0.700 to 0.653 against valid and from 0.725 to 0.694 against test;
- train out-of-fold nested tuned macro-F1 of ranker+none+pseudo falls by 0.0077 (0.6082 to 0.6005), within the ticket's 0.01 gate;
- one selection run (`20260925T014822-cd37c4`, 0.5884) ties the same model on the ticket-11 detector (`20260925T001755-e309a3`, 0.5871): delta +0.0012, 95% −0.0187 .. +0.0205.

**Decision: `join_decoys` defaults to off.** The detector stays the ticket-11 one, so the guarantee of this ADR holds unchanged: stream output is unchanged under any Decoy injection. At a tie, this ADR's rule against features that move with the Decoy rate wins. Four reasons:

- the selection gain is a tie;
- train gets slightly worse;
- about 1 in 5 of the gap joins in test, the split with the most Decoys, is expected to be false;
- a Decoy that joins at a stream's end can extend it and make an ended stream look active: Active Streams per Client rose from 1.38 to 1.44 in valid, and train did not move.

The Decoy-injection tests run with `join_decoys` off and on. With it on they inject only Decoys that are off a stream's schedule or amount, since those are all it guarantees against, and a test shows that random Decoys at a stream's exact amount rarely join (under 10%). Follow-up: let a Decoy-described payment fill a gap between a stream's payments but never extend the stream beyond its first or last payment, then re-run the gate. `join_strays=False` still restores the detector before ticket 11.
