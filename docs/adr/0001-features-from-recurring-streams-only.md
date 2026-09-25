# Model features come only from Recurring Streams, never raw transactions

Decoy Transactions appear about 0.16 times per Client in train, 3.1 in valid, 4.5 in test and 0 in the unlabelled set. So any feature counted over raw transactions (description frequencies, per-MCC spend, transaction counts) would shift between training and test. We therefore build every model feature from detected Recurring Streams. A payment only joins a stream when its description and MCC point to one Merchant Family and its amount fits the stream's amount cluster. This makes the features insensitive to the decoy rate by construction.

## Considered Options

- **Raw transaction features plus synthetic decoys injected into train at test-like rates.** Rejected as the primary approach because it depends on matching an unknown test decoy profile. It is kept as a robustness check if the gap between train and valid scores grows.
- **Ignore the shift.** Rejected: train looks nothing like test on this axis.

## Consequences

A detector test must show that stream output is unchanged when Decoy Transactions are injected into a history. Any new feature that reads transactions directly needs to justify itself against this ADR.

## Amendment (ticket 11): Stray Payments

A Stray Payment can now join an existing stream when its amount fits the stream's amount cluster and its date fits the stream's schedule. A Stray Payment is a payment whose description names no Merchant Family, such as a Filler Description on another family's home MCC or on no home MCC. Valid and test book many stream payments this way, and without the join their streams showed missed payments that train's streams did not. A Stray Payment never starts a stream. Decoy Transactions, shop payments and fees are still dropped before any join, so the guarantee of this ADR holds: the Decoy-injection tests pass unchanged. `StreamParams(join_strays=False)` restores the earlier detector, which the committed milestone submissions were made with.
