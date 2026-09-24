# Model features come only from Recurring Streams, never raw transactions

Decoy Transactions appear about 0.16 times per Client in train, 3.1 in valid, 4.5 in test and 0 in the unlabelled set. So any feature counted over raw transactions (description frequencies, per-MCC spend, transaction counts) would shift between training and test. We therefore build every model feature from detected Recurring Streams. A payment only joins a stream when its description and MCC point to one Merchant Family and its amount fits the stream's amount cluster. This makes the features insensitive to the decoy rate by construction.

## Considered Options

- **Raw transaction features plus synthetic decoys injected into train at test-like rates.** Rejected as the primary approach because it depends on matching an unknown test decoy profile. It is kept as a robustness check if the gap between train and valid scores grows.
- **Ignore the shift.** Rejected: train looks nothing like test on this axis.

## Consequences

A detector test must show that stream output is unchanged when Decoy Transactions are injected into a history. Any new feature that reads transactions directly needs to justify itself against this ADR.
