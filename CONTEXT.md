# Transaction Activity Forecasting

Predicting, for each Client, which Merchant Family they will next make a recurring payment to after the Cutoff, from their transaction history. This is the 2026 UBS Swiss{ai}Weeks hackathon challenge.

## Language

### Prediction

**Client**:
A bank customer identified by a `client_id`, whose transaction history up to the Cutoff is the input to a prediction.
_Avoid_: Customer, account, user

**Merchant Family**:
The kind of recurring payee: one of `cloud`, `gym`, `insurance`, `mobile`, `music`, `software` or `streaming`. It is the unit we predict, never the individual merchant.
_Avoid_: Category (clashes with MCC merchant category), merchant, merchant type

**Cutoff**:
The date (2026-01-01) that splits a Client's known history from the future being predicted.
_Avoid_: Snapshot date, as-of date

**Horizon**:
The 90 days after the Cutoff in which a Recurring Stream must recur to count.
_Avoid_: Window, forecast period

**Next Recurring Family**:
The label: the Merchant Family of the first Recurring Stream payment a Client makes within the Horizon, or `none`. It is defined by what happens after the Cutoff, not by any rule over the history.
_Avoid_: Target merchant, next merchant

**Shifted Cutoff**:
A Cutoff moved earlier inside the known history, so that its whole Horizon has already been observed.
_Avoid_: Pseudo-cutoff, backtest date

**Pseudo-Label**:
The Next Recurring Family observed after a Shifted Cutoff, taken from the Client's own later payments rather than from a label file.
_Avoid_: Weak label, silver label

### Streams

**Recurring Stream**:
A Client's repeated payments to the same kind of payee at a regular interval. A Recurring Stream belongs to one Merchant Family.
_Avoid_: Subscription (insurance premiums and phone contracts are not subscriptions), recurring transaction

**Active Stream**:
A Recurring Stream with at least three payments whose most recent payment is within about 1.6 periods of the Cutoff.
_Avoid_: Live stream, current subscription

**Candidate Stream**:
Any of a Client's Recurring Streams, considered as the one that will carry the Client's Next Recurring Family. It need not be an Active Stream.
_Avoid_: Option, choice

**Survival Race**:
A way to predict the Next Recurring Family from a Client's Candidate Streams. Each stream survives the Cutoff with its own probability, and the streams race in projected payment order: the label is the family of the first survivor, or `none` if none survives. Under the soft race order each stream's payment date is uncertain (its own schedule spread) and the race is averaged over every order the dates allow; `none` is unchanged by the order. Streams behind the label's stream are censored in training: nobody saw whether they would have paid.
_Avoid_: Churn model, hazard model

**Decoy Transaction**:
A card payment with a generic, subscription-sounding description (e.g. "digital order", "merchant charge") that belongs to no Recurring Stream. Valid and test book some real stream payments with such a description. The detector can let one join a stream like a Stray Payment (`join_decoys`), but that is off by default (ADR 0001, ticket 12).
_Avoid_: Noise, fake transaction

**Filler Description**:
A generic description (e.g. "member plan", "subscription charge") that appears on a payment inside a real Recurring Stream and takes that stream's Merchant Family.
_Avoid_: Generic description (ambiguous with Decoy Transaction)

**Stray Payment**:
A card payment with a description that names no Merchant Family (a Filler Description, or an ambiguous one such as "monthly plan") that no stream of its family fits, often because it was booked on another family's home MCC or on no home MCC. It joins an existing Recurring Stream whose amount and schedule it fits, and never starts one.
_Avoid_: Orphan payment, hidden payment (a "hidden series" is a run of such payments that fits no stream)
