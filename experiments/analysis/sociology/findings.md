# Ticket 20: sociological churn mechanisms and `none`

## Feature mapping (written and committed before any label was read)

The features are in `features.py`. I looked only at the train transactions to map them, and read no labels.

**Schema.** Transaction types and directions in train:

| type | direction | rows |
|---|---|---|
| card_payment | out | 93,464 |
| topup | in | 12,031 |
| transfer | out | 10,011 |
| atm | out | 8,083 |
| refund | in | 7,834 |
| p2p_transfer | in / out | 5,996 / 6,017 |
| fee | out | 4,023 |

- Every `topup` has the description "salary", with a median amount of about 5,000. History runs from 2024-11-07 to 2025-12-31 in train and test alike.
- Card payments use only 10 MCCs: 5812, 5411, 5732, 5734, 4814, 4111, 5912, 7011, 6300 and 7997.

**The mapping:**

- **`income_trend`:** the data has an inbound top-up/salary type, so there is no deviation here. The feature uses `type == "topup"`, which is always inbound and always "salary". Received P2P transfers are not salary or top-ups, so they are left out.
  - The last 90 days are [Cutoff − 90d, Cutoff).
  - The comparison is the total over [Cutoff − 360d, Cutoff − 90d) divided by 3.
  - The +1 follows the ticket.
- **`spend_contraction`:** all outgoing card payments (`card_payment`, `out`), recurring ones included, in the same windows.
- **Currencies:** amounts stay in each transaction's own currency. Both features are ratios within one Client, so no conversion is made.
- **`essential_share`:** amount-weighted, over the last 180 days of card spend. The list uses standard MCC meanings, written out in `ESSENTIAL_MCCS`:
  - groceries: 5411, 5422, 5441, 5451, 5462, 5499;
  - pharmacies: 5912, 5122;
  - utilities: 4900;
  - fuel: 5541, 5542, 5983;
  - public transport: 4111, 4112, 4131.

  4814 (telecommunication services) is left out. It is not "utilities" in the standard meaning, which is 4900, and it is also the mobile family's home MCC. In this data only 5411, 5912 and 4111 occur. A Client with no card spend in the window gets 0.
- **`price_rise`:** for each Active Stream in the detector's stream table (default `StreamParams`), the last member payment is divided by the median of the stream's earlier member payments, minus 1. The feature is the maximum over the Client's Active Streams, clipped at 0. A Client with no Active Stream gets 0.
- **`discretionary_share`:** the share of Active Streams in the families streaming, music or gym. A Client with no Active Stream gets 0.
- **`recent_stops`:** the number of streams whose last payment is within 180 days of the Cutoff and that are not `active`, using the stream table's own definition. I read "no longer active" as a stream that once qualified, so only streams with at least `min_payments` (3) payments count. One- and two-payment clusters were never Active Streams, so they cannot stop.

**Label-free notes before the test:**

- `income_trend` is strongly bimodal: salary top-ups are lumpy, about six per Client over 14 months. Many Clients have none in one window, so with +1 the log ratio sits near ±8. The formula is kept as registered, and the model standardises its inputs.
- `recent_stops` averages 0.49 on train against 0.20 on test, and `price_rise` averages 0.011 against 0.006. The adversarial AUCs below measure this drift.
