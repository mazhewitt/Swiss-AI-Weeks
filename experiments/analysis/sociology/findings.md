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

## Result

The script is `diagnostic.py`, its output is in `results.json`, and it runs in about 3 minutes on 3 cores.

**Gate verdict: it fails on both samples, so the sociological route is closed.**

- On train, the difference is −0.0019, with a 95% CI of [−0.0058, +0.0020].
- On selection, the difference is −0.0207, with a 95% CI of [−0.0363, −0.0050]. On selection, adding the six features does significantly *worse* than base alone.

The prior expectation was a gain below +0.01. The measured gain is negative on both samples.

### Base against base + 6 (logistic regression, standardised, L2, C = 1; 5×5 CV, out-of-fold averaged)

| sample | n (none) | base logit, raw | base LR | base + 6 LR | paired DeLong diff | 95% CI | p |
|---|---|---|---|---|---|---|---|
| train | 2000 (597) | 0.9083 | 0.9079 | 0.9060 | −0.0019 | [−0.0058, +0.0020] | 0.35 |
| selection | 700 (205) | 0.8779 | 0.8769 | 0.8563 | −0.0207 | [−0.0363, −0.0050] | 0.010 |

### Standalone `none` AUC, measured in the direction the literature predicts, with the DeLong 95% CI of the raw AUC in brackets

| feature | predicted | train | selection |
|---|---|---|---|
| income_trend | lower → none | 0.512 [0.460, 0.515]* | 0.476 [0.477, 0.571]* |
| spend_contraction | lower → none | 0.457 [0.514, 0.571]* | 0.459 [0.494, 0.588]* |
| essential_share | higher → none | 0.519 [0.490, 0.547] | 0.566 [0.519, 0.613] |
| price_rise | higher → none | 0.418 [0.390, 0.445] | 0.371 [0.329, 0.413] |
| discretionary_share | higher → none | 0.403 [0.377, 0.429] | 0.388 [0.343, 0.432] |
| recent_stops | higher → none | 0.571 [0.546, 0.595] | 0.527 [0.494, 0.560] |

\* For the features predicted "lower → none", the bracket is the CI of the "high → none" AUC. The predicted-direction AUC is 1 minus that AUC.

What the standalone AUCs show:

- **Growing card spend goes with `none`**, the opposite of the prediction, on both samples.
- **`price_rise` and `discretionary_share` point strongly the wrong way.** A Client with no Active Stream gets 0 on both, so these features mostly restate "has Active Streams", and the base already carries that.
- **`recent_stops` is the only feature in the predicted direction on both samples.** Its CI excludes 0.5 only on train.

### Coefficients (full-sample fit, standardised), with the share of the 25 CV folds where the coefficient is positive

| feature | predicted sign | train | matches | selection | matches |
|---|---|---|---|---|---|
| base | + | +2.728 (1.00) | | +1.664 (1.00) | |
| income_trend | − | −0.031 (0.20) | yes | +0.147 (1.00) | no |
| spend_contraction | − | +0.223 (1.00) | no | +0.315 (1.00) | no |
| essential_share | + | −0.009 (0.40) | no | +0.163 (1.00) | yes |
| price_rise | + | +0.040 (0.92) | yes | −0.033 (0.24) | no |
| discretionary_share | + | +0.036 (1.00) | yes | −0.036 (0.28) | no |
| recent_stops | + | +0.219 (1.00) | yes | +0.114 (1.00) | yes |

Across the two samples only `recent_stops` keeps the predicted sign, and its train strength does not carry over. `spend_contraction` has a stable sign, but the wrong one. The other four flip between train and selection.

### Adversarial AUC (label-free, each feature on its own, direction-free; "all six" is a logistic regression scored 5-fold out-of-fold)

| feature | train vs test | selection vs test | train vs selection |
|---|---|---|---|
| income_trend | 0.507 | 0.515 | 0.509 |
| spend_contraction | 0.501 | 0.500 | 0.501 |
| essential_share | 0.519 | 0.503 | 0.521 |
| price_rise | 0.564 | 0.508 | 0.557 |
| discretionary_share | 0.533 | 0.527 | 0.505 |
| recent_stops | 0.605 | 0.504 | 0.601 |
| all six | 0.653 | 0.514 | 0.640 |

`recent_stops` is the one feature that helps on train, and it is also the one that shifts most: mean 0.49 on train against 0.20 on test and selection. Selection looks like test here, and train does not. So what `recent_stops` shows on train would not carry to test, which fits the pattern of the other analyses.

### Deviations and choices, all fixed before any label was read

1. **`income_trend`:** no deviation. The data has an inbound top-up type, always described "salary", and the feature uses it. Received P2P transfers are excluded.
2. **Essential MCCs:** telecom (4814) is not counted as a utility, and the standard utilities code 4900 is. Of the listed codes, only 5411, 5912 and 4111 occur in the data.
3. **`recent_stops`:** it counts only lapsed streams with at least `min_payments` (3) payments. I read "no longer active" as "once qualified as active".
4. **Empty sets:**
   - `price_rise` and `discretionary_share` are 0 for a Client with no Active Stream.
   - `essential_share` is 0 for a Client with no card spend in the last 180 days.
5. **Base:** P(none) is clipped to [1e-6, 1 − 1e-6] before the logit.
6. **Reporting method, which the ticket did not specify:**
   - coefficients come from a full-sample fit, with the fold sign shares alongside;
   - the adversarial AUCs are single-feature and direction-free, plus a six-feature logistic regression.
