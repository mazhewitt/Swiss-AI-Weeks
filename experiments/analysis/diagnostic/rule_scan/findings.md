# Rule scan: a fresh look at the raw train data

Train data only: 2000 clients, 147k transactions, dated 2024-11-07 to 2025-12-31. No transaction falls after the cutoff.
Labels: none 597, streaming 225, insurance 214, music 198, software 195, mobile 191, cloud 190, gym 190.
Scripts are in this folder. Cached pickles went to the session scratchpad. Run order: profile, features, rules, refunds, order, lgbm, hazard, amtcluster, periods, lgbm2, hazard2, age.

## Profile
- There are only 12 MCCs. The family home MCCs are 4814 mobile, 6300 insurance, 7997 gym, 5732 cloud and 5734 software. Music and streaming both sit on **5812**, which is also the restaurant MCC. On 5812 the two families can only be told apart by description or by amount.
- Family keywords: music is audio / member pass; streaming is media stream / video; mobile is phone / service bill; insurance is cover / policy / insurance / safe; gym is gym / fit / urban; cloud is cloud / storage / service plan; software is saas / productivity / suite / software.
- Some descriptions are ambiguous and take the family of the stream they sit in. "digital plus" appears on 4814 and 5812. "premium plan" appears on 5734 and 5812. "monthly plan", "member plan", "subscription charge" and "digital service" are spread across MCCs.
- Decoys ("digital order", "merchant charge", "service payment", "card purchase") have wide amounts and random MCCs.
- Descriptions are built from a vocabulary of 73 tokens, with prefixes (billing / pay / member) and suffixes (plus / online / digital / core / service). There are also abbreviations (dgtl, prem, mth, prod) and truncations ("urban", "safe", "cloud").
- **Fee is noise.** About 11-14% of rows have a fee, whatever the type or currency.
- **Refunds are noise.** 3.3k subscription-looking refunds copy a payment 0-15 days later (median 2.6 days). Refunding the last payment does not signal cancellation: P(label) is 0.36 with a refund and 0.32 without.
- Nothing in the descriptions mentions a renewal, cancellation or trial, and no transaction is dated after the cutoff.
- Amounts are about 1% CV within a real stream. Noise payments that carry a family description are about 5% CV.

## Rule macro-F1 on train (8 labels)
| Rule | macro-F1 |
|---|---|
| all none | 0.058 |
| most recent family payment (any window) | 0.32-0.35 |
| soonest by last day-of-month, last payment <=35 days before cutoff | 0.468 |
| soonest last+median gap, raw per-family grouping, last payment <=30 days | 0.485 |
| same rule on amount clusters (2% tol, n>=4, gap 25-35, alive <=1.2 periods) | 0.507 |
| same rule, allowing gaps of 10-40 days (includes biweekly streams) | **0.521** (best rule) |
| LightGBM on naive raw per-family features, 5-fold | 0.574 (0.577 with none down-weighted) |
| LightGBM on amount-cluster + raw features, 5-fold | 0.560 / 0.563 (two seeds) |

No naive feature set beats our 0.61 out-of-fold score.

## Structural findings
1. **Scheduling is not the bottleneck.** Take the clients whose label is one of their live clean streams and assume `none` is known. Picking the soonest projected payment (last payment + median gap) is right **78%** of the time. Picking the most recent last payment is right 76%. Periods differ by stream (median 30 days, IQR 28.4-30.9), so a calendar day-of-month anchor predicts worse: 3.5 days median error against 2.3 for last+gap.
2. **`none` is the main loss, and it behaves like independent per-stream survival.** Among clients with clean live streams, the none rate is 56% with 0 streams, 26% with 1, 15% with 2 and 3% with 3. That fits a per-stream survival of about 0.75. None is also higher when the client already had streams that died (n_dead): 0.24 with none dead, 0.44 with one, 0.71 with two (clients with 1 live stream).
3. **Young streams usually die (strongest new signal).** For the soonest live stream, P(none) is 0.63 at 3 payments, 0.51 at 4, 0.23 at 5 and about 0.10 at 7-9 payments. It rises again to 0.20-0.29 at 11-13 payments, which may be churn around a 12-month mark. Stream age behaves the same way: 64-68% none at 2-3 periods old, 10% at 7-9 periods. Irregular streams also die more (top gap-MAD quintile: 0.41 none), as do less stable amounts (top CV quintile: 0.41 none). If the pipeline's survival model does not use payment count or age as a non-monotone curve, this is a candidate for the missing structure.
4. **Other periods exist.** About 7% of clean streams are biweekly (gap about 14 days, up to 28 payments) and about 9% have gaps of about 45-65 days, which could be bimonthly or monthly with skips. All families have both. Allowing 10-40 day gaps raised the rule from 0.507 to 0.521.
5. **Some labels cannot be reached from history.** 241 of 1403 non-none clients have no clean live stream. For 83 of those, the label family has no family-tagged payment in the history at all (music 19, streaming 18, gym 14, and so on), so the label comes from a new stream after the cutoff. That caps non-none recall at roughly 83-94%.
6. **Music vs streaming on 5812** can only be resolved through ambiguous descriptions ("digital plus", "premium plan") by matching the client's known amount. The per-class F1 is about 0.50-0.58 and is not especially low for either class.
7. **Possible oddity, not verified:** clients whose last-90-day transaction count is above their yearly average (top quintile) have a 38% none rate, against 11-18% otherwise, among clients with a live stream. This could be a generator artefact such as a burst before churn.
