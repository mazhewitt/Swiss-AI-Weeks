# Data facts (fact-finding pass, 2026-09-24)

Measured on train (2,000 Clients) and confirmed on valid unless noted. The reference scripts in this folder (`load.py`, `streams.py`, `rules.py`) are throwaway analysis code, not the implementation. Use them for behaviour and numbers only.

## Merchant Family mapping (high confidence)

A Merchant Family is identified by MCC plus description. Amount is only needed to separate music from streaming, which share MCC 5812.

| Family | Core descriptions | Home MCC | % of rows on home MCC | Median amount (5–95%) |
|---|---|---|---|---|
| cloud | cloud access/backup, storage plan, service plan | 5732 | 94% | 6.8 (2.5–11.7) |
| gym | urban gym, gym membership, fit club, fitness monthly | 7997 | 95% | 66 (39–93) |
| insurance | cover plan, insurance monthly, safe cover, policy premium | 6300 | 99% | 109 (53–173) |
| mobile | phone contract, service bill, monthly plan, digital plus | 4814 | 97% | 47 (22–73) |
| software | saas billing, productivity suite, software access, premium plan | 5734 | 93% | 39 (14–68) |
| streaming | media streaming, video access, digital plus, premium plan | 5812 | 97% | 17.9 (10–25) |
| music | audio streaming, member pass, digital plus, premium plan | 5812 | 94% | 13.5 (9–19) |

MCC 5812 carries the real streaming and music payments; it is not noise.

- "premium plan"@5734 is software and "digital plus"@4814 is mobile.
- Around 5% of a stream's payments carry a stray MCC, but the amount stays on-stream.
- **Filler Descriptions** ("member plan", "subscription charge", "digital service", partly "monthly plan") appear inside real streams.
- **Decoy Transactions** ("digital order", "merchant charge", "service payment", "card purchase") are spread over about 10 MCCs, are mostly one-offs, and have a median amount of about 67.
- Descriptions are noised with prefixes (pay/billing/member), suffixes (plus/online/digital/core/service), abbreviations (prem, dgtl, mth, prod) and truncations.

## Recurring Streams

- Of streams with 3+ payments, 84% are monthly (a gap of 25–35 days) and about 6% are biweekly (12–16 days). There are no weekly, quarterly or annual streams. Around 2% of gaps are doubled because a payment was missed.
- Timing jitter: the interquartile range is ±2 days, and 90% of gaps fall within ±6 days of the period.
- Amounts are stable: the median coefficient of variation is 1%. Some streams drift in price, with 16% rising 3–20% over their life.
- A Client has between 0 and 7 streams (mostly 1–3). Streams start between Dec 2024 and Oct 2025.
- 99.3% of streams stay in a single currency, so no FX conversion is needed.
- About 2,400 refunds carry subscription descriptions, and some streams are fully refunded yet still labelled with their family (e.g. C000012 is labelled software). Refunds do not end a stream.
- All `transfer` rows are outgoing payments with shop descriptions and are never streams.

## Label semantics (medium confidence)

- For Clients whose label is not `none`, the label's family appears in the history for 94.7% with 1+ payment and 78.5% with 3+ payments. 11% of these labels are a brand-new stream first paid in Nov–Dec 2025.
- Rule "Active Streams with 3+ payments, pick the soonest projected next payment (rolled forward past the Cutoff), else `none`": **macro-F1 0.485 on train and 0.479 on valid**. Other ordering rules do worse.
- `none` Clients: 73% still have Active Streams. Their streams are short (median 4 payments against 8) and start late (Q3–Q4 2025).
  - P(none) is 0.79 when the longest Active Stream has 3 payments, against 0.15–0.20 when it has 7 or more.
  - P(none) is 0.72 when the earliest Active Stream began in Q4 2025, against 0.18 when it began in H1 2025.

## Splits

- train has 2,000 Clients, valid 1,000, test 1,000 (the same IDs as `sample_submission.csv`) and unlabeled 10,000 (IDs start with `U`). No IDs overlap between splits.
- All histories cover 2024-11-07 to 2025-12-31.
- Label shares are similar in train and valid. `none` is about 29–30%, and each family is 9–12%.
- **Decoy shift:** Decoy Transactions per Client are 0.16 in train, 3.1 in valid, 4.5 in test and 0 in unlabeled. Filler Descriptions per Client are 0.31, 1.5 and 2.1 in train, valid and test. See ADR 0001.
