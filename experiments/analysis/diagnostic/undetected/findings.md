# Never-detected labels and the `none` bucket (train only)

Scripts run in order: `prep.py` → `look.py` (writes tx.pkl) → `pairs.py`, `sibling*.py`, `newness.py`, `none_bucket.py`, `odd.py` (writes members.pkl) → `odd_time.py`, `odd_stream.py`, `impact.py`, `ceiling.py`. Only the train transactions and train labels are read. The OOF probabilities come from `experiments/analysis/survival/oof_*.csv`.

## 1. The never-detected labels have no pre-Cutoff evidence and can't be recovered
- 87 of 2000 train Clients (4.35%, or 6.2% of family labels) have a label family with no Candidate Stream. By family: gym 16, streaming 15, music 14, insurance 14, cloud 11, software 10, mobile 7.
- Over (Client, undetected family) pairs, the base rate is 87/9085 = 0.96%. 71% of the positives have no transaction of that family at all: no payment, Decoy, refund or fee, on any MCC or in any currency.
- The best rules are weak. A leftover payment of the family in the last 30 days has precision 2.7% and covers 10% of these labels. A refund in the last 14 days has precision 4.6% and covers 3%. Single payments, Decoys on the home MCC and refund-only "phantom" series are all at the base rate.
- 15 of the 29 undetected music/streaming labels have a stream of the sibling family. But an unhinted 5812 stream with n≤2 is the sibling family's label only 9 times in 173 (5%). Relabelling it would hurt.
- Ceiling: fixing all 87 would give +0.049 (S-full OOF, 0.634 → 0.682). Sending the wrong-family predictions to `none` gives +0.006. These are streams that start after the Cutoff, so the loss can't be avoided on train.

## 2. Top finding: in the months before the Cutoff, truth-`none` Clients' stream payments turn "odd"
- "Odd" means the payment has an off-home MCC, a Filler Description, or a variant description (with a suffix, prefix or abbreviation).
- Share of stream-member payments that are odd, by month before the Cutoff: truth-`none` Clients 38–41% (months 0–4) and 28–32% (5–6), then 8–14% from month 7 onward. Family-labelled Clients stay at 9–15% throughout. The shift starts around June 2025.
- Client-level rules on the recent odd share (last 6 months of stream payments):
  - rec_odd ≥ 0.4: 328 Clients, 82% `none`, 45% of all `none`;
  - rec_odd ≥ 0.5: 238 Clients, 86% `none`;
  - an Active Stream whose last payment is off its home MCC: 123 Clients, 71.5% `none`.
- AUC for `none`: rec_odd 0.80, recent-minus-old contrast 0.76.
- Nested 5-fold OOF: a logistic stack of logit(P_none) plus these features, with the `none` threshold tuned inside the folds:
  - survival race (S-full): 0.6262 → 0.6545 (+0.028); `none` AUC 0.911 → 0.931;
  - v2: 0.6086 → 0.6186 (+0.010);
  - contrast-only features (robust to a split-wide level shift): +0.019 (S-full), +0.008 (v2).
- The none model today excludes description and MCC statistics (none_model.py docstring, ADR 0001), so this signal is unused.
- It is probably the same phenomenon as the "hidden series" in churn/README (99% `none`): odd payments that fail to join a stream.
- Risk: ticket 11 says valid and test book many real stream payments as strays, which are odd by this definition. The absolute level may shift, so prefer the recent-minus-old contrast. Check this label-free: compare the odd rate by month in valid and test.

## 3. Secondary signals
- Within family-labelled Clients, the recent oddness of each stream barely separates the label stream (AUC 0.53). A stream with rec_odd > 0.4 carries the label 20% of the time, against 40–44% otherwise. It is a small survival feature.
- Classic churn signals are weak:
  - a final refund: 25% `none` against 20%;
  - overdue in the top quintile: 28% against 18–20%;
  - an amount change on the last payment in either tail: 28% against 15–16%;
  - a currency change: only 6 streams.
- New streams already carry the label often: a stream with first payment ≤35 days ago carries it 36% of the time. The ranker already has days_since_first.
