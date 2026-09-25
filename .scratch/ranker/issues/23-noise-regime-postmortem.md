# 23: Post-mortem — is the train→valid/test regime change description and MCC noise?

Status: ready-for-agent

**Why:** after the final submission the user learned how 0.68 was reached: there is a regime change in the data. Valid and test are noisier in their descriptions and MCCs. Our stream detector decides a payment's family from its description and home MCC, so noise would break streams apart, fake missed slots (which look like churn), and misassign families. This ticket tests that explanation after the fact. Nothing here can change the leaderboard.

## 1. Measure the noise (label-free)

- Run the existing stream detector on train, valid (all 1,000 Clients, transactions only) and test.
- For payments that belong to a stream's schedule, meaning in the stream or fitting its amount and date slot (ticket 11/12 tolerances), measure per split:
  - **description noise:** the share whose description does not name the stream's family, split into Filler, Decoy, another family's words, and other;
  - **MCC noise:** the share whose MCC is not the family's home MCC;
  - stream fragmentation: missed-slot rate and payments per stream.
- From the difference between test and train, derive per-payment corruption rates, plus the empirical distributions the noisy descriptions and MCCs are drawn from. **These are fixed and committed before step 2 runs.**

## 2. Recreate the regime in train

- Corrupt train's transactions at the measured rates: independent per payment, seed 0, with descriptions and MCCs replaced by draws from the measured noise distributions. Labels are unchanged.
- **Doses:** 0 (clean), 0.5× and 1× the measured rates.
- For each dose, run hail mary configuration A's train pipeline (v2 plus the hard race, 5-fold out-of-fold) and score nested tuned E3, as in ticket 17's Gate A.
- **Prediction (fixed now):** if the noise explains the train→test gap, 1× drops the nested macro-F1 from about 0.64 toward our test level (about 0.60), and 0.5× lands in between. If 1× costs less than 0.01, the noise is not the explanation.
- Also report the stream-feature drift: the train-vs-test adversarial AUC of the stream features (the v2 `none` model's), clean against 1×.

## Rules

- Train labels only; no valid or test labels are read.
- Code goes in `experiments/analysis/noise_regime/`, with caches git-ignored.

## Result

(to fill)
