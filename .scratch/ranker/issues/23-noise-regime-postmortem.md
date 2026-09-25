# 23: Post-mortem — is the train→valid/test regime change description and MCC noise?

Status: done

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

Code and numbers are in `experiments/analysis/noise_regime/`: `schedule.py`, `measure.py`, `noise_params.json` (committed in ed0934d before step 2 ran), `regime.py` and `results.json`.

**Step 1 — the noise, measured on payments on a stream's schedule** (stream members plus payments fitting an empty slot, with chance hits subtracted using a half-period-off control):

| | train | valid | test |
|---|---|---|---|
| Description not naming the family | 0.057 | **0.387** | **0.534** |
| … of which Filler ("member plan", "subscription charge", "digital service", "monthly plan") | 0.050 | 0.362 | 0.513 |
| MCC not the family's home MCC | 0.038 | 0.121 | 0.153 |
| Dispersion of description noise per stream (1 = independent per payment) | 1.71 | 1.03 | 1.02 |
| Share of streams with a single payment | 0.179 | 0.302 | 0.300 |
| Active Streams per Client | 1.61 | 1.38 | 1.37 |

- **In valid and test, about half of real subscription payments carry a Filler Description, independently per payment,** and about 15% carry an alternative MCC, independently of the description.
- **In train, the little noise there is clusters on a few Clients.** That is why it marked `none` there (ticket 11).
- The noisy MCCs are a fixed set of alternatives per family. Insurance has none.

**Step 2 — train corrupted to test's noise level** (per family: c = (test − train)/(1 − train); draws from test's own noise distributions; 0.5× is a subset of 1×; labels unchanged), scored with configuration A's nested tuned E3 on train out-of-fold:

| Dose | Nested macro-F1 | Change vs clean (paired bootstrap 95%) |
|---|---|---|
| 0 (clean) | 0.631 | — |
| 0.5× | 0.621 | −0.011 (−0.025 .. +0.004) |
| 1× | **0.594** | **−0.037 (−0.054 .. −0.020)**; all 5 folds below clean |

- **The pre-registered prediction held.** Corrupting train to test's noise level drops it to our test level (0.594 against 0.59–0.61 on selection and test), and 0.5× lands in between.
- **Stream-feature drift** (train vs test adversarial AUC): 0.725 clean → 0.643 at 1×. Noise explains about a third of the drift above chance. What remains is led by `stream_first_days`, `min_days_since_first`, `stream_activity_ratio60` and `n_ended`.
- **Checks:**
  - at dose 0, the out-of-fold probabilities equal the hail mary's caches exactly;
  - the stream and Pseudo-Label caches (keyed on file size and mtime) were bypassed, and the train Pseudo-Label `none` share moves 0.145 → 0.160 → 0.183 with the dose;
  - at 1× the corrupted train measures 0.496 / 0.133 noise, slightly under test.
- **Caveats:**
  - one corruption seed;
  - the models are trained and scored on the same corrupted data, so this measures the information lost to noise, not the cost of training clean and predicting noisy;
  - valid is less noisy than test, yet v2 scored lower on selection than on test (within sampling error);
  - the clean baseline is configuration A's 0.631; the ticket's "about 0.64" was the soft-race stack.
- **Process note:** the harness refused a subagent's write of `findings.md` (subagents return text rather than write report files). The agent stopped as instructed, and this Result carries the findings instead.
