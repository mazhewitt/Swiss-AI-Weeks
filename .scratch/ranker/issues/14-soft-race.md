# 14: Soft race order — average the Survival Race over uncertain payment dates

**Status:** ready-for-human (built, reviewed, selection run done; the upload is a human decision)
**Branch:** `ranker/14-soft-race`
**Blocked by:** none (ticket 13 merged)

## Why

The Survival Race (ticket 13, ADR 0002) fixes each Client's race order from the projected next payment
(`days_to_next`, or a monthly slot for single-payment streams). On the selection set, v2's largest loss is
picking the wrong detected stream: 111 Clients. In 23 of them the true and the picked stream were projected
within 3 days of each other. A monthly stream with a schedule MAD of 2 days can pay either side of such a
gap, so there a hard order is a coin toss dressed as a fact. ADR 0002 lists "a soft race order" as an open
follow-up.

**What a soft order can and cannot change.** In the race, P(`none`) = prod_j (1 - s_j) whatever the
order. So averaging over orders leaves every Client's `none` probability exactly as it is and only
redistributes the family mass among detected streams. The `none` buckets (71 + 31 Clients on selection) are
out of reach; the target is the 111, and within it mostly the 23 close calls. Even fixing all 23 is worth
about +0.04 on selection (the 111 are worth +0.19); a realistic fraction is +0.01, a tie under our 0.03
rule. This ticket is worth doing because it is the more faithful likelihood and it is cheap, not because
selection can show it winning.

## What to build

Each stream's next payment date is a random variable, not a point:

    T_i = mu_i + sigma_i * Z_i,   Z_i independent standard normal

- `mu_i` is what the race uses today: `days_to_next`, or the monthly slot for a single-payment stream
  (the soft race sits on top of `--race-order monthly-slot`; see open question 2).
- `sigma_i` comes from the stream's own schedule jitter, `gap_mad_days` (the median absolute deviation of
  its folded gaps from its period). For a normal, sigma = 1.4826 * MAD. Two-payment streams have MAD 0 by
  construction and three-payment ones a very noisy MAD, so the scale needs a floor:

      sigma_i = sqrt(a^2 + (b * gap_mad_i)^2)          projected streams
      sigma_i = c                                       single-payment streams (monthly slot)

  `a`, `b`, `c` are constants fixed by a calibration on history before any model is fitted (step 1 below),
  and recorded in `survival.py` like `MONTH_DAYS`. They are not tuned on labels.

**Prediction: the exact average over orders.** With independent dates, the probability that stream i is
next, averaged over every order the dates can produce, is

    P(i next)  = s_i * INT f_i(t) * prod_{j != i} (1 - s_j * F_j(t)) dt
    P(`none`)  = prod_j (1 - s_j)                       (unchanged: order-free)

where `f_i`, `F_j` are the normal density and CDF of `T_i`, `T_j`. Given `T_i = t`, each other stream is
before it independently with probability `F_j(t)`, so the product is the expected hard-race factor; the
integral averages over where stream i lands. It is a one-dimensional integral per stream, done by
Gauss-Hermite quadrature (about 20 nodes) and vectorised per Client. This is the "average over many
plausible orders" without Monte Carlo: deterministic, no seed, no sampling noise, and tests can pin closed
forms (two streams with the same `mu` and `sigma` give P(1 next) = s_1 (1 - s_2 / 2); sigma -> 0 recovers
the hard race exactly). A Monte Carlo version (K draws of the dates, hard race per draw, mean) is written
in the prototype as a cross-check only.

**Training: soft-weighted rows.** Today a Client labelled with stream m's family gives m target 1 and every
stream ordered before m target 0; later streams are censored (no row). Under uncertain dates, "before m"
holds for stream j with probability

    w_j = P(T_j < T_m) = Phi((mu_m - mu_j) / sqrt(sigma_m^2 + sigma_j^2))

so stream j gets a target-0 row with sample weight `w_j`, and m keeps its target-1 row at weight 1. This
is the expectation over orders of the hard race's log-likelihood (a Jensen lower bound on the log of the
averaged likelihood), and it is an ordinary weighted LightGBM binary fit. Rows with `w_j` below 1e-3 are
dropped (they would be censored under every plausible order). Unchanged: a `none` Client gives every
stream target 0 at weight 1; other streams of the label's family give no row (the earliest one in the base
order takes the target, as today); a Client whose label family has no Candidate Stream is unexplained and
gives no rows.

**Features:** unchanged, the ranker's 16 stream-table features (ADR 0001). `next_rank` stays the place in
the base (monthly-slot) order. An expected rank, 1 + sum_j P(T_j < T_i), is a natural soft feature but is
not in this ticket (open question 4).

## Pre-registered variants (train 5-fold out-of-fold, folds of `rf cv`, nested tuned macro-F1; no others)

Baselines: V2 `monthly-slot` hard race (0.6323 nested tuned) and V0 `unprojected-last` (0.6239, the
committed candidate).

| Variant | Fit | Predict | What it isolates |
|---|---|---|---|
| **A** soft-predict | today's hard rows (s identical to V2) | soft race | the combination alone |
| **B** soft-fit | soft-weighted rows | soft race | the full soft likelihood |

Diagnostics, all on train out-of-fold:

1. **Close-call ceiling.** Among Clients V2 gets wrong whose true family's stream is detected: how many have
   the true and the picked stream within 3 days, and in how many of those is s_true > s_picked? Only those
   can the soft race flip. This number bounds the gain before anything is fitted.
2. **Net flips.** Among the close-call Clients: fixed minus broken, for A and B.
3. **P(`none`) identity.** A's `none` column equals V2's to 1e-12 (a test, and a check that the exact
   integral is implemented right: the family columns must sum to 1 - P(`none`)).
4. **Argmax changes** vs V2, and a paired bootstrap of nested tuned A and B vs V2 on train.
5. **How many active streams are overdue-and-rolled** (last payment more than one period before the
   Cutoff): the size of open question 3.

## Plan and gates

| Step | What | Gate |
|---|---|---|
| 1. Calibrate jitter | `experiments/analysis/survival/jitter_calibration.py`: for every train stream with 3+ payments, drop its last payment, re-summarise, project, and measure the residual of the real last payment (folded past missed slots). Fit `a`, `b` so that `sigma(gap_mad)` matches the residual spread by MAD bins, and pick `c` from the two-payment streams' residuals. Check the tails: if the residuals are far from normal (kurtosis), switch the family to Laplace, once, before step 2. Reads transactions only, no labels. | Residual spread rises with `gap_mad` (b > 0). If it does not, `sigma` is one constant and the ticket is simpler, not dead. |
| 2. Prototype | `experiments/analysis/survival/soft_race.py`: variants A and B against V2 and V0 on the `rf cv` folds; diagnostics 1-5; results in `soft_race.json`. | **Go** if the better of A/B has nested tuned macro-F1 >= 0.6323 - 0.005 and diagnostic 2 is a net gain. Otherwise write up the finding (ADR 0002 amendment) and stop. |
| 3. Implement | `--soft-race` on `rf train` and `rf cv` (`--model survival` only, refused otherwise); `SurvivalModel(soft=True)`; `variant` reads `+soft`, so the log's model column is `survival+monthly-slot+soft+tuned`; `survival.meta.json` records it; a model file saved before the setting loads as hard. Tests in `tests/test_survival.py`. | Fast suite passes; every existing output is byte-identical (the default is off). |
| 4. Critics | adversarial-critic, lenses spec-scenarios, test-validity and regression-robustness; fix rounds. | Behaviour and leakage findings block. |
| 5. Selection run | one run, paired bootstrap against `20260925T080905-fd1dc5` (V0, 0.5903) and `20260925T085645-c70788` (V2, 0.5862), predictions committed. **Rule, fixed now:** it replaces `submissions/day2_final_survival.csv` only if it scores at least 0.5903 on selection. | |
| 6. Write-up | ticket outcome; SOLUTION.md; ADR 0002 amendment (soft order built, trade-offs, result); CONTEXT.md's Survival Race entry says "in expected payment order" once the soft race is the default, if it is. | |

No holdout checkpoint (used once, for v2). Whether anything is uploaded is a human decision and is not
assumed here.

## Implementation notes

- `survival.py`: `Jitter(a, b, c)` constants and `jitter_scale(rows) -> sigma`; `before_weights(table,
  sigma)` for the fit; `soft_race_proba(table, s, sigma, clients)` for prediction, beside `race_proba`;
  `training_rows` returns weights too (all 1 under the hard race); `fit` passes `sample_weight`.
  `race_order` is unchanged and still supplies the base order and `next_rank`.
- `cli.py`: `--soft-race`, wired like `--race-order` (train, cv, meta, log column, refusal for other models).
- Tests to pin: sigma -> 0 equals the hard race to 1e-12; equal `mu`/`sigma` pair gives the closed form;
  P(`none`) is order-free; rows sum to 1 for random s and sigma; the soft weights (m weight 1 target 1, a
  stream far earlier ~1, far later dropped, same date 0.5, `none` Client all 1); unexplained count
  unchanged; save/load round trip; old save loads hard; flag reaches fit and every tuned fold; refused for
  other models; the log names it; the quadrature agrees with a seeded Monte Carlo to 1e-3.

## Open questions for discussion

1. **Exact quadrature or Monte Carlo?** Recommended: exact. Same answer, deterministic, tests pin closed
   forms. Monte Carlo stays in the prototype as a cross-check.
2. **Which base order?** Recommended: soft on top of `monthly-slot` only (every stream needs a date to
   jitter), refusing `--soft-race` with the other two orders. `unprojected-last` can be expressed (single-
   payment streams at +infinity, hard ties among them) but it is extra branches for a variant that ties.
   Note the committed candidate is V0 `unprojected-last`, so the selection comparison is against both runs.
3. **Overdue streams.** An active stream whose projection fell before the Cutoff is rolled a full period
   forward (a monthly stream 2 days overdue races at day 28). Jitter says it may instead pay late, on day
   1 or 2. A faithful `T_i` would put some mass just after the Cutoff. Recommended: not in this ticket;
   diagnostic 5 measures how many streams this touches, and it becomes ticket 15 if the count is large.
4. **Expected rank as a feature?** Recommended: no; keep the 16 features fixed so A/B isolate the race.
5. **Calibration source.** Recommended: train transactions only (no labels are read either way).
   Adding the unlabeled split's transactions would tighten `a`, `b`; not needed for a first pass.
6. **Where the effort ceiling is.** If diagnostic 1 says fewer than ~20 train Clients are flippable, the
   soft race is a likelihood nicety and we should say so in ADR 0002 rather than run selection.

## Acceptance criteria

- [x] `jitter_calibration.py` run; `a`, `b`, `c` and the tail check recorded here and as constants
- [x] Prototype table for A, B, V2, V0 and diagnostics 1-5 recorded here; gate decision stated
- [x] `rf train|cv|evaluate|submit --model survival --soft-race` works with `--decision tuned`; save then
      load is identical; rows sum to 1; the log's model column names it
- [x] Unit tests listed above pass; the hard race's tests are untouched and pass
- [x] Every existing output is byte-identical with the flag off (`day2_final_survival.sh`, v2 script). The rule covers predictions, submissions and the log; the untracked `artifacts/survival.json` and its meta gain `soft`/`soft_fit`/`soft_race` keys (false) with the flag off
- [x] One selection run logged, predictions committed; the candidate rule applied as fixed above
- [x] Fast suite passes; ADR 0002 amended

## Step 1: jitter calibration (done)

`experiments/analysis/survival/jitter_calibration.py [splits]`: for every stream with 3+ payments, drop its
last payment, re-summarise the head as the detector does, project `head[-1] + period`, and measure the
residual of the real last payment folded past missed slots. It runs under `data.pseudo_labelling()`, which
refuses every label file. Run on train (4,344 streams; `jitter_calibration.json`) and on train plus the
unlabeled split (27,101 streams; `jitter_calibration_train+unlabeled.json`, the adopted one).

| MAD of the head's folded gaps | streams | robust sigma of the residual (days) |
|---|---|---|
| 0 (a head of 2 payments: 1 gap) | 4,913 | 6.2 |
| 0-0.5 | 1,791 | 3.9 |
| 0.5-1 | 3,100 | 3.5 |
| 1-2 | 8,343 | 3.6 |
| 2-3 | 5,687 | 3.7 |
| 3-5 | 2,782 | 4.6 |
| over 5 | 485 | 9.4 |

- **The spread is flat at about 3.6 days until the MAD passes about 2.6, then rises with it.** The
  pre-registered `sqrt(a^2 + (b MAD)^2)` underfits that shape (weighted rms 0.45 days); `max(a, b MAD)` fits
  it twice as well (0.20). Adopted: **sigma = max(3.6, 1.4 x gap_mad_days)**, and **6.2 days for a
  single-payment stream** (the spread of two-payment heads' projections, the closest proxy). The train-only
  run gave 3.5 / 1.46 / 5.8: the unlabeled split moved the constants by a tenth of a day.
- **Why the residual is much wider than the within-history MAD:** the projection carries the last payment's
  own jitter and the period estimate's error as well as the new payment's jitter, so a stream with MAD 1.5
  still misses its projection by 3.6 days typically. The residual, not the MAD, is the right sigma for T_i.
- **Tail shape:** standardised residuals have 67.9% within 1 sigma (normal: 68.3%) but 92.7% within 2
  (normal 95.4%, Laplace 94.0%) and excess kurtosis 7.6. The core is normal and the tails hold outliers
  (misfolded slots, a misassigned payment). Race order near a tie is about the core, so the density stays
  **normal**; the pre-registered switch to Laplace is not taken, and this is a judgement call recorded here.
- By period: biweekly 3.6, monthly 4.2 days (robust). One sigma model for both.

## Step 2: prototype (gate passed for A; B fails)

`experiments/analysis/survival/soft_race.py`: train 5-fold out-of-fold on the `rf cv` folds, current
detector, no valid labels. Baselines V2 (`monthly-slot`) and V0 (`unprojected-last`). Results in
`soft_race.json`; out-of-fold probabilities in `oof_soft_{V0,V2,A,B}.csv`.

| Variant | argmax | tuned | nested tuned | nested vs V2 (paired bootstrap, 95%) | argmax changed vs V2 |
|---|---|---|---|---|---|
| V0 `unprojected-last` (the committed candidate) | 0.6322 | 0.6418 | 0.6239 | -0.0084 (-0.024 .. +0.008) | 212 |
| V2 `monthly-slot` hard race | 0.6343 | 0.6503 | 0.6323 | | 0 |
| **A: soft race over V2's fit** | 0.6346 | 0.6530 | **0.6387** | **+0.0064 (-0.007 .. +0.019)** | 97 |
| B: soft race over the soft-weighted fit | 0.6273 | 0.6448 | 0.6227 | -0.0097 (-0.025 .. +0.004) | 190 |

The first run used 24 quadrature nodes; the rerun with the final 48 gives every figure above and below identically.

Diagnostics (train out-of-fold, V2 argmax):

1. **Close-call ceiling.** V2 picks the wrong detected family for 334 Clients. The true and the picked
   stream race within 3 days in 75 of them, and in 30 of those the true stream has the higher s: the ones a
   soft order can flip. Within 7 days: 164, of which 68. The gap's median is 7.2 days (quartiles 3.3 and
   14.4). In 56 of the 334 the true stream raced first and still lost, so s, not the order, was wrong there.
2. **Net flips.** A (argmax): 39 fixed, 38 broken overall; among the 75 close calls **24 fixed, 0 broken**.
   A (nested tuned): 76 fixed, 60 broken; close calls 20 fixed, 1 broken. B (argmax): 73 fixed, 84 broken;
   close calls 16 fixed, 0 broken. B (nested): 87 fixed, 100 broken.
3. **P(`none`) is order-free:** max |P(`none`) A - V2| = 3e-16 over 2,000 Clients.
4. Above: 97 argmax changes for A; the paired bootstrap of nested tuned A vs V2 is +0.0064, a tie under
   the 0.03 rule, and the gate (>= 0.6323 - 0.005 and a net gain on the close calls) is met.
5. **Overdue-and-rolled Active Streams:** 319 of 3,214 (9.9%), touching 305 of the 1,700 train Clients
   with an Active Stream. That is large enough for open question 3 to become **ticket 15**.

**Decision after the prototype (a change from the plan):** `soft=True` is variant A, the soft race over the
hard fit; the soft-weighted fit is `soft_fit=True`, reachable from Python only, with no CLI flag. B's loss
is consistent: it fixes fewer close calls than A and breaks more elsewhere. A plausible reason is that a
target-0 row at weight P(T_j < T_m) for a stream due *after* the label's stream teaches "stopped" to streams
that were most likely alive, so s falls where it should not. It is recorded, not pursued.

## Step 3: implementation

- `survival.py`: `race_slot` (the day each stream races; `race_order` now sorts by it, unchanged
  behaviour), `jitter_scale`, `_padded` + `_soft_next` (the Gauss-Hermite average over orders, 48 nodes,
  chunks of 256 Clients), `soft_race_proba` (P(`none`) exactly prod (1 - s); the quadrature's family mass
  scaled to 1 - P(`none`), which corrects at most 5e-8), `soft_training_rows` (the soft fit's rows and
  weights), `SurvivalModel(soft=, soft_fit=)`. Saved as `soft` and `soft_fit`; a file without them loads
  hard. `variant` reads `+soft` (or `+soft-fit`), so the log's model column is
  `survival+monthly-slot+soft+tuned`.
- `cli.py`: `--soft-race` on `train` and `cv`; refused for other models and with any order but
  `monthly-slot`; `survival.meta.json` records `soft_race`.
- Tests (`tests/test_survival.py`, "ticket 14"): vanishing jitter gives the hard race; two streams due
  the same day split evenly (closed form); a stream far ahead races first; P(`none`) exactly the hard
  race's and rows sum to 1; the quadrature's family mass equals 1 - P(`none`) to 1e-6 before scaling; the
  quadrature agrees with a 200,000-draw Monte Carlo over orders to 3e-3; `jitter_scale`; the soft weights
  (hand Client); soft rows equal the hard rows as jitter vanishes; the soft fit passes its weights to
  LightGBM and the soft race proper passes none; the refusals; save/load and an old file; the flag reaches
  the model, every fold, the meta and the log; the CLI refusals.
- The implementation notes' `before_weights` and a weight-returning `training_rows` became `soft_training_rows` beside an untouched `training_rows`, chosen by `SurvivalModel._training_rows`: the hard path's code is unchanged.
- Fast suite: 521 passed. `bash scripts/day2_final_survival.sh` on this branch rebuilds `submissions/day2_final_survival.csv` byte for byte (the hard race, unprojected-last, is unchanged).

## Step 4: review (three critics, adversarial-critic on Opus)

- **Regression and leakage:** nothing blocking. Predictions at main and on the branch are byte-identical for
  every race order (booster strings, `n_training_rows`, `n_unexplained` equal); the calibration reads
  transactions only under `data.pseudo_labelling()`; the prototype reads train labels only; the soft race
  reads stream features only at predict time. Nits taken: a finiteness guard in `soft_race_proba` (a NaN
  date would have made the whole Client `none` silently), and the products and sums over a Client's
  streams are now accumulated one stream at a time, so a Client's probabilities are bit-identical alone and
  in any batch (a blocked reduction over the padded width moved the last bit). The untracked model and
  meta JSON gain `soft`/`soft_fit`/`training_weight`/`soft_race` keys with the flag off; the byte-identity
  rule covers predictions, submissions and the log.
- **Spec scenarios:** nothing blocking. Should-fix taken: the calibration's departures from the plan (the
  `max` form over the `sqrt` form, the normal density kept despite heavy tails, train plus unlabeled as the
  source) and the prototype's gate decision are recorded above. Nits taken: the quadrature comment no
  longer claims per-stream accuracy (it is about 1e-7 per stream while no spread is more than 4x another,
  which holds on this data: at most 2 of 1,938 train Clients exceed it); `training_weight` is saved, so a
  loaded soft-fit model's `summary()` works; the Monte Carlo test is at 1e-3 and the vanishing-jitter test
  at 1e-12, as the plan said.
- **Test validity (mutation run):** 11 mutants killed, 5 equivalent. One **blocking** survivor: with
  `predict_proba`'s soft branch disabled every test still passed, because no test showed the model's
  prediction *is* the soft race. Closed by a test on a hand table with the survival probabilities patched:
  the model's output equals `soft_race_proba` exactly, differs from the hard race, keeps `none`, and a
  single-payment stream races at its monthly slot with the 6.2-day spread (an even split with a projected
  stream due the same day: the closed form). Also closed: unequal spreads in the weight test (the pairwise
  `sqrt(sigma_m^2 + sigma_j^2)`), and a same-family later stream that would carry weight 0.35 gives no row
  (before, it sat 30 days out and the 1e-3 threshold dropped it for the wrong reason). Left untested, on
  purpose: the exact `>=` at `MIN_SOFT_WEIGHT`.
- The test critic restored file snapshots while it worked, which overwrote one of this round's source
  patches; it was re-applied and the tree checked (`git diff`), and the suite re-run.
- Fast suite: 524 passed.

## Step 5: selection run (12:22 UTC)

`rf cv --model survival --soft-race` reproduces the prototype's variant A out-of-fold probabilities to a max
abs diff of 4e-16 (argmax 0.6346, tuned 0.6530, nested tuned 0.6387).

Run **`20260925T102231-5456a7`** (`survival+monthly-slot+soft+tuned`) scores **0.5987** on the selection set.
`rf compare` on the four candidates (`submissions/day2_final_compare_soft.md`):

| Compared with | Its score | Delta (95% interval) | Verdict |
|---|---|---|---|
| hard race, unprojected-last `20260925T080905-fd1dc5` (the committed candidate) | 0.5903 | +0.0084 (−0.0335 .. +0.0156, as their minus ours) | tie |
| hard race, monthly-slot `20260925T085645-c70788` | 0.5862 | +0.0125 | tie |
| v2 `20260925T001755-e309a3` | 0.5871 | +0.0116 | tie |

The soft race is the compare's winner: the simplest candidate no other beats. Per label, against the
committed candidate: cloud .567 → .604, gym .591 → .584, insurance .612 → .621, mobile .701 → .730, music
.508 → .464, software .517 → .531, streaming .559 → .564, none .668 → .691.

- **By the rule fixed before the run (at least 0.5903), it replaces the candidate.**
  `scripts/day2_final_survival_soft.sh` refits on train plus the selection set and writes
  `submissions/day2_final_survival_soft.csv`; `day2_final_survival.csv` and its script stay as the record
  of the hard race. The upload is a human decision.
- **As predicted, it reads as a tie.** The whole gain is inside the wrong-stream bucket, and `none` F1 moved
  only through the decision layer's re-tuning (P(`none`) itself is identical).
- **Candidate file:** `submissions/day2_final_survival_soft.csv` (valid, 1,000 Clients). On test it agrees
  with the hard-race candidate on 88.6% of Clients and with v2 on 83.5% (the hard race and v2: 84.0%). Its
  `none` share is 23.6% against the hard race's 23.5%: the order moves no `none`. It predicts streaming
  (12.0% vs 9.1%) and software (11.1% vs 9.0%) more, music (7.2% vs 9.9%) and mobile (12.5% vs 13.7%) less.
- **Ticket 15** (overdue streams may pay late) is opened from diagnostic 5.
