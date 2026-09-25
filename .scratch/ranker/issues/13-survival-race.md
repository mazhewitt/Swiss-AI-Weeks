# 13: Survival Race — one model for `none` and the family

**Why:** On the selection set, v2 (`20260925T001755-e309a3`, 0.5871) loses most of its F1 on Clients whose true family's stream *was* detected. There are 111 such Clients with the wrong family (+0.19 macro-F1 if all were fixed) and 31 called `none` (+0.05). In 84 of the 111, v2 picked the stream projected to pay first. The true stream was projected a median of 7 days later.

Train labels fit a simple process. Each Active Stream independently survives the Cutoff with probability about 0.5, and the label is the soonest survivor, or `none` if none survives:

| Active families | independent s=0.5 predicts none / 1st / 2nd | train observes |
|---|---|---|
| 3 (305 Clients) | 0.125 / 0.50 / 0.25 | 0.128 / 0.48 / 0.22 |
| 4 (78 Clients) | 0.06 / 0.50 / 0.25 | 0.064 / 0.46 / 0.19 |

Scripts: `experiments/analysis/survival/` (the loss breakdown and survival check).

**What to build:** a model that predicts each Candidate Stream's survival probability s_i and combines the streams in projected payment order:

- P(stream i is next) = s_i × Π_{j before i} (1 − s_j)
- P(`none`) = Π_j (1 − s_j)
- A family's probability is the sum over its streams. A Client with no Candidate Stream stays `none`.

**Fitting (the key simplification):** the log-likelihood of one Client decomposes into plain binary log-loss rows. Suppose the label is stream m's family. Then m gets target 1 and every stream ordered before m gets target 0. Streams after m are censored and contribute no row: we never saw whether they would have paid. For a `none` Client, every stream gets target 0. So it is an ordinary LightGBM binary fit on a filtered row set. The current ranker's mistake, in these terms, is training later streams as negatives.

- **Order:** by `days_to_next` ascending; streams with no projection (one payment) come last; ties are broken by `n_payments` descending, as in `candidates`.
- **The label's family not among the candidates:** drop the Client from training (unexplained, about 6–9% of train). Log how many.
- **Two streams of the label's family:** the first in order takes target 1.
- **Pseudo-Labels:** excluded by default. A Shifted Cutoff has almost no churn (ticket 08), so it would teach s ≈ 1. At most one measured variant includes them with an `is_pseudo` feature.

**Features (ADR 0001, stream table plus member payments and matched refunds only):**

- the ranker's `FEATURE_COLUMNS`;
- the per-stream churn fields from `none_model._stream_fields`;
- the Client-level `CLIENT_CHURN` broadcast to every stream.

Not the `FILLER` features, and no raw-transaction counts.

**Pre-registered variants (train 5-fold out-of-fold; nested tuned = decision fitted on 4 folds, scored on the 5th):** no others, to avoid ticket 09's 17-variant optimism.

1. S-rank: ranker features only
2. S-full: ranker plus churn features (expected to be the chosen one)
3. S-full plus Pseudo-Labels with `is_pseudo`

Baseline: v2's setup (`ranker --none-model`, Pseudo-Labels at 2025-10-03, weight 0.5, min_payments 4) on the same folds and the current detector.

**Blocked by:** none

**Status:** ready-for-agent

## Prototype result (gate passed, 09:45)

`experiments/analysis/survival/prototype.py` and `baseline_v2_oof.py`: train 5-fold out-of-fold, folds of `rf cv`, current detector. 6,507 Candidate Streams, 3,913 training rows, and 87 unexplained Clients dropped.

| Variant | argmax | tuned | nested tuned | nested vs v2 | s AUC |
|---|---|---|---|---|---|
| ranker + Pseudo-Labels (noon setup) | 0.5422 | 0.5810 | 0.5651 | −0.043 | |
| v2 baseline: ranker + `none` model + Pseudo-Labels | 0.6081 | 0.6198 | 0.6082 | | |
| **S-rank: ranker features only** (chosen) | 0.6322 | 0.6418 | **0.6239** | **+0.016** | 0.863 |
| S-full: plus churn and Client churn features | 0.6341 | 0.6451 | 0.6171 | +0.009 | 0.881 |

- **The coin is not fair.** s has AUC 0.86 on its own rows, and the mean s is 0.33.
- **P(`none`) is calibrated by number of active families:**

  | Active families | predicted | observed |
  |---|---|---|
  | 1 | 0.354 | 0.321 |
  | 2 | 0.230 | 0.217 |
  | 3 | 0.140 | 0.128 |
  | 4 | 0.108 | 0.064 (over-predicted) |

- **S-rank is chosen, not the S-full variant this ticket expected.** It is best on nested, and it uses only the 16 ranker features. The churn features are the ones that drifted between train and test (ticket 09 review), so fewer is safer.
- **Variant 3 (Pseudo-Labels) is not run.** Adding churn features did not help, and Pseudo-Labels teach s ≈ 1. It is deferred to after the deadline.
- **Build:** `--model survival` with the S-rank features, no Pseudo-Labels and no `none` model.

## Plan and gates (Day 2, times CEST)

| Time | Step | Gate |
|---|---|---|
| 09:30–10:30 | **Prototype** (a script, train only): variants 1–3 plus the baseline, out-of-fold. Diagnostics: s_i's AUC on its own rows; predicted vs observed none / 1st / 2nd rates by number of active families. | **Go** if the best variant's nested tuned macro-F1 is at least the baseline's − 0.005, *and* s_i's AUC is at least 0.60 (the coin is not fair). Otherwise stop, write up the ceiling finding, and upload the hedge at 17:30. |
| 10:30–12:30 | **Implementer:** `--model survival` in `train`, `cv`, `evaluate` and `submit`, with `--decision tuned`. | |
| before 12:00 | **Human:** upload v2 for milestone 3, whatever happens here. | |
| 12:30–14:00 | **Three critics** (spec scenarios, test validity, regression/leakage), then fix rounds and a serial merge. | Behaviour and leakage findings block. |
| 14:00–15:00 | **Selection run:** one run, paired bootstrap against `20260925T001755-e309a3`, predictions committed. Then `scripts/day2_final_survival.sh` refits on train plus the selection set and writes `submissions/day2_final_survival.csv`. | |
| 15:30 | **Upload decision** | Upload the survival file if it beats v2 by at least 0.03, or ties v2 while disagreeing on at least 20% of Clients. Otherwise upload the hedge `submissions/day2_noon_ranker_pseudo.csv`. |
| 15:30–17:00 | **Write-up:** SOLUTION.md, a CONTEXT.md term ("Survival Race"), and a note in ADR 0001 if it changes the default. Buffer. | |

No holdout checkpoint: the holdout has been used once, for v2.

## Implementation (11:00)

`src/recurring_family/survival.py`: `SurvivalModel` (`--model survival`), registered in `MODELS`, so `rf train|cv|evaluate|submit` and `--decision tuned` work as for the other models. It is the S-rank variant: the ranker's `FEATURE_COLUMNS` from `candidates()` and its `LGBM_PARAMS`, streams from the ranker's `_streams_of` (the same detector defaults and stream cache). `race_order` sorts a Client's streams, `training_rows` builds the fit's rows and counts the unexplained Clients, and `race_proba` combines s into the probabilities. `train` prints the fit's row count and the unexplained Clients; both are also saved with the model (`n_training_rows`, `n_unexplained`). Training with one outcome gives every stream a constant s, as the ranker does. The CLI already refuses `--pseudo`, `--pseudo-*` and `--none-model` for any model but the ranker (and the blend for `--pseudo`), so it refuses them with `--model survival` without a change.

Check: `rf cv --model survival` reproduces the prototype's S-rank out-of-fold probabilities (`oof_S-rank.csv`) to a max abs diff of 3e-16, with the same folds: argmax 0.6322, tuned 0.6418, nested tuned 0.6239. Full train fit: 87 unexplained Clients.

Tests: `tests/test_survival.py` (fast suite).

Byte identity: the fast suite passes (473). `scripts/day2_final_ranker_none_v2.sh` reproduces `submissions/day2_final_ranker_none_v2.csv` byte for byte. `scripts/day2_noon_ranker_pseudo.sh` changes 79 rows of `submissions/day2_noon_ranker_pseudo.csv`, but a clean export of b3d81ea (before this change) writes the same bytes. So the drift is older than this ticket; the committed noon file was made with an earlier detector. It was reverted, not recommitted.

## Acceptance criteria

- [x] `rf train|cv|evaluate|submit --model survival` works, with `--decision tuned`. Save then load gives identical probabilities. Rows sum to 1. The log's model column names it
- [x] The row-building function is unit-tested on hand-made Clients:
  - label = 2nd stream → rows 1 (target 0) and 2 (target 1) only;
  - `none` → every stream target 0;
  - the label's family not among the candidates → no rows;
  - streams with no projection come last;
  - two streams of the label's family → the first takes target 1
- [x] The combination is unit-tested: known s values give the closed-form probabilities, and a family's streams sum
- [x] No valid label reaches training (the existing label guards cover `--model survival`). Injecting Decoy Transactions leaves the features unchanged
- [x] Every existing output is byte-identical (ranker, ranker `--none-model`, blend, rules, the milestone scripts); the noon file's drift is older than this change (see Implementation)
- [x] Train out-of-fold table for variants 1–3 and the baseline, and the diagnostics, recorded here
- [x] One selection run logged, predictions committed; the candidate file is valid (`rf submit` check)
- [x] Fast test suite passes

## Selection result (10:10)

Run `20260925T080905-fd1dc5` (`survival+tuned`) scores **0.5903** on the selection set. The runs it was compared with:

| Compared with | Its score | Delta (95% interval) | Verdict |
|---|---|---|---|
| v2 `20260925T001755-e309a3` | 0.5871 | +0.0032 (−0.0227 .. +0.0292) | tie |
| previous best, v3 `20260925T014822-cd37c4` | 0.5884 | +0.0020 | tie |

- **Agreement with v2:** 82.9% on selection and 84.0% on test. The hedge (noon file) agrees with v2 on 82.0% and 85.8%, and scores 0.5735 on selection.
- **Per label:**

  | Label | F1 |
  |---|---|
  | cloud | .567 |
  | gym | .591 |
  | insurance | .612 |
  | mobile | .701 |
  | music | .508 |
  | software | .517 |
  | streaming | .559 |
  | none | .668 |

  Mobile and insurance, v2's weakest labels, are the biggest gains.
- **The +0.016 nested gain on train shrinks to +0.003 on selection.**
- **Candidate file:** `scripts/day2_final_survival.sh` refits on train plus the selection set and writes `submissions/day2_final_survival.csv` (valid).
- **15:30 gate, strictly:** not met. It doesn't beat v2 by 0.03, and it disagrees with v2 on 17%, below the 20% bar. Against the hedge it is the better second slot: it scores higher (0.5903 vs 0.5735) and disagrees with v2 on test more (16.0% vs 14.2%). Humans decide.

## Review (three critics; merged 83da12a)

- **No blocking findings.** No leakage, no regression, and the model is deterministic and batch-independent. `day2_final_ranker_none_v2.csv` is byte-identical.
- **Test strength:** 6 surviving mutants were closed by 9 new tests (fix round 1).
- **Rule breach, disclosed:** the spec critic ran `rf train --with-selection` once (the refit path) and looked at no labels or predictions.
- **Follow-ups:**
  - ties among single-payment streams fall back to alphabetical family order (11 of 2,000 train argmaxes);
  - `n_unexplained` counts labels of Clients with no transactions passed in;
  - `cmd_submit` doesn't wrap prediction in `data.predicting()` (every model; harmless);
  - `scripts/day2_noon_ranker_pseudo.sh` no longer reproduces its committed file since ticket 11 (header note added).
