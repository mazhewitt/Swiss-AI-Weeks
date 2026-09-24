# 01: End-to-end skeleton with a prior-probability model

**What to build:** A team member can fetch the data, train a trivial model, evaluate it and write a valid submission, all through the one CLI. The trivial "prior" model assigns every Client the class-frequency probabilities learned from training labels. Taking the argmax gives all `none`, which is the milestone-1 safety submission. This is the walking skeleton every later ticket plugs into.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Python 3.12 project managed with uv at the repo root, with pandas, LightGBM, scikit-learn and pytest; organisers' README and challenge folder untouched; a short root solution README
- [x] Raw data, interim data and model artifacts are git-ignored; submissions are not
- [x] `fetch-data` unpacks the challenge zip into the raw data area and is idempotent (a second run changes nothing)
- [x] Typed loaders for transactions (UTC timestamps), labels and the sample submission for train, valid, test and unlabeled
- [x] Cutoff (2026-01-01), 90-day Horizon and the eight allowed labels are defined once and imported everywhere
- [x] A model interface in which a model is fitted on labelled Clients and returns a per-Client probability for each of the eight labels; the prior model implements it
- [x] `train --model prior` then `evaluate` reports macro-F1 over the fixed eight labels and per-family F1; a family never predicted scores 0
- [x] Every evaluate run appends one row to the experiment log (date, change description, macro-F1, per-family F1, conclusion)
- [x] `submit` writes a CSV for the test Clients and rejects missing, extra or duplicate Client IDs and labels outside the allowed set
- [x] The all-`none` safety submission is written to the submissions folder and passes the validator
- [x] A small hand-built fixture data folder mirrors the real file layout; all tests run against it through the CLI (Seam 1)

## Comments

- Merged into main as 1ce9af9 (merge of `mvp/01-end-to-end-skeleton`).
- Tests: `uv run pytest -m "not slow"` -> 21 passed.
- Experiment: prior model on valid (1000 Clients): macro-F1 0.0567; all seven subscription families F1 0.0000, none F1 0.4532. Logged to experiments/log.csv (commit 5aa1024).
- All-none safety submission written to submissions/milestone1_all_none.csv and passes `rf submit --check`.
- `fetch-data` verified idempotent on real data (second run wrote 0 files).
