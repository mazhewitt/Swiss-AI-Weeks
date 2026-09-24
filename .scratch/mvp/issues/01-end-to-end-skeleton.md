# 01: End-to-end skeleton with a prior-probability model

**What to build:** A team member can fetch the data, train a trivial model, evaluate it and write a valid submission, all through the one CLI. The trivial "prior" model assigns every Client the class-frequency probabilities learned from training labels. Taking the argmax gives all `none`, which is the milestone-1 safety submission. This is the walking skeleton every later ticket plugs into.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Python 3.12 project managed with uv at the repo root, with pandas, LightGBM, scikit-learn and pytest; organisers' README and challenge folder untouched; a short root solution README
- [ ] Raw data, interim data and model artifacts are git-ignored; submissions are not
- [ ] `fetch-data` unpacks the challenge zip into the raw data area and is idempotent (a second run changes nothing)
- [ ] Typed loaders for transactions (UTC timestamps), labels and the sample submission for train, valid, test and unlabeled
- [ ] Cutoff (2026-01-01), 90-day Horizon and the eight allowed labels are defined once and imported everywhere
- [ ] A model interface in which a model is fitted on labelled Clients and returns a per-Client probability for each of the eight labels; the prior model implements it
- [ ] `train --model prior` then `evaluate` reports macro-F1 over the fixed eight labels and per-family F1; a family never predicted scores 0
- [ ] Every evaluate run appends one row to the experiment log (date, change description, macro-F1, per-family F1, conclusion)
- [ ] `submit` writes a CSV for the test Clients and rejects missing, extra or duplicate Client IDs and labels outside the allowed set
- [ ] The all-`none` safety submission is written to the submissions folder and passes the validator
- [ ] A small hand-built fixture data folder mirrors the real file layout; all tests run against it through the CLI (Seam 1)
