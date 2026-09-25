#!/usr/bin/env bash
# Day-2 candidate (ticket 13): the Survival Race. Each Candidate Stream gets a survival probability; the
# label is the soonest surviving stream, or none if every stream stops. One model for none and the family,
# on the ranker's stream features only: no Pseudo-Labels, no none model.
#
# On the selection set run 20260925T080905-fd1dc5 (survival+tuned) scores 0.5903 against 0.5871 for
# 20260925T001755-e309a3 (v2): delta +0.0032 (95% -0.0227 .. +0.0292), a tie; the two agree on 82.9% of
# Clients. This refits on train plus the selection set. Whether to upload it is a human decision. Never
# reads the sealed holdout.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_survival.sh
set -euo pipefail

NAME=day2_final_survival

uv run rf train --model survival --with-selection --decision tuned
uv run rf submit --model survival --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
