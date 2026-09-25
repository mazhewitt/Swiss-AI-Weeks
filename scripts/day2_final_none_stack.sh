#!/usr/bin/env bash
# Day-2 target-domain `none` stack (ticket 19, .scratch/ranker/issues/19-target-none-stack.md): the hail-mary
# average B (v2 and the hard Survival Race, fitted on train only) with its P(none) replaced by a LightGBM
# stacker on [logit P_B(none)] plus ticket 18's 130 raw features ranked within each domain, fitted on the 700
# selection Clients; the families are rescaled to the remaining mass and the batch decision D is picked on the
# 1,000 test Clients' own probabilities (no labels). The sealed holdout is never read.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_none_stack.sh
set -euo pipefail

D=experiments/analysis/none_stack
NAME=day2_final_none_stack

# the hail mary's train-only rehearsal caches (git-ignored), which base_test.py checks against and E3 is fitted on
H=experiments/analysis/hailmary/hailmary.py
for M in v2 surv; do
    uv run python "$H" oof rehearsal "$M"
    uv run python "$H" fit rehearsal "$M"
done
uv run python "$D/base_test.py" v2    # train-only v2: selection (checked against hail mary's cache) and test
uv run python "$D/base_test.py" surv  # train-only hard race, likewise
uv run python "$D/none_stack.py" predict
uv run python "$D/none_stack.py" submit
uv run rf submit --check "submissions/$NAME.csv"
