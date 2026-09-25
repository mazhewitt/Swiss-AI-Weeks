#!/usr/bin/env bash
# Day-2 target weighting (ticket 21): the hail mary's configuration A (the average of v2 and the Survival
# Race, E3 decision) with the 700 labelled selection Clients counted 3 times (two duplicates each) in the
# refit on train plus the selection set. E3 is fitted on the weight-1 out-of-fold probabilities of the
# hail mary's final (artifacts/hailmary/final/oof_{v2,surv}.csv), which this script rebuilds if missing.
# The rehearsal (cross-fitted over two halves of the selection set) and its scoring are in
# experiments/analysis/target_weight/target_weight.py. Never reads the sealed holdout.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_target_weight.sh
set -euo pipefail

H=experiments/analysis/hailmary/hailmary.py
T=experiments/analysis/target_weight/target_weight.py
NAME=day2_final_target_weight

for m in v2 surv; do
    [ -f "artifacts/hailmary/final/oof_$m.csv" ] || uv run python "$H" oof final "$m"
    [ -f "artifacts/hailmary/final/target_$m.csv" ] || uv run python "$H" fit final "$m"
done
[ -f artifacts/target_weight/halves.csv ] || uv run python "$T" halves
rm -f artifacts/target_weight/final/target_*_w*.csv  # no stale output can stand in for a failed fit
PIDS=()
for w in 3 1; do
    for m in v2 surv; do
        uv run python "$T" fit final "$w" "$m" &
        PIDS+=($!)
    done
done
for pid in "${PIDS[@]}"; do wait "$pid"; done  # each fit's own exit status, so set -e sees a failure
for h in h0 h1; do
    for m in v2 surv; do
        [ -f "artifacts/target_weight/$h/oof_$m.csv" ] || uv run python "$T" oof "$h" "$m"
        for w in 1 3; do
            [ -f "artifacts/target_weight/$h/target_${m}_w$w.csv" ] || uv run python "$T" fit "$h" "$w" "$m"
        done
    done
done
uv run python "$T" predict
uv run rf submit --check "submissions/$NAME.csv"
