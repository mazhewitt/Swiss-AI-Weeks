#!/usr/bin/env bash
# Day-2 unsealed final (ticket 22): ticket 21's final fit (the hail mary's configuration A, the average of v2
# and the Survival Race with the E3 decision, labelled valid Clients at weight 3 as two duplicates each), with
# the fit set = train plus all 1,000 valid Clients: the user unsealed the 300-Client holdout for training this
# file only. Its labels are read inside data.training_run(with_selection=True, with_holdout=True); nothing is
# scored. E3 is fitted on the weight-1 5-fold out-of-fold probabilities of the 3,000 labelled Clients.
# The last stages apply the ticket's pre-registered rule against ticket 21's file and v2's file.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`, with ticket 21's
# submissions/day2_final_target_weight.csv and submissions/day2_final_ranker_none_v2.csv present:
#     bash scripts/day2_final_unsealed.sh
set -euo pipefail

T=experiments/analysis/target_weight/target_weight.py
NAME=day2_final_unsealed

# no stale output can stand in for a failed fit
rm -f artifacts/target_weight/unsealed/oof_*.csv artifacts/target_weight/unsealed/target_*_w*.csv "submissions/$NAME.csv"
PIDS=()
for m in v2 surv; do
    uv run python "$T" oof unsealed "$m" &
    PIDS+=($!)
    uv run python "$T" fit unsealed 3 "$m" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done  # each fit's own exit status, so set -e sees a failure
uv run python "$T" unsealed
uv run rf submit --check "submissions/$NAME.csv"
uv run python "$T" unsealed-checked
