#!/usr/bin/env bash
# Post-competition regime-aware file (ticket 24), never uploaded. The rehearsal's best candidate (B if none
# beat it; recorded as "chosen" in experiments/analysis/regime_aware/results.json by the `score` stage) is fitted
# on train (as the candidate transforms it) plus all 1,000 valid Clients at weight 3 (ticket 22's unsealed
# domain, labels read inside data.training_run(with_selection=True, with_holdout=True)); E3 on the weight-1
# 5-fold out-of-fold probabilities of those 3,000 Clients; test is predicted. Nothing is scored.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_postmortem_regime.sh
set -euo pipefail

R=experiments/analysis/regime_aware/regime_aware.py
NAME=day2_postmortem_regime
C=$(uv run python -c "import json; print(json.load(open('experiments/analysis/regime_aware/results.json'))['chosen'])")
echo "candidate: $C"

# no stale output can stand in for a failed step
rm -f artifacts/regime_aware/$C/pseudo.pkl artifacts/regime_aware/$C/unsealed/*.csv "submissions/$NAME.csv"
case "$C" in R1|R1R2) uv run python "$R" corrupt ;; esac
uv run python "$R" pseudo "$C"
PIDS=()
for m in v2 surv; do
    uv run python "$R" oof "$C" unsealed "$m" &
    PIDS+=($!)
    uv run python "$R" fit "$C" unsealed "$m" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done  # each fit's own exit status, so set -e sees a failure
uv run python "$R" final
uv run rf submit --check "submissions/$NAME.csv"
