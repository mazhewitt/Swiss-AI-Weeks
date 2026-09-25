#!/usr/bin/env bash
# Day-2 hail mary (ticket 14): the average of v2 and the Survival Race, with self-training on the test Clients'
# transactions if the rehearsal on the selection set chose it. The configuration and whether label-shift EM
# is used come from the committed rehearsal result, experiments/analysis/hailmary/results.json (rule in
# .scratch/ranker/issues/14-hail-mary.md). This refits on train plus the selection set and never reads the
# sealed holdout.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_hailmary.sh
set -euo pipefail

H=experiments/analysis/hailmary/hailmary.py
NAME=day2_final_hailmary

uv run python "$H" oof final v2
uv run python "$H" oof final surv
uv run python "$H" fit final v2
uv run python "$H" fit final surv
CHOSEN=$(uv run python -c "import json; print(json.load(open('experiments/analysis/hailmary/results.json'))['chosen'] or '')")
case "$CHOSEN" in
    st_*)
        read -r REFIT Q < <(echo "$CHOSEN" | sed -E 's/^st_([a-z]+)_q([0-9.]+).*/\1 \2/')
        uv run python "$H" selftrain final "$REFIT" "$Q"
        ;;
esac
uv run python "$H" submit
uv run rf submit --check "submissions/$NAME.csv"
