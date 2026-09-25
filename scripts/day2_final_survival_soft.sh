#!/usr/bin/env bash
# Candidate (ticket 14): the Survival Race with the soft race order. Each stream's next payment date is
# uncertain (its own schedule spread: sigma = max(3.6, 1.4 x gap_mad_days) days, 6.2 for a single-payment
# stream, calibrated on train plus unlabeled history), and the race is averaged exactly over every order the
# dates can produce. P(none) is unchanged by the order; only the split among detected families moves.
#
# On the selection set run 20260925T102231-5456a7 (survival+monthly-slot+soft+tuned) scores 0.5987 against
# 0.5903 for 20260925T080905-fd1dc5 (the hard race, unprojected-last): delta +0.0084 (95% -0.0156 .. +0.0335),
# a tie, and above the 0.5903 bar fixed before the run, so it replaces that file as the Survival Race
# candidate. This refits on train plus the selection set. Whether to upload it is a human decision. Never
# reads the sealed holdout.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_survival_soft.sh
set -euo pipefail

NAME=day2_final_survival_soft

uv run rf train --model survival --race-order monthly-slot --soft-race --with-selection --decision tuned
uv run rf submit --model survival --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
