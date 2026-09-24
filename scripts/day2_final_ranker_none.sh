#!/usr/bin/env bash
# Day-2 17:30 candidate (ticket 09): the Stream Ranker with Pseudo-Labels and its Client-level none model.
#
# The setup was chosen on train out-of-fold only (scripts/none_model_variants.py: nested tuned macro-F1
# 0.5699 -> 0.6095). On the selection set run 20260924T225229-cc7243 (ranker+none+pseudo+tuned) scores
# 0.5764, a tie with 20260924T171108-3251a4 (0.5735, the Day-2 12:00 upload): delta +0.0029
# (95% -0.0221 .. +0.0267). This refits the same settings on train plus the selection set. Whether to
# upload it is a human decision. Never reads the sealed holdout.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_ranker_none.sh
set -euo pipefail

NAME=day2_final_ranker_none

uv run rf train --model ranker --none-model --with-selection --decision tuned \
    --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 \
    --pseudo-weight 0.5 --pseudo-min-payments 4
uv run rf submit --model ranker --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
