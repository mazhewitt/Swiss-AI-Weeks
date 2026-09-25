#!/usr/bin/env bash
# Day-2 candidate v2 (ticket 11): scripts/day2_final_ranker_none.sh on the ticket-11 stream detector, where a
# stray Filler Description payment joins a stream whose amount and schedule it fits.
#
# Same settings as the v1 script (ticket 09): the Stream Ranker with Pseudo-Labels and its Client-level none
# model. On the selection set run 20260925T001755-e309a3 (ranker+none+pseudo+tuned, new detector) scores
# 0.5871 against 0.5764 for 20260924T225229-cc7243 (the same model on the old detector): delta +0.0107
# (95% -0.0105 .. +0.0332), a tie. This refits on train plus the selection set. Whether to upload it is a
# human decision. Never reads the sealed holdout. The v1 script and file are left as they were; the v1
# script now produces a different file too, as the detector changed.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_ranker_none_v2.sh
set -euo pipefail

NAME=day2_final_ranker_none_v2

uv run rf train --model ranker --none-model --with-selection --decision tuned \
    --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 \
    --pseudo-weight 0.5 --pseudo-min-payments 4
uv run rf submit --model ranker --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
