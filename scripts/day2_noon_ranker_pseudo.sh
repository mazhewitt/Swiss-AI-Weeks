#!/usr/bin/env bash
# Day-2 12:00 milestone upload: the Stream Ranker with Pseudo-Labels.
#
# rf compare (submissions/day2_noon_rules_gate4.md) found a three-way tie on the selection set,
# which ticket 06 resolved to the simpler gated rule. That rule's test predictions equal the
# milestone-2 file, and team rank is the best of any milestone, so this milestone uploads the
# highest-scoring candidate instead: run 20260924T171108-3251a4 (ranker+pseudo+tuned, 0.5735),
# refit with the same settings on train plus the selection set. Never reads the sealed holdout.
#
# Reproduces the committed file only at commit 82f79d3 or earlier: ticket 11's detector (Stray Payments,
# 16588eb) changes 79 rows, and the ranker takes no --param to switch it off. Upload the committed file.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_noon_ranker_pseudo.sh
set -euo pipefail

NAME=day2_noon_ranker_pseudo

uv run rf train --model ranker --with-selection --decision tuned \
    --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 \
    --pseudo-weight 0.5 --pseudo-min-payments 4
uv run rf submit --model ranker --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
