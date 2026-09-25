#!/usr/bin/env bash
# Day-2 candidate v3 (ticket 12): scripts/day2_final_ranker_none_v2.sh on the ticket-12 stream detector, where a
# Decoy-described payment also joins a stream under the Stray Payment rules (`StreamParams.join_decoys`).
#
# Same settings as the v1 and v2 scripts: the Stream Ranker with Pseudo-Labels and its Client-level none model.
# On the selection set run 20260925T014822-cd37c4 (ranker+none+pseudo+tuned, ticket-12 detector) scores 0.5884
# against 0.5871 for 20260925T001755-e309a3 (the same model on the ticket-11 detector): delta +0.0012
# (95% -0.0187 .. +0.0205), a tie. This refits on train plus the selection set. Whether to upload it is a human
# decision. Never reads the sealed holdout. The v1 and v2 scripts and files are left as they were.
#
# Reproducible at commit 73ec6a5 only, where `join_decoys` defaulted on. Afterwards the default went back to off
# (the ticket-11 detector; see ADR 0001 and ticket 12), and the ranker commands take no stream parameters, so at
# later commits this script runs v2's commands on v2's detector (not checked byte for byte) and does not
# reproduce submissions/day2_final_ranker_none_v3.csv.
#
# Run from the repo root at commit 73ec6a5, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_final_ranker_none_v3.sh
set -euo pipefail

NAME=day2_final_ranker_none_v3

# refuse to run where join_decoys is off: that would overwrite the v3 file with v2's predictions
uv run python -c "import sys; from recurring_family.streams import StreamParams; sys.exit(0 if StreamParams().join_decoys else 'join_decoys is off at this commit: run this script at commit 73ec6a5')"

uv run rf train --model ranker --none-model --with-selection --decision tuned \
    --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 \
    --pseudo-weight 0.5 --pseudo-min-payments 4
uv run rf submit --model ranker --decision tuned --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"
