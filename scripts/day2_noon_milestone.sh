#!/usr/bin/env bash
# Day-2 12:00 milestone submission (ranker spec, ticket 06).
#
# Picks the best of the three logged selection-set candidates by paired bootstrap (deltas under
# 0.03 are ties; a tie goes to the simpler candidate), refits the winner on train plus the
# selection set, writes the test submission and validates it. The comparison is written next to
# the submission as submissions/$NAME.md. Never reads the sealed holdout.
#
# The rule is refit with `--param join_strays=false`: the stream detector as it was when this upload was
# made, before ticket 11 let stray Filler Description payments join a stream on its schedule. So the
# current code writes the committed submissions/$NAME.csv byte for byte.
#
# Run from the repo root, after `uv run rf fetch-data` and `uv run rf split`:
#     bash scripts/day2_noon_milestone.sh
set -euo pipefail

NAME=day2_noon_rules_gate4
RULES_GATE4=20260924T151704-615cf5    # milestone-2 rule: E1 plus the none-gate at 4 payments
RANKER=20260924T154434-ee87f3         # Stream Ranker, tuned E3 decision
RANKER_PSEUDO=20260924T171108-3251a4  # Stream Ranker with Pseudo-Labels, tuned E3 decision

# 1. choose, simplest candidate first
comparison=$(uv run rf compare --run "$RULES_GATE4" --run "$RANKER" --run "$RANKER_PSEUDO" \
    --out "submissions/$NAME.md" --title "Day-2 12:00 milestone: candidate comparison")
echo "$comparison"
if ! grep -q "winner: $RULES_GATE4 " <<<"$comparison"; then
    echo "the winner is no longer $RULES_GATE4: refit and submit the new winner instead" >&2
    exit 1
fi

# 2. refit the winner on train plus the selection set, 3. write and validate the submission
uv run rf train --model rules --none-gate --with-selection --param join_strays=false
uv run rf submit --model rules --name "$NAME"
uv run rf submit --check "submissions/$NAME.csv"

cat >>"submissions/$NAME.md" <<EOF

## Submission

\`submissions/$NAME.csv\`: the winner, the milestone-2 rule (soonest Active Stream due within the
Horizon, \`none\` when the Client's longest such stream has at most 4 payments), refit on train plus
the selection set and applied to the 1,000 test Clients. The rule learns nothing from labels, so the
refit only checks the labels are allowed; its test predictions equal the milestone-2 submission's.

Made by \`bash scripts/day2_noon_milestone.sh\`, which runs:

    uv run rf compare --run $RULES_GATE4 --run $RANKER --run $RANKER_PSEUDO \\
        --out submissions/$NAME.md --title "Day-2 12:00 milestone: candidate comparison"
    uv run rf train --model rules --none-gate --with-selection --param join_strays=false
    uv run rf submit --model rules --name $NAME
    uv run rf submit --check submissions/$NAME.csv

The sealed holdout was not read: the comparison uses the selection-set runs above and the refit
uses train plus the selection set only.
EOF
