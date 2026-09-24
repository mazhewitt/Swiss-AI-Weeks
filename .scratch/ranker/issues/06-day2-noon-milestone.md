# 06: Day-2 12:00 milestone submission

**What to build:** Pick the best of the rules baseline with the gate, the ranker, and the ranker with Pseudo-Labels on the selection set, using paired bootstrap (deltas under 0.03 are ties; on a tie prefer the simpler candidate). Refit the winner on train plus the selection set, write the test submission, validate it and commit it with the exact command that produced it. The human uploads it.

**Blocked by:** 01, 03, 05

**Status:** done

- [x] A short comparison of the three candidates' selection scores and bootstrap intervals is written next to the submission
- [x] The submission has 1,000 test Clients, allowed labels only, and passes `submit --check`
- [x] The file and the command that produced it are committed
- [x] The sealed holdout is not read

## Resolution

- New `rf compare --run RUN --run RUN ... [--out MD]` (candidates simplest first): each run's committed predictions scored on the same Clients, each candidate's macro-F1 with a 95% bootstrap interval, every pair compared by paired bootstrap (deltas under 0.03 are ties), winner = the simplest candidate no other candidate beats. It refuses sealed-holdout runs, mixed splits and runs without saved predictions, and reads selection labels only. Tests: `tests/test_compare_candidates.py`.
- Comparison on the 700-Client selection set (`submissions/day2_noon_rules_gate4.md`): rules+gate4 20260924T151704-615cf5 0.5490 (95% 0.5106 .. 0.5848); ranker+tuned 20260924T154434-ee87f3 0.5714 (0.5315 .. 0.6071); ranker+pseudo+tuned 20260924T171108-3251a4 0.5735 (0.5344 .. 0.6075). Paired deltas: ranker vs rule +0.0224 (-0.0111 .. +0.0565), ranker+pseudo vs rule +0.0245 (-0.0072 .. +0.0575), ranker+pseudo vs ranker +0.0021 (-0.0272 .. +0.0306): all ties, so the simplest candidate, the gated rule, wins.
- `submissions/day2_noon_rules_gate4.csv`: the rule refit on train plus the selection set (`train --model rules --none-gate --with-selection`), 1,000 test Clients, allowed labels, passes `submit --check`. The rule learns nothing from labels, so it equals the milestone-2 submission byte for byte. Made by `bash scripts/day2_noon_milestone.sh`; slow check `tests/test_milestone_submission_real_data.py` reproduces it.
- No new selection-set evaluate run: the refit model is fitted on the selection set, and the three candidates' runs were already logged with predictions committed. The sealed holdout was not read.
- For the human: both ranker variants lead the rule by about 0.02 on the selection set, under the 0.03 margin. Uploading a ranker instead would be a judgement call outside this ticket's tie rule.
