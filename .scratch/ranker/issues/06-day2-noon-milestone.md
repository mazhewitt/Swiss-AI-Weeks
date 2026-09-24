# 06: Day-2 12:00 milestone submission

**What to build:** Pick the best of the rules baseline with the gate, the ranker, and the ranker with Pseudo-Labels on the selection set, using paired bootstrap (deltas under 0.03 are ties; on a tie prefer the simpler candidate). Refit the winner on train plus the selection set, write the test submission, validate it and commit it with the exact command that produced it. The human uploads it.

**Blocked by:** 01, 03, 05

**Status:** ready-for-agent

- [ ] A short comparison of the three candidates' selection scores and bootstrap intervals is written next to the submission
- [ ] The submission has 1,000 test Clients, allowed labels only, and passes `submit --check`
- [ ] The file and the command that produced it are committed
- [ ] The sealed holdout is not read
