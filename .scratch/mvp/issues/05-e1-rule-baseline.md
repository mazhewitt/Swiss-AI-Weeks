# 05: E1 rule baseline

**What to build:** The team can train and evaluate the rule baseline. From each Client's Active Streams it predicts the Merchant Family whose projected next payment is soonest within the Horizon, else `none`, for every Client including those with no streams. The experiment log also records its diagnostics. This unlocks the milestone-1 submission.

**Blocked by:** 02, 03

**Status:** wontfix (superseded by .scratch/ranker/issues/01; its branch is resumed there)

- [ ] `train --model rules` predicts for every Client in a split, including Clients with no streams
- [ ] The ordering rule is selectable (default: soonest projected next payment); only Active Streams with next payment within the Horizon are considered
- [ ] The experiment log row includes coverage (how often the true family is among surviving streams) and selection accuracy given coverage
- [ ] A fixture with two Active Streams picks the one due soonest; a Client whose streams have all stopped gets `none`
- [ ] A slow, marked real-data test: macro-F1 on the valid selection set lands within ±0.04 of 0.479
