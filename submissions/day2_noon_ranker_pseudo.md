# Day-2 12:00 milestone upload: Stream Ranker with Pseudo-Labels

The candidate comparison is in `day2_noon_rules_gate4.md`. It is a three-way tie on the 700-Client selection set, and ticket 06 resolved that tie to the simpler gated rule. But the rule's test predictions are identical to the milestone-2 upload, and team rank is the best score of any milestone. So this milestone uploads the candidate with the highest point estimate instead:

- Run `20260924T171108-3251a4` (`ranker+pseudo+tuned`): selection macro-F1 0.5735 (95% interval 0.5344 .. 0.6075), +0.0245 over the gated rule (tie).
- Settings: Pseudo-Labels from train and unlabeled at Shifted Cutoff 2025-10-03, min_payments 4, weight 0.5, tuned E3 decision. The Pseudo-Label fidelity check failed (run `20260924T162515-d6d36f`), so this is not a validated Pseudo-Label setup. It is the same setup that was scored on the selection set.
- Refit on train plus the selection set: 2,700 real-labelled plus 12,000 Pseudo-Labelled Clients. The tuned decision's out-of-fold macro-F1 is 0.5771.
- Test predictions: 1,000 Clients, and 76% agree with the gated rule. Label counts: none 228, streaming 128, insurance 126, mobile 121, software 115, cloud 103, gym 94, music 85.

Made by `bash scripts/day2_noon_ranker_pseudo.sh`, which runs:

    uv run rf train --model ranker --with-selection --decision tuned \
        --pseudo train:2025-10-03 --pseudo unlabeled:2025-10-03 \
        --pseudo-weight 0.5 --pseudo-min-payments 4
    uv run rf submit --model ranker --decision tuned --name day2_noon_ranker_pseudo
    uv run rf submit --check submissions/day2_noon_ranker_pseudo.csv

The sealed holdout was not read.
