# 01: Finish the rules baseline with the `none`-gate

**What to build:** The team can train and evaluate the milestone-2 rule from the CLI. It predicts the family of the Active Stream whose projected next payment is soonest within the Horizon, but predicts `none` when the Client's longest qualifying stream has at most N payments (the `none`-gate, default N = 4, and it can be switched off). The rule with and without the gate is logged on the selection set, so every later experiment has an honest previous best. Start from the MVP's ticket 05 branch (`mvp/05-e1-rule-baseline`), bring it up to date with main and finish it. This supersedes the MVP's ticket 05.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `train --model rules` and `evaluate --model rules` work on current main for every Client, including Clients with no streams
- [x] The `none`-gate threshold is a model parameter (default 4, off when unset) and is saved with the model
- [x] A fixture Client whose longest qualifying stream has 3 payments gets `none` with the gate on and a family with it off
- [x] The slow, marked real-data E1 reference is amended to macro-F1 0.5365 ± 0.02 on the selection set (gate off)
- [x] The gated rule reproduces the committed milestone-2 predictions on test (submissions/milestone2_rules_none_gate_v2.csv)
- [x] Both variants are logged on the selection set, with per-run predictions committed

## Outcome

- `uv run rf train --model rules [--none-gate [N]]` (gate off unless given; N defaults to 4, saved in `artifacts/rules.json`; the log's model column reads `rules+gate4`).
- Selection set (700 Clients): E1 gate off 0.5365 (run 20260924T151636-c6fcf8, improvement over E2 +0.0643); gate 4 0.5490 (run 20260924T151704-615cf5, tie with E1 at +0.0126). The gated rule is now the previous best.
- `train --model rules --none-gate` then `submit --model rules` reproduces `submissions/milestone2_rules_none_gate_v2.csv` row for row (slow test `test_gated_rules_reproduce_the_milestone2_submission`).
