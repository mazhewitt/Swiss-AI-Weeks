# 12: Stream payments that carry a Decoy description

**What to build:** Let a payment with a Decoy description ("digital order", "merchant charge", ...) join an existing Recurring Stream, but only under the Stray Payment rules from ticket 11:

- the payment fits the stream's amount range;
- it fits an empty slot on the stream's schedule, within `schedule_tolerance_days`;
- the stream has at least two payments and a period of at least `stray_min_period_days`;
- it never starts a stream.

Ticket 11's diagnosis (`experiments/analysis/filler_streams/`, the ticket's Comments) found 8.7 Decoy-described payments per 100 valid streams that fill a gap on the schedule, and 7.3 in test. The off-schedule control is 1.0 in valid. In train the on-schedule rate is 0.88, the same as its control. So in valid and test, some real stream payments carry a Decoy description, while in train almost none do. They break valid and test streams just as Filler Descriptions did.

ADR 0001's concern is that Decoy Transactions drift (0.16 per Client in train, 3.1 in valid, 4.5 in test) and must not create features. A random Decoy lands on an empty slot at the stream's amount at about the control rate. So the join must be judged by the gap between the on-schedule rate and the control rate in each split, and the ADR amended with those figures.

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] Figures per split (label-free): Decoy-described on-schedule joins against the off-schedule control (the expected false-join rate), before the change
- [ ] Behind `StreamParams.join_decoys` (default on only if the gate below passes, and in the cache key): Decoy-described payments join under the Stray Payment rules. Unit tests: a Decoy on an empty slot at the stream's amount joins; off schedule, at another amount or on a stream shorter than two payments it does not; it never starts a stream. The ADR 0001 Decoy-injection test is updated to inject off-schedule or off-amount Decoys and still passes, plus a test that random Decoys rarely join
- [ ] The label-free shift check as in ticket 11 (classifier AUC train vs valid and train vs test on the candidate fields and the `none` model's features; mean `max_missed_rate`, `n_short`, `n_ended`), before and after
- [ ] Train out-of-fold (argmax, tuned, nested tuned) for ranker+pseudo and ranker+none+pseudo, before and after
- [ ] Gate: the shift narrows and train gets worse by no more than 0.01 nested. Then one selection run of ranker+none+pseudo (tuned), with `rf compare` against `20260925T001755-e309a3` (0.5871), predictions committed, and `submissions/day2_final_ranker_none_v3.csv` from a v3 script (refit on train plus selection, `rf submit --check`). Otherwise leave `join_decoys` off by default and record the result
- [ ] ADR 0001 amended with the figures; `--param join_decoys=false` reproduces the ticket 11 outputs; slow tests (except those that run `rf evaluate`) and the fast suite pass
