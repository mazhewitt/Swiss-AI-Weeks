# 11: Streams that don't break on Filler Descriptions

**What to build:** Make stream detection robust to the Filler Description rate of valid and test, so that the stream features of train Clients look like those of valid and test Clients.

Ticket 09's review found a covariate shift in the stream table. Label-free, from the package's public functions:

- A classifier telling train from valid Clients reaches AUC 0.73 on the churn features. Against test it reaches 0.80.
- `max_missed_rate` averages 0.069 in train, 0.129 in valid and 0.167 in test.
- `n_ended` falls from 0.58 to 0.30, and `n_short` rises from 1.14 to 1.48.
- The top description of a stream is a Filler Description for 1.3% of streams in train and 16.7% in valid (`experiments/analysis/churn/drift_streams.csv`).

That shift is the likely reason ticket 09's +0.040 on train became a tie on selection, and it probably costs every stream-based model some of its score.

Likely mechanism, to confirm first: `streams._evidence` drops a description with no family word unless its MCC is a family's home MCC. So a stream payment that carries a Filler Description ("member plan", "subscription charge") on a non-home MCC is dropped. The stream then shows a gap: a "missed" payment, or a stream split in two. Valid and test have many more Filler Descriptions, so their streams break more often.

Fix direction: let a Filler Description payment on any MCC (never a Decoy Transaction, a shop payment or a fee) join an existing stream when its amount fits the stream's amount cluster and its date fits the stream's schedule. It must never start a stream. Other fixes are fine if the diagnosis points elsewhere.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] A diagnosis with figures: why stream features differ between train, valid and test. Label-free: transactions and stream tables only, never a valid or test label. Recorded in a comment on the ticket
- [ ] The detector change, with unit tests at the detector seam: a stream whose payments partly carry Filler Descriptions on other MCCs is detected whole; a Filler Description payment at another amount, or off the schedule, does not join; a Decoy Transaction never joins; the Decoy-injection test of ADR 0001 still passes
- [ ] Label-free shift check before and after: the train-vs-valid and train-vs-test classifier AUC on the ranker's candidate fields and on ticket 09's `none`-model features, plus the means of `max_missed_rate`, `n_short` and `n_ended`, as a table in the ticket. The change should narrow the shift
- [ ] Train out-of-fold (argmax, tuned, nested tuned) for the ranker with Pseudo-Labels, with and without `--none-model`, before and after the change. Pseudo-Label settings as in run `20260924T171108-3251a4`
- [ ] If the shift narrows and train does not get worse by more than 0.01: one selection run each for `ranker+pseudo` and `ranker+none+pseudo` (tuned), each with a paired bootstrap against the same model before the change (`20260924T171108-3251a4` and `20260924T225229-cc7243`). Predictions committed
- [ ] The Day-2 noon upload stays reproducible from its script at the commit it was made from. The script may now produce a different file on the new detector. That's expected, so say so in the ticket
- [ ] The full fast test suite passes
