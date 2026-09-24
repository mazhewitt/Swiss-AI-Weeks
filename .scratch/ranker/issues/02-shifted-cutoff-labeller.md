# 02: Shifted-Cutoff labeller

**What to build:** A public function beside stream detection returns each Client's Pseudo-Label for any Cutoff and Horizon: the Merchant Family of the first Recurring Stream payment after that Cutoff and within the Horizon, or `none`. It uses the Client's full history to decide which Horizon payments belong to a Recurring Stream, so a stream that starts inside the Horizon still counts once it repeats. The stream table's columns do not change.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Of two streams due at different dates in the Horizon (one biweekly, one monthly), the earlier payment's family wins
- [x] A stream that stopped before the Shifted Cutoff yields `none`; a Client with no Horizon payments yields `none`
- [x] A new stream that starts inside the Horizon and repeats counts
- [x] A Decoy Transaction, one-off payment or refund that is the first Horizon payment is ignored
- [x] A payment with a Filler Description inside a real stream counts as that stream's family
- [x] The minimum number of stream payments is a parameter
- [x] A Shifted Cutoff whose Horizon ends after the last observed day (2025-12-31) is refused with a clear error
- [x] Injecting Decoy Transactions leaves every Pseudo-Label unchanged (mirrors the ADR 0001 detector test)
- [x] Tests sit at Seam 2 (the stream module's public functions), built from hand-made histories

## Comments

**Implementation (ranker/02-shifted-cutoff-labeller).** `recurring_family.streams.pseudo_labels(transactions, cutoff=SHIFTED_CUTOFF, horizon=HORIZON, min_payments=2, params=StreamParams())` returns a `string` Series indexed by sorted `client_id` (named `target_next_recurring_merchant`), one row for every Client in the table. `config` gains `HISTORY_END` (2026-01-01, so the last observed day is 2025-12-31) and `SHIFTED_CUTOFF` (2025-10-03).

- The Horizon is `[cutoff, cutoff + horizon)`: a payment at the Cutoff instant is in the Horizon, matching `detect_streams`, whose history is `timestamp < cutoff`.
- Streams are detected over the whole known history (everything before `HISTORY_END`), including payments after the Horizon for an earlier Shifted Cutoff. `min_payments` counts all of a stream's payments, before and after the Cutoff. Values below 2 are refused.
- Payments at the same instant from two families go to the first family by name.
- `detect_streams` output is unchanged: identical frame on the real train split before and after the refactor. The source-file hash means existing stream caches are rebuilt once.

**For ticket 04 (fidelity).** Train at the default Shifted Cutoff, `none` share: 0.082 (min_payments 2), 0.116 (3), 0.154 (4). The real train share is 0.298. The `none`-share tolerance (0.05) looks hard to meet by `min_payments` alone. Labelling 2,000 train Clients takes about 33 s, the same as one `detect_streams` pass.
