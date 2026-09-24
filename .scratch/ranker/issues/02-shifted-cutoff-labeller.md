# 02: Shifted-Cutoff labeller

**What to build:** A public function beside stream detection returns each Client's Pseudo-Label for any Cutoff and Horizon: the Merchant Family of the first Recurring Stream payment after that Cutoff and within the Horizon, or `none`. It uses the Client's full history to decide which Horizon payments belong to a Recurring Stream, so a stream that starts inside the Horizon still counts once it repeats. The stream table's columns do not change.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Of two streams due at different dates in the Horizon (one biweekly, one monthly), the earlier payment's family wins
- [ ] A stream that stopped before the Shifted Cutoff yields `none`; a Client with no Horizon payments yields `none`
- [ ] A new stream that starts inside the Horizon and repeats counts
- [ ] A Decoy Transaction, one-off payment or refund that is the first Horizon payment is ignored
- [ ] A payment with a Filler Description inside a real stream counts as that stream's family
- [ ] The minimum number of stream payments is a parameter
- [ ] A Shifted Cutoff whose Horizon ends after the last observed day (2025-12-31) is refused with a clear error
- [ ] Injecting Decoy Transactions leaves every Pseudo-Label unchanged (mirrors the ADR 0001 detector test)
- [ ] Tests sit at Seam 2 (the stream module's public functions), built from hand-made histories
