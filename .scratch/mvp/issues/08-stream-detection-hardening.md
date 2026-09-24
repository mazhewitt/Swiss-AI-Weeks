# 08: Stream detection hardening

**What to build:** The stream table is correct in the cases the ticket-03 reviewers found weak, and it is safe to rebuild during threshold sweeps. Music and streaming never merge into one stream. The cache can never serve a table built by different code or settings. Every Decoy Transaction pattern is covered. Refunds are credited only to the payments they reverse. The amount-variation and gap-regularity columns that E2 will use as features are pinned by tests. All behaviour is tested through Seam 2, except the cache, which is tested through the `streams` CLI (Seam 1).

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] A Client with a music stream and a streaming stream on MCC 5812 whose amounts sit within the amount tolerance of each other (e.g. 13.5 and 14.2) keeps two separate streams with the right families; description hints separate them before amount clustering can chain them together
- [ ] The stream cache key includes the detector code version and the family table and all stream parameters; changing any of them rebuilds the table instead of serving a stale one; tables for different parameter sets can coexist per split, so a sweep does not evict on every change
- [ ] The decoy-injection test covers every Decoy Transaction pattern in the data facts ("digital order", "merchant charge", "service payment", "card purchase"); removing any one from the decoy handling makes a test fail
- [ ] Refunds are credited only to a payment of the same stream that precedes them within a short window (days, not months) with a matching amount; a stream's refund rate never exceeds 1; refunds dated after the Cutoff never count
- [ ] The post-Cutoff test appends refunds as well as card payments and the stream table is unchanged
- [ ] Amount variation and gap regularity have value-level tests (a stream with known jittered gaps and amounts produces the expected values within tolerance); forcing either column to 0 makes a test fail
- [ ] Features built later should use Active Streams or streams with 3+ payments, because raw stream counts shift between train and valid/test due to description noise on non-home MCCs; record this in the ticket comments for 06, no code change here
