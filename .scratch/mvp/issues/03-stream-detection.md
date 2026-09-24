# 03: Stream detection with Merchant Family mapping

**What to build:** For any set of Clients, the team can produce a table of their Recurring Streams. Each stream has its Merchant Family, period, median amount, amount variation, gap regularity, payment count, first and last payment, projected next payment (rolled forward past the Cutoff), Active Stream flag and refund rate. The table can be inspected with a `streams` CLI summary and is cached per split. This is Seam 2. See the family table and behaviour notes in the MVP research folder (`data-facts.md`, plus the reference scripts for behaviour only) and ADR 0001.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Only outgoing card payments are candidates; transfers, shop-description payments and service-fee rows never form streams
- [ ] Merchant Family comes from MCC plus description keywords, per the family table: "premium plan" under MCC 5734 is software; under 5812 it is streaming or music; "digital plus" under 4814 is mobile
- [ ] Music versus streaming on MCC 5812 is decided by description hint first, then amount
- [ ] Filler Descriptions stay inside the stream whose amount and MCC they fit; a stray MCC on one payment does not split a stream
- [ ] Injecting Decoy Transactions at test-like rates (about 4.5 per Client) leaves the stream table unchanged
- [ ] Appending transactions after the Cutoff leaves the stream table unchanged
- [ ] One missed payment (a doubled gap) does not change a stream's period
- [ ] An overdue projected next payment is rolled forward by whole periods into the Horizon
- [ ] Active Stream = 3+ payments and last payment within about 1.6 periods of the Cutoff
- [ ] Refunded payments stay in the stream and are counted in its refund rate
- [ ] All thresholds (amount tolerance, active window, minimum payments) live in one parameter object
- [ ] The stream table for each split is cached, and the `streams` command prints a per-family summary
