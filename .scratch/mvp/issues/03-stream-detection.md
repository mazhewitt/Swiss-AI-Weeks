# 03: Stream detection with Merchant Family mapping

**What to build:** For any set of Clients, the team can produce a table of their Recurring Streams. Each stream has its Merchant Family, period, median amount, amount variation, gap regularity, payment count, first and last payment, projected next payment (rolled forward past the Cutoff), Active Stream flag and refund rate. The table can be inspected with a `streams` CLI summary and is cached per split. This is Seam 2. See the family table and behaviour notes in the MVP research folder (`data-facts.md`, plus the reference scripts for behaviour only) and ADR 0001.

**Blocked by:** 01

**Status:** resolved

- [x] Only outgoing card payments are candidates; transfers, shop-description payments and service-fee rows never form streams
- [x] Merchant Family comes from MCC plus description keywords, per the family table: "premium plan" under MCC 5734 is software; under 5812 it is streaming or music; "digital plus" under 4814 is mobile
- [x] Music versus streaming on MCC 5812 is decided by description hint first, then amount
- [x] Filler Descriptions stay inside the stream whose amount and MCC they fit; a stray MCC on one payment does not split a stream
- [x] Injecting Decoy Transactions at test-like rates (about 4.5 per Client) leaves the stream table unchanged
- [x] Appending transactions after the Cutoff leaves the stream table unchanged
- [x] One missed payment (a doubled gap) does not change a stream's period
- [x] An overdue projected next payment is rolled forward by whole periods into the Horizon
- [x] Active Stream = 3+ payments and last payment within about 1.6 periods of the Cutoff
- [x] Refunded payments stay in the stream and are counted in its refund rate
- [x] All thresholds (amount tolerance, active window, minimum payments) live in one parameter object
- [x] The stream table for each split is cached, and the `streams` command prints a per-family summary

## Comments

- Merged into main as 94e5469 ("Merge ticket 03: stream-detection"), a no-conflict `--no-ff` merge of `mvp/03-stream-detection`.
- Tests: `uv run pytest -m "not slow"` 89 passed. Every acceptance box has a matching Seam 1 (CLI) or Seam 2 (stream detection) test.
- Experiment: `rf streams` built and cached stream tables for all three splits under `artifacts/streams/` (train 3.8s, valid and test 2.0s each).
  - train: 2000 Clients, 6476 Recurring Streams, 3149 Active
  - test: 1000 Clients, 2862 Recurring Streams, 1336 Active
  - valid: 1000 Clients, 3029 Recurring Streams, 1340 Active. Per family (streams / active / clients): cloud 422/172/325, gym 415/220/341, insurance 424/198/322, mobile 405/194/338, music 422/183/345, software 430/190/342, streaming 511/183/389.
  - A second `streams --split valid` loaded the cached table. Nothing was logged, because stream detection is not a scored model run.
