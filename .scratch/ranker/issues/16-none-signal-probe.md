# 16: Where is the `none` signal another team found? (probe)

**Status:** ready-for-human (probe done; the findings decide what, if anything, to build)
**Blocked by:** none

**Why:** another team reached 0.68 on the test leaderboard against our 0.606 (v2). On 1,000 Clients that is
0.07, more than twice the 95% half-width: a signal, not noise. Our own ceiling analysis (ticket 13) says the
only bucket big enough is `none`: fixing every `none` error on selection would take v2 from 0.587 to about
0.70. So the question is what predicts `none` that our stream-table features (ADR 0001) do not.

Scripts and results: `experiments/analysis/none_probe/`. Train labels only, except the last step, which
cross-validates over train plus the selection set inside `data.training_run(with_selection=True)`, the
refit's own footing. The sealed holdout was not read.

## 1. Leak checks (`leak_checks.py`): clean

- Numeric Client id and label-file row order: AUC for `none` 0.52; lag-1 autocorrelation of `none` in file
  order −0.02 (independent: 0 ± 0.02); 1,678 label runs against an independent expectation of about 1,680.
- No split has a transaction at or after the Cutoff (train, valid, test, unlabeled).
- History shape carries mild honest signal: fewer card payments (AUC 0.66), less spend (0.64), fewer
  transactions (0.64) lean `none`. The stream count already carries most of it.

## 2. Kitchen-sink `none` model (`kitchen_sink_none.py`, 5-fold out-of-fold on train)

Raw-transaction features that ADR 0001 excludes: counts by type, window counts (7–90 days), recency of the
last card / family-named / Decoy / generic / refund payment, bags of the 80 most common description words
(all history, last 90 days, last payment), 25 MCC bags, plus the pre-ticket-11 stray-payment counts.

| Features | `none` AUC | train-vs-test AUC |
|---|---|---|
| streams only (the v2 `none` model's 104 features) | 0.888 | 0.728 |
| raw kitchen sink (283) | 0.893 | **1.000** |
| stray payments as before ticket 11 (3) | 0.601 | 0.908 |
| streams + raw (387) | **0.936** | |
| streams + raw + stray (390) | 0.937 | |

The gain is one word. The count of card payments described with **"subscription"** has gain 25,252 against
13,044 for the next feature; alone it has AUC 0.69, with a mean of 0.84 such payments for `none` Clients
against 0.09 for family Clients. And the raw features separate train from test perfectly.

## 3. The word, across splits (`subscription_word.py`)

Share of Clients with a card payment whose description contains the word:

| word | train `none` | train family | valid | test | unlabeled |
|---|---|---|---|---|---|
| subscription | 0.446 | 0.081 | 0.649 | 0.767 | 0.136 |
| charge | 0.526 | 0.195 | 0.977 | 0.992 | 0.136 |
| member | 0.645 | 0.483 | 0.850 | 0.856 | 0.433 |
| order / merchant / payment / purchase (Decoy words) | ~0.16 | ~0.14 | ~0.94 | ~0.98 | 0.000 |

- It is the three Filler Descriptions: **"member plan", "digital service", "subscription charge"** are each on
  45–50% of train `none` Clients and 8% of family Clients. In train the `none` rate climbs with the count of
  "subscription charge" payments: 0 → 20%, 1 → 56%, 2 → 91%, 3 → 97%, 4+ → 100%. Recency matters less
  (83% within 15 days, 48% beyond 120).
- In train `none` Clients, 57% of these payments are **strays** (in no detected stream; MCCs include 5411,
  groceries); in family Clients 98% sit inside a stream. In **test 91% sit inside a stream**, on 77% of
  Clients, 2.9 per Client (train: 19% of Clients, 1.65). The unlabeled split looks like train (13.6%).
- So train marks churned Clients with scattered Filler Descriptions, and valid/test spray the same
  descriptions over most Clients' real streams. This is ticket 11's finding seen from the description side,
  and the reason the `none` model's train gain vanished on selection (ticket 09 → 11).

## 4. Filler features in the Survival Race (`filler_survival.py`, soft race, train out-of-fold)

Per-stream: filler payments, share, recency, last-90-day count. Client-level: filler share and counts,
stray filler counts. Relative (base-rate free): the stream's filler share minus the Client's mean, its rank,
its share of the Client's filler payments, its last filler payment relative to the Client's most recent.

| Variant | nested tuned macro-F1 | s AUC | extra features' train-vs-test AUC |
|---|---|---|---|
| base (soft race, ticket 14) | 0.6387 | 0.872 | |
| R: relative only | 0.6275 | 0.869 | 0.753 |
| RA: + per-stream absolute | 0.6365 | 0.873 | 0.851 |
| RAC: + Client-level absolute | 0.6409 | 0.875 | 0.998 |

Inside the race, a stream's own filler payments barely predict its stopping. The Client-level counts are
where the train signal lives, and they are the ones that do not transfer. (Limitation: the Client-level
"generic" count here also includes shop descriptions; the stream-level ones do not.)

## 5. Learned where the base rate matches test (`filler_on_selection.py`)

| Variant | cv over train + selection, scored on the 700 selection Clients | cv over the selection set alone |
|---|---|---|
| base | 0.5796 | 0.5615 |
| R | 0.5854 | 0.5718 |
| RA | 0.5824 | 0.5772 |
| RAC | 0.5909 | 0.5849 |

Even trained on valid-like data the descriptions add +0.01 to +0.02 nested tuned macro-F1: a tie, and
nothing on the road to 0.68.

## Conclusion

- The files leak nothing. The strongest raw signal for `none` in train is the Filler Descriptions, and it is
  a property of how train was generated: valid and test spray the same descriptions over most Clients. It
  does not transfer as an absolute count, and as a within-Client relative feature it is worth at most a tie.
- **0.68 is not reachable from descriptions or transaction counts with our labels.** Whatever the other
  team does is elsewhere: training that leans on the valid labels much harder than our 2,000 + 700 refit,
  a different reading of the Horizon and label, or the generator's structure, which we ruled out on
  purpose (ADR 0002).
- Nothing built. Upload the soft race if a slot is free (ticket 14); the remaining honest lever is still
  `none`, and this probe closes the description route to it.
