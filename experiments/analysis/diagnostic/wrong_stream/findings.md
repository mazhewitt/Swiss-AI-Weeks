# Wrong detected stream: findings (train only, 5-fold OOF, argmax, untuned)

Bucket = label family has a Candidate Stream, model picks another family. Survival Race OOF: 331/2000 (16.6%);
v2: 348. Oracle-fixing it: 0.632 -> 0.837 macro-F1. Scripts run in order: prep.py, a1.py, proj.py, a3/a4.py,
race.py, a6/a7.py, cvrace.py (reproduces oof_surv exactly: 0.6322), backtest_proj.py, backtest2.py, cvrace_order.py,
a8-a13.py, cvrace_refund.py. Caches in cache/.

## 1. The bucket is mostly irreducible under the race, and the ordering is already close to ideal
- Where the label sits in projected order, given it is not none (a4.py), vs an ideal order with s=0.5 iid:
  k=2 0.63/0.37 (ideal .667/.333); k=3 .59/.27/.14 (ideal .571/.286/.143); k=4 .51/.21/.16/.11 (.533/.267/.133/.067).
  A random order gives .51/.49 for k=2.
- Race P is calibrated: accuracy matches max P in every bin. In the bucket, mean P(picked) is 0.58 and P(true) is 0.20.
- Best-case gain from a perfect order is about +0.01 to +0.015 macro-F1 (the k=2 rank-1 gap 0.63 -> 0.667).

## 2. Projection: fixed ~30.4-day step from the last payment beats fitted period (backtest_proj.py, backtest2.py)
Backtest at Shifted Cutoffs 2025-09/10/11 over 7,205 monthly streams, scored on pairwise within-Client order accuracy:
last+median folded gap (current) 0.792; last+30.44 0.823; same day-of-month next calendar month 0.825; last+31 0.832;
last gap 0.739; anchored schedule 0.766; regression 0.802. The schedule behaves like a random walk:
- next payment = previous + ~30.4 d + jitter, error sd about 3.5 d, successive gap-residual correlation -0.18;
- only 10% of payments repeat the previous day-of-month;
- weekday is uniform, and hours are uniform from 08 to 21;
- per-stream fitted periods of 26-33 days are noise.
CV race: original order 0.6322; last+30.44 0.6386; calendar month 0.6409; regression 0.6389. That is +0.006 to +0.009, a tie.

## 3. True vs picked stream in the bucket (a8.py, a9.py)
- The true stream is projected later than the picked one (true rank >=2 in 88% of the bucket). It has fewer payments
  (median 6 vs 9) and is younger (183 vs 263 days).
- 37.5% of true streams are not Active (72 single-payment streams, 31 two-payment streams).
- A single-payment stream whose payment was 15-31 days ago is the label 58% of the time (n=130). Its survival is about
  0.47, so it fits the race when it is placed at its monthly slot.
- Per-stream payment features barely help: descriptions/stream, amount levels, fee share, currency different from the
  Client's main currency, last-payment deviation, MCCs, day-of-month and hour.
  - They raise survival AUC on active rows of non-none Clients from 0.739 to 0.764.
  - They move race macro-F1 by only +0.004.
- Currency mismatch: survival 0.55 vs 0.44 (11% of active rows).
- Top-quintile amount noise: survival 0.24 vs 0.57. The model already has this through amount_cv.

## 4. History outside streams (a12.py, a13.py)
- No cancel, price, reminder, trial, failed, renewal or promo text exists anywhere: 0 matches. The only description
  vocabulary is shops, family names and fillers/decoys.
- Refunds of the family that were not credited to the stream: survival 0.32 vs 0.50 (24% of active rows). CV race
  change is -0.002, because refund_rate already captures it.
- Non-stream payments that name a family do not separate true from picked in the bucket (9.4% vs 10.9%).
- Hidden-series count, a Client-level feature: +0.031 macro-F1 (0.6322 -> 0.6629, accuracy +3.75 points). It works
  almost entirely through none: bucket accuracy is only 6.6%.
  - Clients with more than 5 such payments are 89% none (232 Clients, 11.6%).
  - Already known (churn/README). It was parked because of the decoy drift in valid and test.

## Bottom line
The wrong-stream bucket is not a projection bug. Streams survive roughly as independent coin flips (~0.5), and our
calibrated race picks the Bayes-optimal first stream. Improving the order gets at most about +0.01. The remaining
structural lever found is the hidden-series/none signal (+0.03 train), and it needs a drift check.
