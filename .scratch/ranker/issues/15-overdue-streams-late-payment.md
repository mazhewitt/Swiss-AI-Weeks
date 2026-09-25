# 15: Overdue streams may pay late, not a period later

**Status:** needs-triage
**Blocked by:** 14

**Why:** the detector rolls a projection that fell before the Cutoff forward by whole periods, so a monthly
stream 2 days overdue races at day 28. Its own jitter (ticket 14: a 3.6-day spread around the projection)
says it may instead pay on day 1 or 2, still late. On train, 319 of 3,214 Active Streams (9.9%) are overdue
and rolled, touching 305 of the 1,700 Clients with an Active Stream (ticket 14, diagnostic 5). Under the
soft race such a stream's date T_i is a mixture: with the jitter's tail mass beyond the days already overdue
it pays late (T_i ~ the unrolled projection's jitter, truncated to t > 0); otherwise it pays a period later
(the rolled slot's jitter). The survival model's s already sees `days_since_last` and `days_to_next`, so
part of this is learned; the order is what the roll gets wrong.

**What to decide first:** whether the late-payment mass should also lower s (an overdue stream that never
pays late is evidence of a stop), or whether s stays the model's and only the date moves. Recommended: only
the date moves, so the change stays inside the race and P(`none`) stays order-free.

**Pre-registered check:** train out-of-fold nested tuned macro-F1 against the soft race (ticket 14, A), and
the net flips among Clients with an overdue-and-rolled Active Stream.
