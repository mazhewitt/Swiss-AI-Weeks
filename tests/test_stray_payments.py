"""Stray Payments (ticket 11): the one-sided rules of joining a stream on its schedule, the slot and window
bounds, backward chains, music and streaming targets, and refunds of joined payments. Each test pins a rule
that a plausible bug would break (the ticket 11 review's mutation run)."""

import pandas as pd
from recurring_family.streams import detect_streams, detect_stream_payments
from test_streams import tx, frame, series, one, _joins, DAY


def test_early_within_tolerance_joins():
    assert _joins("member plan", "5411", 47, "2025-08-01") == 1


def test_early_off_schedule_never_joins():
    assert _joins("member plan", "5411", 47, "2025-07-27") == 0


def test_two_periods_before_first_never_joins():
    assert _joins("member plan", "5411", 47, "2025-02-04") == 0


def test_two_strays_in_one_slot_only_one_joins():
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 8)
    del rows[4]
    extra = [tx("C1", "2025-08-03", 47, "member plan", "5411"), tx("C1", "2025-08-04", 47, "subscription charge", "5734")]
    s = one(detect_streams(frame(rows + extra)))
    assert s.n_payments == 8


def test_backward_chain_of_strays_extends_stream():
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 8)
    extra = [tx("C1", "2025-03-06", 47, "member plan", "5411"), tx("C1", "2025-02-04", 47, "member plan", "5411")]
    s = one(detect_streams(frame(rows + extra)))
    assert s.n_payments == 10


def test_biweekly_off_schedule_by_5_days_never_joins():
    rows = series("C1", "phone contract", "4814", 47, "2025-06-02", 12, every_days=14)
    del rows[5]
    # missing slot at 2025-06-02 + 70d = 2025-08-11; 5 days late
    s = one(detect_streams(frame(rows + [tx("C1", "2025-08-16", 47, "member plan", "5411")])))
    assert s.n_payments == 11


def test_biweekly_stray_2_5_days_from_a_payment_never_joins():
    rows = series("C1", "phone contract", "4814", 47, "2025-06-02", 12, every_days=14)
    s = one(detect_streams(frame(rows + [tx("C1", pd.Timestamp("2025-06-30") + 2.5 * DAY, 47, "member plan", "5411")])))
    assert s.n_payments == 12


def test_music_stream_takes_a_stray():
    rows = series("C1", "audio pass", "5812", 12.9, "2025-04-05", 8)
    del rows[4]
    s = one(detect_streams(frame(rows + [tx("C1", "2025-08-03", 12.9, "subscription charge", "5734")])))
    assert (s.family, s.n_payments) == ("music", 8)


def test_price_drift_range_both_ends():
    amts = [40, 41, 42, 43, 44, 45, 46, 47]
    rows = [tx("C1", pd.Timestamp("2025-04-05") + 30 * k * DAY, a, "phone contract", "4814") for k, a in enumerate(amts)]
    early = [r for k, r in enumerate(rows) if k != 1]
    s = one(detect_streams(frame(early + [tx("C1", rows[1]["timestamp"], 47, "member plan", "5411")])))
    assert s.n_payments == 8
    late = [r for k, r in enumerate(rows) if k != 6]
    s = one(detect_streams(frame(late + [tx("C1", rows[6]["timestamp"], 40, "member plan", "5411")])))
    assert s.n_payments == 8


def test_stream_of_two_takes_a_stray():
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 2)
    s = one(detect_streams(frame(rows + [tx("C1", "2025-06-04", 47, "member plan", "5411")])))
    assert s.n_payments == 3


def test_nearest_slot_wins_over_nearest_amount():
    gym = series("C1", "gym membership", "7997", 47, "2025-04-05", 8)
    mob = series("C1", "phone contract", "4814", 48, "2025-04-07", 8)
    del gym[4], mob[4]
    t = pd.DataFrame(mob)["timestamp"].iloc[3] + 30 * DAY  # mobile's empty slot, 2 days after gym's
    st = detect_streams(frame(gym + mob + [tx("C1", t, 47, "member plan", "5411")]))
    assert dict(zip(st.family, st.n_payments)) == {"gym": 7, "mobile": 8}


def test_family_refund_reverses_a_joined_payment():
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 8)
    del rows[4]
    extra = [tx("C1", "2025-08-03", 47, "member plan", "5411"),
             tx("C1", "2025-08-05", 47, "phone contract", "4814", type_="refund", direction="in")]
    s = one(detect_streams(frame(rows + extra)))
    assert (s.n_payments, s.n_refunds) == (8, 1)


def test_stray_refund_is_never_credited():
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 8)
    extra = [tx("C1", "2025-07-06", 47, "member plan", "5411", type_="refund", direction="in")]
    s = one(detect_streams(frame(rows + extra)))
    assert s.n_refunds == 0


def test_row_order_never_matters_with_strays():  # sort mutants
    rows = series("C1", "phone contract", "4814", 47, "2025-04-05", 8)
    del rows[4]
    extra = [tx("C1", "2025-08-02", 47, "member plan", "5411"), tx("C1", "2025-08-04", 47, "subscription charge", "5734")]
    a = detect_stream_payments(frame(rows + extra))[1]
    b = detect_stream_payments(frame(list(reversed(extra)) + rows))[1]
    pd.testing.assert_frame_equal(a, b)
    assert pd.Timestamp("2025-08-02", tz="UTC") in set(a.timestamp)
