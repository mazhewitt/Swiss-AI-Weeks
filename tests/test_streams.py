"""Seam 2: stream detection (Client transactions + Cutoff -> Recurring Stream table)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from recurring_family.config import CUTOFF, HORIZON
from recurring_family.data import TRANSACTION_DTYPES
from recurring_family.streams import StreamParams, detect_streams

DAY = pd.Timedelta(days=1)


def tx(client, when, amount, description, mcc, type_="card_payment", direction="out", currency="chf"):
    return {
        "client_id": client,
        "timestamp": pd.Timestamp(when).tz_localize("UTC") if pd.Timestamp(when).tzinfo is None else pd.Timestamp(when),
        "amount": float(amount),
        "currency": currency,
        "direction": direction,
        "type": type_,
        "mcc": mcc,
        "description": description,
        "fee": 0.0,
    }


def frame(rows):
    df = pd.DataFrame(rows)
    return df.astype(TRANSACTION_DTYPES)


def series(client, description, mcc, amount, start, n, every_days=30, **kw):
    first = pd.Timestamp(start)
    return [tx(client, first + i * every_days * DAY, amount, description, mcc, **kw) for i in range(n)]


def one(streams, client=None):
    s = streams if client is None else streams[streams.client_id == client]
    assert len(s) == 1, s
    return s.iloc[0]


# --- candidates ---------------------------------------------------------------


def test_transfers_shop_payments_and_service_fees_never_form_streams():
    rows = (
        series("C1", "gym membership", "7997", 66, "2025-07-05", 6, type_="transfer")
        + series("C1", "grocery store", "5411", 40, "2025-07-06", 6)
        + series("C1", "coffee shop", "5812", 15, "2025-07-07", 6)
        + series("C1", "service fee", "6012", 1.2, "2025-07-08", 6)
        + series("C1", "service fee", "6012", 1.2, "2025-07-08", 6, type_="fee")
        + series("C1", "cloud backup", "5732", 7, "2025-07-09", 6, direction="in", type_="p2p_transfer")
    )
    assert detect_streams(frame(rows)).empty


def test_shop_payments_never_join_a_stream_even_on_its_mcc_and_amount():
    rows = series("C1", "media streaming", "5812", 15, "2025-07-05", 6)
    rows += series("C1", "coffee shop", "5812", 15.1, "2025-07-20", 3)
    rows += series("C1", "online marketplace", "5812", 15, "2025-07-21", 2, type_="transfer")
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("streaming", 6)


def test_one_outgoing_card_series_forms_one_stream_with_its_statistics():
    rows = series("C1", "gym membership", "7997", 66, "2025-07-05", 6)
    s = one(detect_streams(frame(rows)))
    assert s.family == "gym"
    assert s.n_payments == 6
    assert s.period_days == pytest.approx(30)
    assert s.median_amount == pytest.approx(66)
    assert s.amount_cv == pytest.approx(0)
    assert s.first_payment == pd.Timestamp("2025-07-05", tz="UTC")
    assert s.last_payment == pd.Timestamp("2025-12-02", tz="UTC")
    assert s.active


# --- Merchant Family mapping --------------------------------------------------


@pytest.mark.parametrize(
    "description, mcc, amount, family",
    [
        ("cloud backup", "5732", 7, "cloud"),
        ("urban gym", "7997", 66, "gym"),
        ("policy premium", "6300", 109, "insurance"),
        ("phone contract", "4814", 47, "mobile"),
        ("saas billing", "5734", 39, "software"),
        ("premium plan", "5734", 39, "software"),
        ("digital plus", "4814", 47, "mobile"),
        ("media streaming", "5812", 18, "streaming"),
        ("audio streaming", "5812", 13.5, "music"),
    ],
)
def test_family_comes_from_mcc_plus_description(description, mcc, amount, family):
    s = one(detect_streams(frame(series("C1", description, mcc, amount, "2025-08-01", 5))))
    assert s.family == family


def test_premium_plan_is_software_on_5734_and_streaming_or_music_on_5812():
    rows = series("C1", "premium plan", "5734", 39, "2025-08-01", 5) + series("C2", "premium plan", "5812", 18, "2025-08-01", 5)
    streams = detect_streams(frame(rows))
    assert one(streams, "C1").family == "software"
    assert one(streams, "C2").family == "streaming"


def test_ambiguous_descriptions_never_take_a_family_they_cannot_mean():
    # "premium plan" is software, streaming or music; on a gym MCC it is not gym.
    rows = series("C1", "premium plan", "7997", 66, "2025-08-01", 5)
    assert detect_streams(frame(rows)).empty


@pytest.mark.parametrize("description", ["premium plan", "digital plus"])
def test_music_versus_streaming_on_5812_is_decided_by_amount_without_a_hint(description):
    rows = series("C1", description, "5812", 12.5, "2025-08-01", 5) + series("C2", description, "5812", 19.9, "2025-08-01", 5)
    streams = detect_streams(frame(rows))
    assert one(streams, "C1").family == "music"
    assert one(streams, "C2").family == "streaming"


def test_description_hint_beats_amount_for_music_versus_streaming():
    # A cheap video stream and a pricey audio stream: the hint wins over the amount.
    rows = (
        series("C1", "video access", "5812", 11, "2025-08-01", 3)
        + series("C1", "premium plan", "5812", 11, "2025-11-01", 2)
        + series("C2", "member pass", "5812", 22, "2025-08-01", 5)
    )
    streams = detect_streams(frame(rows))
    assert one(streams, "C1").family == "streaming"
    assert one(streams, "C1").n_payments == 5
    assert one(streams, "C2").family == "music"


# --- Filler Descriptions, stray MCC, Decoy Transactions ------------------------


def test_filler_descriptions_stay_in_the_stream_their_amount_and_mcc_fit():
    rows = series("C1", "gym membership", "7997", 66, "2025-06-05", 7)
    rows[2]["description"] = "member plan"
    rows[4]["description"] = "subscription charge"
    rows[5]["description"] = "digital service"
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("gym", 7)


def test_filler_descriptions_alone_never_form_a_stream():
    rows = series("C1", "subscription charge", "7997", 66, "2025-06-05", 6)
    assert detect_streams(frame(rows)).empty


def test_a_stray_mcc_on_one_payment_does_not_split_the_stream():
    rows = series("C1", "premium plan", "5734", 39, "2025-06-05", 7)
    rows[3]["mcc"] = "5411"
    rows[5]["mcc"] = "5812"
    rows[1]["mcc"] = "7299"
    rows[1]["description"] = "saas billing"
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("software", 7)


def _client_histories():
    return (
        series("C1", "gym membership", "7997", 66, "2025-06-05", 7)
        + series("C1", "premium plan", "5812", 17.9, "2025-03-10", 10)
        + series("C1", "grocery store", "5411", 45, "2025-06-01", 12, every_days=15)
        + series("C2", "phone contract", "4814", 47, "2025-01-20", 12)
        + series("C2", "cloud backup", "5732", 6.8, "2025-09-02", 4)
        + series("C2", "insurance monthly", "6300", 109, "2025-02-11", 11)
        + series("C3", "saas billing", "5734", 39, "2025-10-01", 3)
        + series("C3", "audio streaming", "5812", 13.5, "2025-04-01", 5, every_days=14)
    )


DECOYS = ["digital order", "merchant charge", "service payment", "card purchase", "pay digital order online"]
DECOY_MCCS = ["7997", "5812", "5734", "4814", "5732", "6300", "5411", "4111", "5999", "7011"]


def test_injecting_decoys_at_test_like_rates_leaves_the_stream_table_unchanged():
    base = frame(_client_histories())
    clean = detect_streams(base)
    assert len(clean) == 7

    rng = np.random.default_rng(0)
    decoys = []
    for client in ["C1", "C2", "C3"]:
        for _ in range(rng.poisson(4.5) + 3):
            decoys.append(
                tx(
                    client,
                    pd.Timestamp("2024-12-01") + int(rng.integers(0, 390)) * DAY,
                    # amounts include exact hits on each stream's amount
                    float(rng.choice([66, 17.9, 47, 6.8, 109, 39, 13.5, 67, 25, 120])),
                    str(rng.choice(DECOYS)),
                    str(rng.choice(DECOY_MCCS)),
                )
            )
    # decoys that exactly shadow a stream's amount, MCC and cadence
    decoys += series("C1", "merchant charge", "7997", 66, "2025-06-06", 3)
    decoys += series("C2", "digital order", "4814", 47, "2025-12-01", 2)
    noisy = pd.concat([base, frame(decoys)], ignore_index=True).sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(detect_streams(noisy), clean)


# --- Cutoff -------------------------------------------------------------------


def test_transactions_after_the_cutoff_leave_the_stream_table_unchanged():
    base = frame(_client_histories())
    later = (
        series("C1", "gym membership", "7997", 66, "2026-01-01", 3)
        + series("C3", "saas billing", "5734", 39, "2026-01-05", 2)
        + series("C4", "phone contract", "4814", 47, "2026-01-02", 3)
        + [tx("C2", "2026-01-01T00:00:00", 7.0, "cloud backup", "5732")]
    )
    before = detect_streams(base)
    after = detect_streams(pd.concat([base, frame(later)], ignore_index=True))
    pd.testing.assert_frame_equal(after, before)


def test_cutoff_is_an_argument():
    rows = series("C1", "gym membership", "7997", 66, "2025-01-05", 6)  # last payment 2025-06-04
    assert one(detect_streams(frame(rows), cutoff=pd.Timestamp("2025-06-20", tz="UTC"))).active
    assert one(detect_streams(frame(rows), cutoff=pd.Timestamp("2025-04-01", tz="UTC"))).n_payments == 3


# --- period, projection, activity ---------------------------------------------


def test_one_missed_payment_does_not_change_the_period():
    for n, missing in [(3, 1), (8, 4)]:
        rows = series("C1", "gym membership", "7997", 66, "2025-05-05", n + 1)
        del rows[missing]
        s = one(detect_streams(frame(rows)))
        assert s.period_days == pytest.approx(30), (n, missing)

    rows = series("C1", "audio streaming", "5812", 13.5, "2025-09-01", 5, every_days=14)
    del rows[2]
    assert one(detect_streams(frame(rows))).period_days == pytest.approx(14)


def test_projected_next_payment_is_last_plus_period_when_not_overdue():
    rows = series("C1", "gym membership", "7997", 66, "2025-08-09", 5)  # last 2025-12-07
    s = one(detect_streams(frame(rows)))
    assert s.next_payment == pd.Timestamp("2026-01-06", tz="UTC")


def test_overdue_projected_next_payment_is_rolled_forward_by_whole_periods_into_the_horizon():
    # last payment 2025-10-23; +30 = 2025-11-22 (overdue), +60 = 2025-12-22 (overdue), +90 = 2026-01-21
    rows = series("C1", "gym membership", "7997", 66, "2025-06-25", 5)
    s = one(detect_streams(frame(rows)))
    assert s.next_payment == pd.Timestamp("2026-01-21", tz="UTC")
    assert CUTOFF <= s.next_payment < CUTOFF + HORIZON


def test_active_stream_needs_three_payments_and_a_recent_last_payment():
    rows = (
        series("A", "gym membership", "7997", 66, "2025-08-10", 5)  # last 2025-12-08: 24 days
        + series("B", "gym membership", "7997", 66, "2025-06-18", 5)  # last 2025-10-16: 77 days = 2.6 periods
        + series("C", "gym membership", "7997", 66, "2025-11-06", 2)  # 2 payments
        + series("D", "gym membership", "7997", 66, "2025-07-19", 4)  # last 2025-10-17: 76 days = 2.5 periods
        + series("E", "gym membership", "7997", 66, "2025-08-15", 4)  # last 2025-11-13: 49 days = 1.63 periods
        + series("F", "gym membership", "7997", 66, "2025-08-20", 4)  # last 2025-11-18: 44 days = 1.47 periods
    )
    streams = detect_streams(frame(rows)).set_index("client_id")
    assert streams.active.to_dict() == {"A": True, "B": False, "C": False, "D": False, "E": False, "F": True}


def test_refunded_payments_stay_in_the_stream_and_count_in_its_refund_rate():
    rows = series("C1", "saas billing", "5734", 37.6, "2025-06-18", 6)
    for i in (1, 3, 4):
        rows.append(tx("C1", rows[i]["timestamp"] + 3 * DAY, 37.9, "saas billing core", "5734", type_="refund", direction="in"))
    rows.append(tx("C1", "2025-07-01", 120, "hotel booking", "7011", type_="refund", direction="in"))
    s = one(detect_streams(frame(rows)))
    assert s.n_payments == 6
    assert s.refund_rate == pytest.approx(0.5)
    assert s.active

    no_refunds = one(detect_streams(frame(rows[:6])))
    assert no_refunds.refund_rate == 0


# --- parameters ---------------------------------------------------------------


def test_thresholds_live_in_one_parameter_object():
    fields = {f.name for f in dataclasses.fields(StreamParams)}
    assert {"amount_tolerance", "active_periods", "min_payments"} <= fields

    rows = series("C", "gym membership", "7997", 66, "2025-11-06", 2)
    assert not one(detect_streams(frame(rows))).active
    assert one(detect_streams(frame(rows), params=StreamParams(min_payments=2))).active

    rows = series("B", "gym membership", "7997", 66, "2025-06-18", 5)
    assert one(detect_streams(frame(rows), params=StreamParams(active_periods=3.0))).active

    rows = series("A", "gym membership", "7997", 66, "2025-06-05", 3) + series("A", "gym membership", "7997", 72, "2025-09-05", 3)
    assert len(detect_streams(frame(rows))) == 2
    assert len(detect_streams(frame(rows), params=StreamParams(amount_tolerance=0.12))) == 1


def test_many_clients_and_empty_input():
    streams = detect_streams(frame(_client_histories()))
    assert sorted(streams.groupby("client_id").size().items()) == [("C1", 2), ("C2", 3), ("C3", 2)]
    empty = detect_streams(frame(_client_histories()).iloc[:0])
    assert empty.empty and list(empty.columns) == list(streams.columns)
