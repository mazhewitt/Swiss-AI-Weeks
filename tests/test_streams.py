"""Seam 2: stream detection (Client transactions + Cutoff -> Recurring Stream table)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from recurring_family.config import CUTOFF, HORIZON
from recurring_family.data import TRANSACTION_DTYPES
from recurring_family.streams import StreamParams, detect_streams, pseudo_labels

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


def test_music_and_streaming_streams_within_the_amount_tolerance_of_each_other_stay_separate():
    # 13.5 and 14.2 are 0.05 apart in log amount, inside the amount tolerance: amount alone would chain them.
    assert np.log(14.2 / 13.5) < StreamParams().amount_tolerance
    rows = (
        # C1: the canonical case, music cheaper than streaming
        series("C1", "audio streaming", "5812", 13.5, "2025-06-03", 6)
        + series("C1", "media streaming", "5812", 14.2, "2025-06-17", 6)
        # C2: other hinted descriptions, plus unhinted payments that each join the stream they fit best
        + series("C2", "member pass", "5812", 13.5, "2025-06-03", 6)
        + series("C2", "video access", "5812", 14.2, "2025-06-17", 6)
        + [tx("C2", "2025-12-02", 13.5, "premium plan", "5812"), tx("C2", "2025-12-16", 14.2, "digital plus", "5812")]
        # C3: music pricier than streaming; the hint still decides the family
        + series("C3", "audio stream", "5812", 14.2, "2025-06-03", 6)
        + series("C3", "video access plus", "5812", 13.5, "2025-06-17", 6)
    )
    streams = detect_streams(frame(rows))
    got = {(s.client_id, s.family): (s.median_amount, s.n_payments) for s in streams.itertuples()}
    assert got == {
        ("C1", "music"): (13.5, 6),
        ("C1", "streaming"): (14.2, 6),
        ("C2", "music"): (13.5, 7),
        ("C2", "streaming"): (14.2, 7),
        ("C3", "music"): (14.2, 6),
        ("C3", "streaming"): (13.5, 6),
    }


def test_unhinted_payments_carry_a_hinted_streams_price_drift():
    # C1: a drifting video stream whose hinted payments sit at two price levels; "premium plan"
    # payments bridge them into one stream
    amounts = [10.4, 10.7, 11.2, 11.8, 12.4, 13.0, 13.2]
    described = ["video access", "video access", "premium plan", "premium plan", "premium plan", "media streaming", "video access"]
    rows = [tx("C1", pd.Timestamp("2025-03-01") + 30 * i * DAY, a, d, "5812") for i, (a, d) in enumerate(zip(amounts, described))]
    # C2: premium plan payments drift up from a video stream, and a later filler fits only the drifted price
    rows += series("C2", "video access", "5812", 23.0, "2025-03-01", 3)
    rows += [tx("C2", "2025-06-01", 24.2, "premium plan", "5812"), tx("C2", "2025-07-01", 25.4, "premium plan", "5812")]
    rows += [tx("C2", "2025-08-01", 26.8, "subscription charge", "5812")]
    streams = detect_streams(frame(rows))
    assert (one(streams, "C1").family, one(streams, "C1").n_payments) == ("streaming", 7)
    assert (one(streams, "C2").family, one(streams, "C2").n_payments) == ("streaming", 6)


def test_a_hinted_stray_mcc_payment_prefers_its_hinted_family_then_falls_back_to_either():
    def music_and_streaming(client):
        return series(client, "audio streaming", "5812", 13.5, "2025-06-03", 6) + series(
            client, "media streaming", "5812", 14.2, "2025-06-17", 6
        )

    rows = music_and_streaming("C1") + [tx("C1", "2025-12-05", 14.1, "audio pass", "5411")]  # nearer streaming
    rows += music_and_streaming("C3") + [tx("C3", "2025-12-12", 13.6, "video access", "5411")]  # nearer music
    rows += series("C2", "audio streaming", "5812", 11.3, "2025-06-03", 6)
    rows += [tx("C2", "2025-12-05", 11.5, "video", "5411")]  # no streaming stream to join: joins the music one
    streams = detect_streams(frame(rows))
    got = {(s.client_id, s.family): s.n_payments for s in streams.itertuples()}
    assert got == {
        ("C1", "music"): 7,
        ("C1", "streaming"): 6,
        ("C3", "music"): 6,
        ("C3", "streaming"): 7,
        ("C2", "music"): 7,
    }


# --- Filler Descriptions, stray MCC, Decoy Transactions ------------------------


def test_filler_descriptions_stay_in_the_stream_their_amount_and_mcc_fit():
    rows = series("C1", "gym membership", "7997", 66, "2025-06-05", 7)
    rows[2]["description"] = "member plan"
    rows[4]["description"] = "subscription charge"
    rows[5]["description"] = "digital service"
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("gym", 7)


@pytest.mark.parametrize(
    "anchor, mcc, amount, family",
    [
        ("gym membership", "7997", 66, "gym"),
        ("insurance policy", "6300", 42, "insurance"),
        ("saas suite", "5734", 19, "software"),
        ("cloud storage", "5732", 9, "cloud"),
    ],
)
@pytest.mark.parametrize("filler", ["monthly plan", "mth plan"])
def test_monthly_plan_is_filler_in_a_non_mobile_stream_it_fits(anchor, mcc, amount, family, filler):
    rows = series("C1", anchor, mcc, amount, "2025-06-05", 7)
    rows[2]["description"] = filler
    rows[5]["description"] = filler
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == (family, 7)


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


# (client, home MCC, amount, first payment) of every stream in `_client_histories`
_HISTORY_STREAMS = [
    ("C1", "7997", 66, "2025-06-05"),
    ("C1", "5812", 17.9, "2025-03-10"),
    ("C2", "4814", 47, "2025-01-20"),
    ("C2", "5732", 6.8, "2025-09-02"),
    ("C2", "6300", 109, "2025-02-11"),
    ("C3", "5734", 39, "2025-10-01"),
    ("C3", "5812", 13.5, "2025-04-01"),
]


@pytest.mark.parametrize(
    "pattern, variants",
    [
        ("digital order", ["digital order", "pay dgtl order online", "digital order plus"]),
        ("merchant charge", ["merchant charge", "billing merchant charge", "merchant charge service"]),
        ("service payment", ["service payment", "member service payment", "service payment core"]),
        ("card purchase", ["card purchase", "pay card purchase", "card purchase digital"]),
    ],
)
def test_every_decoy_pattern_is_excluded_even_when_it_shadows_a_stream(pattern, variants):
    # The worst case for each Decoy Transaction pattern: its noised variants on every stream's home MCC,
    # at that stream's exact amount, between its payments. Filler Descriptions in the same spot would join.
    base = frame(_client_histories())
    clean = detect_streams(base)
    decoys = [
        tx(client, pd.Timestamp(first) + (10 + 30 * k) * DAY, amount, variants[k % len(variants)], mcc)
        for client, mcc, amount, first in _HISTORY_STREAMS
        for k in range(3)
    ]
    noisy = pd.concat([base, frame(decoys)], ignore_index=True)
    pd.testing.assert_frame_equal(detect_streams(noisy), clean)

    # control: the same rows with a Filler Description do change the table
    fillers = frame([{**d, "description": "subscription charge"} for d in decoys])
    assert not detect_streams(pd.concat([base, fillers], ignore_index=True)).equals(clean)


# --- Cutoff -------------------------------------------------------------------


def test_transactions_after_the_cutoff_leave_the_stream_table_unchanged():
    def refund(client, when, amount, description, mcc):
        return tx(client, when, amount, description, mcc, type_="refund", direction="in")

    # C5 pays its gym on 2025-12-29, so refunds of it dated just after the Cutoff would fit it
    base = frame(_client_histories() + series("C5", "gym membership", "7997", 66, "2025-08-01", 6))
    later = (
        series("C1", "gym membership", "7997", 66, "2026-01-01", 3)
        + series("C3", "saas billing", "5734", 39, "2026-01-05", 2)
        + series("C4", "phone contract", "4814", 47, "2026-01-02", 3)
        + [tx("C2", "2026-01-01T00:00:00", 7.0, "cloud backup", "5732")]
        + [
            refund("C5", "2026-01-01T00:00:00", 66, "gym membership", "7997"),
            refund("C5", "2026-01-03", 66.3, "urban gym", "7997"),
            refund("C1", "2026-01-02", 66, "gym membership", "7997"),
            refund("C1", "2026-01-06", 66, "gym membership", "7997"),
            refund("C3", "2026-01-07", 39, "saas billing", "5734"),
        ]
    )
    before = detect_streams(base)
    assert one(before, "C5").last_payment == pd.Timestamp("2025-12-29", tz="UTC")
    assert one(before, "C5").n_refunds == 0
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


def test_each_refund_is_credited_only_to_the_stream_whose_family_and_amount_it_fits():
    def refund(when, amount, description, mcc):
        return tx("C1", when, amount, description, mcc, type_="refund", direction="in")

    software = series("C1", "saas billing", "5734", 37.6, "2025-06-18", 6)
    small_software = series("C1", "productivity suite", "5734", 12.5, "2025-07-02", 5)
    gym = series("C1", "gym membership", "7997", 66, "2025-04-10", 8)
    rows = software + small_software + gym
    rows += [refund(software[i]["timestamp"] + 3 * DAY, 37.9, "saas billing core", "5734") for i in (1, 3, 4)]
    rows += [refund(small_software[i]["timestamp"] + 2 * DAY, 12.4, "prod suite", "5734") for i in (0, 2)]
    rows += [refund(gym[5]["timestamp"] + DAY, 66, "gym membership", "7997")]
    # Subscription-described refunds that fit no stream of their own family are counted nowhere:
    rows += [
        refund("2025-08-20", 80, "saas billing", "5734"),  # a software refund at no software amount
        refund("2025-08-22", 37.6, "gym membership", "7997"),  # a gym refund at the software amount
    ]
    streams = detect_streams(frame(rows)).set_index(["family", "median_amount"])
    assert streams.n_payments.to_dict() == {("gym", 66): 8, ("software", 12.5): 5, ("software", 37.6): 6}
    assert streams.n_refunds.to_dict() == {("gym", 66): 1, ("software", 12.5): 2, ("software", 37.6): 3}
    assert streams.refund_rate.to_dict() == pytest.approx(
        {("gym", 66): 1 / 8, ("software", 12.5): 2 / 5, ("software", 37.6): 3 / 6}
    )


def _refund(client, when, amount, description, mcc):
    return tx(client, when, amount, description, mcc, type_="refund", direction="in")


def test_a_refund_counts_only_shortly_after_a_payment_of_matching_amount():
    rows = series("C1", "saas billing", "5734", 37.6, "2025-06-18", 6)  # 06-18 .. 11-15, every 30 days
    paid = [r["timestamp"] for r in rows]
    rows += [
        _refund("C1", paid[1] + 3 * DAY, 37.9, "saas billing core", "5734"),  # counts: 3 days after, same amount
        _refund("C1", paid[3] + 6 * DAY, 37.6, "saas billing", "5734"),  # counts: 6 days after
        _refund("C1", paid[4] + 12 * DAY, 37.6, "saas billing", "5734"),  # mid-cycle: 12 days after, 18 before
        _refund("C1", paid[0] - 3 * DAY, 37.6, "saas billing", "5734"),  # before the first payment
        _refund("C1", paid[5] + 35 * DAY, 37.6, "saas billing", "5734"),  # a month after the stream's last payment
        _refund("C1", paid[2] + 2 * DAY, 45.0, "saas billing", "5734"),  # right time, amount of no payment
    ]
    s = one(detect_streams(frame(rows)))
    assert (s.n_payments, s.n_refunds) == (6, 2)
    assert s.refund_rate == pytest.approx(2 / 6)


def test_a_refund_matches_the_amount_of_the_payment_it_reverses_not_just_the_stream():
    # a drifting price chains into one stream from 60 to 68; a refund at 60 right after the 68 payment
    # sits inside the stream's amount range but reverses no payment
    amounts = [60, 61.8, 63.6, 65.5, 67.4, 68]
    rows = [tx("C1", pd.Timestamp("2025-06-10") + 30 * i * DAY, a, "gym membership", "7997") for i, a in enumerate(amounts)]
    rows += [
        _refund("C1", "2025-11-09", 60, "gym membership", "7997"),  # 2 days after the 68 payment
        _refund("C1", "2025-08-10", 63.4, "gym membership", "7997"),  # 1 day after the 63.6 payment
    ]
    s = one(detect_streams(frame(rows)))
    assert (s.n_payments, s.n_refunds) == (6, 1)


def test_a_refund_is_credited_to_the_stream_whose_payment_it_follows_not_the_nearest_amount():
    # two gym streams, 66 paid on the 5th and 72 on the 20th; each refund sits nearer the other stream's amount
    low = series("C1", "gym membership", "7997", 66, "2025-06-05", 6)
    high = series("C1", "gym membership", "7997", 72, "2025-06-20", 6)
    rows = low + high + [
        _refund("C1", low[2]["timestamp"] + 2 * DAY, 69.5, "gym membership", "7997"),  # nearer 72, after a 66 payment
        _refund("C1", high[3]["timestamp"] + 2 * DAY, 68, "fit club", "7997"),  # nearer 66, after a 72 payment
        _refund("C1", high[4]["timestamp"] + 2 * DAY, 68.5, "urban gym", "7997"),  # nearer 72, after a 72 payment
    ]
    streams = detect_streams(frame(rows)).set_index("median_amount")
    assert streams.n_payments.to_dict() == {66: 6, 72: 6}
    assert streams.n_refunds.to_dict() == {66: 1, 72: 2}


def test_a_refund_that_fits_two_recent_payments_reverses_the_one_closest_in_amount():
    # gym 66 paid on the 5th, gym 72 on the 8th; refunds on the 10th fit both payments in amount and time
    low = series("C1", "gym membership", "7997", 66, "2025-06-05", 6)
    high = series("C1", "gym membership", "7997", 72, "2025-06-08", 6)
    rows = low + high + [
        _refund("C1", high[1]["timestamp"] + 2 * DAY, 68.5, "gym membership", "7997"),  # closer to 66, paid earlier
        _refund("C1", high[3]["timestamp"] + 2 * DAY, 70.0, "gym membership", "7997"),  # closer to 72, paid later
    ]
    streams = detect_streams(frame(rows)).set_index("median_amount")
    assert streams.n_refunds.to_dict() == {66: 1, 72: 1}


def test_each_payment_is_refunded_at_most_once_so_the_refund_rate_never_exceeds_one():
    software = series("C1", "saas billing", "5734", 39, "2025-09-10", 3)
    gym = series("C2", "gym membership", "7997", 66, "2025-08-01", 5)
    rows = software + gym
    # C1: every payment refunded twice
    rows += [_refund("C1", p["timestamp"] + DAY, 39, "saas billing", "5734") for p in software for _ in range(2)]
    # C2: one payment refunded three times
    rows += [_refund("C2", gym[1]["timestamp"] + d * DAY, 66, "gym membership", "7997") for d in (1, 2, 3)]
    streams = detect_streams(frame(rows)).set_index("client_id")
    assert streams.n_refunds.to_dict() == {"C1": 3, "C2": 1}
    assert streams.refund_rate.to_dict() == pytest.approx({"C1": 1.0, "C2": 1 / 5})


# --- amount variation and gap regularity ----------------------------------------


def _stream_on(dates, amounts, description="insurance monthly", mcc="6300"):
    return [tx("C1", d, a, description, mcc) for d, a in zip(dates, amounts)]


def test_amount_variation_and_gap_regularity_of_a_jittered_monthly_stream():
    # gaps 28, 31, 33, 29, 30 days -> period 30, absolute deviations 2, 1, 3, 1, 0 -> median 1
    # amounts 60, 61, 59, 62, 58, 60 -> mean 60, sample std sqrt(10 / 5) -> variation sqrt(2) / 60
    dates = ["2025-06-01", "2025-06-29", "2025-07-30", "2025-09-01", "2025-09-30", "2025-10-30"]
    s = one(detect_streams(frame(_stream_on(dates, [60, 61, 59, 62, 58, 60]))))
    assert s.period_days == pytest.approx(30)
    assert s.gap_mad_days == pytest.approx(1.0)
    assert s.amount_cv == pytest.approx(np.sqrt(2) / 60)  # 0.02357


def test_amount_variation_and_gap_regularity_with_wider_jitter_and_a_missed_payment():
    # gaps 26, 33, 60 (a missed payment, folds to 30), 28, 34, 30 -> period 30,
    # absolute deviations 4, 3, 0, 2, 4, 0 -> median 2.5
    # amounts 20, 22, 18, 20, 21, 19, 20 -> mean 20, sample std sqrt(10 / 6) -> variation 0.06455
    dates = ["2025-04-01", "2025-04-27", "2025-05-30", "2025-07-29", "2025-08-26", "2025-09-29", "2025-10-29"]
    s = one(detect_streams(frame(_stream_on(dates, [20, 22, 18, 20, 21, 19, 20]))))
    assert s.n_payments == 7
    assert s.period_days == pytest.approx(30)
    assert s.gap_mad_days == pytest.approx(2.5)
    assert s.amount_cv == pytest.approx(np.sqrt(10 / 6) / 20)  # 0.06455


def test_amount_variation_and_gap_regularity_of_a_jittered_biweekly_stream():
    # gaps 13, 15, 14, 17, 14 -> period 14, absolute deviations 1, 1, 0, 3, 0 -> median 1
    # amounts 13.5, 13.5, 13.8, 13.5, 13.2, 13.5 -> mean 13.5, sample std sqrt(0.18 / 5)
    dates = ["2025-09-01", "2025-09-14", "2025-09-29", "2025-10-13", "2025-10-30", "2025-11-13"]
    s = one(detect_streams(frame(_stream_on(dates, [13.5, 13.5, 13.8, 13.5, 13.2, 13.5], "audio streaming", "5812"))))
    assert s.family == "music"
    assert s.period_days == pytest.approx(14)
    assert s.gap_mad_days == pytest.approx(1.0)
    assert s.amount_cv == pytest.approx(np.sqrt(0.18 / 5) / 13.5)  # 0.01405


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


# --- noise suffixes and off-home-MCC keywords -----------------------------------


def test_a_service_noise_suffix_does_not_move_a_payment_out_of_its_stream():
    rows = series("C1", "premium plan", "5812", 18, "2025-06-05", 4)
    rows += series("C1", "premium plan service", "5812", 18, "2025-10-03", 3)
    rows += series("C2", "cover plan", "6300", 109, "2025-06-05", 5)
    rows += series("C2", "cover plan service", "6300", 109, "2025-11-02", 2)
    rows += series("C3", "monthly plan service", "4814", 47, "2025-07-05", 6)
    streams = detect_streams(frame(rows))
    assert (one(streams, "C1").family, one(streams, "C1").n_payments) == ("streaming", 7)
    assert one(streams, "C1").active
    assert (one(streams, "C2").family, one(streams, "C2").n_payments) == ("insurance", 7)
    assert (one(streams, "C3").family, one(streams, "C3").n_payments) == ("mobile", 6)


def test_service_plan_on_the_cloud_mcc_is_a_cloud_stream():
    s = one(detect_streams(frame(series("C1", "service plan", "5732", 6.8, "2025-07-05", 6))))
    assert (s.family, s.n_payments) == ("cloud", 6)


def test_family_keywords_on_a_foreign_mcc_never_start_a_stream():
    rows = [
        tx("C1", "2025-08-03", 27.8, "cloud access", "5411"),
        tx("C1", "2025-09-14", 31.0, "cloud access", "4814"),
        tx("C1", "2025-10-01", 55.0, "gym membership", "5812"),
        tx("C1", "2025-11-20", 12.0, "audio pass", "5734"),
    ]
    assert detect_streams(frame(rows)).empty


def test_a_family_keyword_payment_on_a_foreign_mcc_still_joins_its_family_stream():
    rows = series("C1", "cloud backup", "5732", 6.8, "2025-07-05", 6)
    rows += [tx("C1", "2025-12-20", 6.8, "cloud backup", "5411"), tx("C1", "2025-08-20", 29.0, "cloud backup", "5812")]
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("cloud", 7)


# --- Filler Descriptions with several streams per Client -------------------------

TOLERANCE = StreamParams().amount_tolerance


def _six_streams(client):
    """One Client with a stream of every detection group, on its home MCC, far apart in amount."""
    return (
        series(client, "gym membership", "7997", 66, "2025-06-05", 7)
        + series(client, "media streaming", "5812", 18, "2025-06-07", 7)
        + series(client, "saas billing", "5734", 39, "2025-06-09", 7)
        + series(client, "cloud backup", "5732", 6.8, "2025-06-11", 7)
        + series(client, "phone contract", "4814", 47, "2025-06-13", 7)
        + series(client, "insurance monthly", "6300", 109, "2025-06-15", 7)
    )


def _joined_family(description, mcc, amount):
    """The family of the stream one extra payment joins in `_six_streams`, or None."""
    base = frame(_six_streams("C1"))
    extra = frame([tx("C1", "2025-09-20T12:00:00", amount, description, mcc)])
    before = detect_streams(base).set_index("family").n_payments
    after = detect_streams(pd.concat([base, extra], ignore_index=True)).set_index("family").n_payments
    assert list(after.index) == list(before.index), "an extra payment must never add or split a stream"
    grew = after[after != before]
    assert (grew - before[grew.index]).tolist() in ([], [1]), grew
    return grew.index[0] if len(grew) else None


@pytest.mark.parametrize(
    "description, mcc, amount, family",
    [
        # a filler takes the family of its MCC, even when another family's stream has its amount
        ("member plan", "5812", 18, "streaming"),
        ("member plan", "5812", 66, None),
        ("subscription charge", "7997", 66, "gym"),
        ("subscription charge", "7997", 18, None),
        ("digital service", "6300", 109, "insurance"),
        ("digital service", "6300", 47, None),
        # and must fit its stream's amount within the amount tolerance
        ("member plan", "7997", 66 * np.exp(0.9 * TOLERANCE), "gym"),
        ("member plan", "7997", 66 * np.exp(-0.9 * TOLERANCE), "gym"),
        ("member plan", "7997", 66 * np.exp(1.1 * TOLERANCE), None),
        ("member plan", "7997", 66 * np.exp(-1.1 * TOLERANCE), None),
        ("member plan", "7997", 80, None),
        ("member plan", "7997", 100, None),
    ],
)
def test_a_filler_joins_only_a_stream_whose_amount_and_mcc_it_fits(description, mcc, amount, family):
    assert _joined_family(description, mcc, amount) == family


def test_same_family_streams_at_different_amounts_each_take_only_the_fillers_near_their_amount():
    rows = series("C1", "gym membership", "7997", 45, "2025-06-05", 7)
    rows += series("C1", "gym membership", "7997", 66, "2025-06-07", 7)
    rows += [
        tx("C1", "2025-08-20", 45, "member plan", "7997"),
        tx("C1", "2025-09-20", 66, "member plan", "7997"),
        tx("C1", "2025-10-20", 66, "subscription charge", "7997"),
        tx("C1", "2025-11-20", 55, "member plan", "7997"),  # between the two, fits neither
    ]
    # two gym streams whose amounts lie within the tolerance of a filler: the nearer one takes it
    rows += series("C2", "gym membership", "7997", 66, "2025-06-05", 7)
    rows += series("C2", "gym membership", "7997", 72, "2025-06-07", 7)
    rows += [tx("C2", "2025-09-20", 69.5, "member plan", "7997")]
    streams = detect_streams(frame(rows))
    counts = {(s.client_id, s.median_amount): s.n_payments for s in streams.itertuples()}
    assert set(streams.family) == {"gym"}
    assert counts == {("C1", 45): 8, ("C1", 66): 9, ("C2", 66): 7, ("C2", 72): 8}


# --- ambiguous descriptions the MCC vote leaves unresolved ------------------------


@pytest.mark.parametrize(
    "description, mcc, amount, family",
    [
        # joins a fitting stream of its MCC's home family
        ("digital plus", "7997", 66, "gym"),
        ("premium plan", "6300", 109, "insurance"),
        ("service plan", "5734", 39, "software"),
        # a stray MCC: joins a fitting stream of a family the description can mean
        ("premium plan", "5732", 39, "software"),
        ("service plan", "5734", 6.8, "cloud"),
        ("digital plus", "5411", 47, "mobile"),
        # otherwise joins no stream
        ("digital plus", "7997", 90, None),
        ("premium plan", "6300", 66, None),
        ("premium plan", "5411", 66, None),
        ("service plan", "5734", 18, None),
    ],
)
def test_an_unresolved_ambiguous_payment_joins_only_a_fitting_stream_of_its_home_or_described_family(
    description, mcc, amount, family
):
    assert _joined_family(description, mcc, amount) == family


# --- per-stream MCC, description and share of family-specific descriptions -------
# E2 builds its slot MCC and `none` signals from these columns, never from raw transactions.


def test_a_stream_reports_its_modal_mcc_and_description_and_its_share_of_family_specific_descriptions():
    # 4 "gym membership" payments (a family keyword), 2 Filler Descriptions ("member plan") and one
    # abbreviated noisy variant; 5 on the home MCC and 2 on a stray one.
    rows = series("C1", "gym membership", "7997", 66, "2025-06-05", 4)
    rows += series("C1", "member plan", "7997", 66, "2025-10-03", 2)
    rows += series("C1", "billing gym membership online", "5999", 66, "2025-12-02", 1)
    rows += series("C1", "gym membership", "5999", 66, "2025-06-20", 1)
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments) == ("gym", 8)
    assert s.mcc == "7997"
    assert s.description == "gym membership"
    # family keyword payments: 4 + 1 + 1 of 8 ("member plan" names no family)
    assert s.family_description_share == pytest.approx(6 / 8)


def test_a_stream_of_ambiguous_descriptions_has_no_family_specific_descriptions():
    rows = series("C1", "prem plan", "5812", 18, "2025-06-05", 3)
    rows += series("C1", "premium plan", "5812", 18, "2025-09-03", 2)
    rows += series("C1", "prem plan", "5812", 18, "2025-11-02", 1)
    s = one(detect_streams(frame(rows)))
    assert s.family == "streaming"
    assert s.description == "premium plan"  # abbreviations expanded, so both spellings count as one
    assert s.family_description_share == 0.0


def test_a_stream_mostly_paid_on_a_stray_mcc_reports_that_mcc():
    rows = series("C1", "cloud backup", "5732", 6.8, "2025-06-05", 3)
    rows += series("C1", "cloud backup", "5999", 6.8, "2025-09-03", 4)
    s = one(detect_streams(frame(rows)))
    assert (s.family, s.n_payments, s.mcc) == ("cloud", 7, "5999")
    assert s.family_description_share == 1.0


# --- Pseudo-Labels at a Shifted Cutoff -------------------------------------------
#
# The default Shifted Cutoff is 2025-10-03: its Horizon runs to 2025-12-31, the last observed day.

SHIFTED = pd.Timestamp("2025-10-03", tz="UTC")


def history(rows):
    """A transaction table cut at the end of the known history (every history ends 2025-12-31)."""
    df = frame(rows)
    return df[df["timestamp"] < CUTOFF].reset_index(drop=True)


def monthly(client, description, mcc, amount, start):
    return series(client, description, mcc, amount, start, 20)


def biweekly(client, description, mcc, amount, start):
    return series(client, description, mcc, amount, start, 40, every_days=14)


def test_of_two_streams_due_in_the_horizon_the_earlier_payment_wins():
    rows = (
        # C1: biweekly music first pays on 2025-10-10, monthly gym on 2025-10-19
        biweekly("C1", "audio streaming", "5812", 13.5, "2025-06-06")
        + monthly("C1", "gym membership", "7997", 66, "2025-04-22")
        # C2: monthly gym first pays on 2025-10-05, biweekly music on 2025-10-06
        + monthly("C2", "gym membership", "7997", 66, "2025-04-08")
        + biweekly("C2", "audio streaming", "5812", 13.5, "2025-06-16")
    )
    labels = pseudo_labels(history(rows), SHIFTED)
    assert labels.to_dict() == {"C1": "music", "C2": "gym"}


def test_the_horizon_starts_at_the_cutoff_itself():
    rows = monthly("C1", "cloud backup", "5732", 6.8, "2025-07-05")  # pays 2025-10-03T00:00
    rows += monthly("C1", "phone contract", "4814", 47, "2025-07-06")
    assert pseudo_labels(history(rows), SHIFTED)["C1"] == "cloud"


def test_a_stream_that_stopped_before_the_shifted_cutoff_is_none():
    rows = series("C1", "gym membership", "7997", 66, "2025-01-10", 8)  # last payment 2025-08-08
    rows += biweekly("C1", "grocery store", "5411", 45, "2025-01-03")  # shopping goes on
    assert pseudo_labels(history(rows), SHIFTED)["C1"] == "none"


def test_a_client_with_no_horizon_payments_is_none_and_still_gets_a_label():
    rows = series("C1", "gym membership", "7997", 66, "2025-01-10", 8)
    rows += series("C2", "salary", "0000", 5000, "2025-01-25", 12, type_="transfer", direction="in")
    labels = pseudo_labels(history(rows), SHIFTED)
    assert labels.to_dict() == {"C1": "none", "C2": "none"}


def test_a_new_stream_that_starts_inside_the_horizon_and_repeats_counts():
    rows = series("C1", "gym membership", "7997", 66, "2025-01-10", 8)  # stopped in August
    rows += series("C1", "saas suite", "5734", 19, "2025-10-20", 3)  # new: 10-20, 11-19, 12-19
    rows += monthly("C1", "phone contract", "4814", 47, "2025-06-01")  # first due 2025-10-29
    assert pseudo_labels(history(rows), SHIFTED)["C1"] == "software"

    # at an earlier Shifted Cutoff only its first payment is in the Horizon (to 2025-10-29); its
    # repeats come later in the known history, and still make it a stream
    rows = series("C2", "saas suite", "5734", 19, "2025-10-20", 3)
    earlier = pd.Timestamp("2025-07-31", tz="UTC")
    assert pseudo_labels(history(rows), earlier)["C2"] == "software"
    assert pseudo_labels(history(rows), earlier, min_payments=4)["C2"] == "none"


def test_a_payment_that_never_repeats_is_not_a_pseudo_label():
    rows = series("C1", "saas suite", "5734", 19, "2025-10-20", 1)  # one-off
    rows += monthly("C1", "phone contract", "4814", 47, "2025-06-01")  # first due 2025-10-29
    rows += series("C2", "cloud backup", "5732", 6.8, "2025-11-11", 1)
    assert pseudo_labels(history(rows), SHIFTED).to_dict() == {"C1": "mobile", "C2": "none"}


def test_the_minimum_number_of_stream_payments_is_a_parameter():
    rows = series("C1", "saas suite", "5734", 19, "2025-10-05", 2)  # 2025-10-05 and 2025-11-04
    rows += monthly("C1", "phone contract", "4814", 47, "2025-06-01")  # 8 payments, first due 2025-10-29
    rows += series("C2", "saas suite", "5734", 19, "2025-10-05", 2)
    assert pseudo_labels(history(rows), SHIFTED).to_dict() == {"C1": "software", "C2": "software"}
    assert pseudo_labels(history(rows), SHIFTED, min_payments=3).to_dict() == {"C1": "mobile", "C2": "none"}
    assert pseudo_labels(history(rows), SHIFTED, min_payments=9).to_dict() == {"C1": "none", "C2": "none"}


def test_the_minimum_counts_payments_on_both_sides_of_the_shifted_cutoff():
    # two payments before the Shifted Cutoff and one inside the Horizon make a three-payment stream
    rows = series("C1", "gym membership", "7997", 66, "2025-08-20", 3)  # 08-20, 09-19, 10-19
    assert pseudo_labels(history(rows), SHIFTED, min_payments=3)["C1"] == "gym"


@pytest.mark.parametrize("min_payments", [0, 1])
def test_a_minimum_below_two_payments_is_refused(min_payments):
    rows = monthly("C1", "gym membership", "7997", 66, "2025-04-22")
    with pytest.raises(ValueError, match="at least 2"):
        pseudo_labels(history(rows), SHIFTED, min_payments=min_payments)


def _gym_first_due_on_10_19(client="C1"):
    return monthly(client, "gym membership", "7997", 66, "2025-04-22")


@pytest.mark.parametrize(
    "first",
    [
        # Decoy Transactions, including ones that shadow the gym stream's amount and MCC
        tx("C1", "2025-10-04", 66, "merchant charge", "7997"),
        tx("C1", "2025-10-04", 19, "digital order", "5734"),
        tx("C1", "2025-10-04", 66, "pay card purchase", "7997"),
        # one-off payments: another family, and the gym family at an amount no stream has
        tx("C1", "2025-10-04", 9.9, "cloud backup", "5732"),
        tx("C1", "2025-10-04", 25, "gym membership", "7997"),
        # a refund of the gym payment, and a shop payment
        tx("C1", "2025-10-04", 66, "gym membership", "7997", type_="refund", direction="in"),
        tx("C1", "2025-10-04", 45, "grocery store", "5411"),
    ],
    ids=["decoy", "decoy-other-mcc", "decoy-variant", "one-off", "one-off-same-family", "refund", "shop"],
)
def test_a_non_stream_first_horizon_payment_is_ignored(first):
    rows = _gym_first_due_on_10_19() + [first]
    assert pseudo_labels(history(rows), SHIFTED)["C1"] == "gym"


def test_a_filler_description_inside_a_stream_takes_the_streams_family():
    rows = _gym_first_due_on_10_19()
    first_in_horizon = next(r for r in rows if r["timestamp"] >= SHIFTED)
    first_in_horizon["description"] = "member plan"
    rows += monthly("C1", "phone contract", "4814", 47, "2025-06-01")  # first due 2025-10-29
    assert pseudo_labels(history(rows), SHIFTED)["C1"] == "gym"

    # control: the same filler without its stream is no Pseudo-Label
    rows = [{**first_in_horizon, "client_id": "C2"}, *monthly("C2", "phone contract", "4814", 47, "2025-06-01")]
    assert pseudo_labels(history(rows), SHIFTED)["C2"] == "mobile"


@pytest.mark.parametrize(
    "cutoff, horizon",
    [
        (pd.Timestamp("2025-10-04", tz="UTC"), HORIZON),
        (CUTOFF, HORIZON),
        (pd.Timestamp("2025-09-01", tz="UTC"), pd.Timedelta(days=150)),
    ],
)
def test_a_horizon_that_ends_after_the_last_observed_day_is_refused(cutoff, horizon):
    rows = monthly("C1", "gym membership", "7997", 66, "2025-04-22")
    with pytest.raises(ValueError, match="2025-12-31") as refused:
        pseudo_labels(history(rows), cutoff, horizon)
    latest = (CUTOFF - horizon).strftime("%Y-%m-%d")  # 2025-10-03 for the 90-day Horizon
    assert f"latest Shifted Cutoff is {latest}" in str(refused.value)


def test_an_earlier_shifted_cutoff_and_a_shorter_horizon_are_arguments():
    rows = _gym_first_due_on_10_19()  # ... 2025-07-21, 2025-08-20, 2025-09-19, 2025-10-19
    rows += monthly("C1", "phone contract", "4814", 47, "2025-06-01")  # ... 2025-08-30, 2025-09-29
    august = pd.Timestamp("2025-08-25", tz="UTC")
    assert pseudo_labels(history(rows), august)["C1"] == "mobile"
    assert pseudo_labels(history(rows), SHIFTED, pd.Timedelta(days=10))["C1"] == "none"
    assert pseudo_labels(history(rows), SHIFTED, pd.Timedelta(days=20))["C1"] == "gym"


def _horizon_histories():
    return (
        # C1: gym first due 2025-10-17, streaming ("premium plan" at 17.9 on 5812) 2025-10-24 -> gym
        monthly("C1", "gym membership", "7997", 66, "2025-05-20")
        + monthly("C1", "premium plan", "5812", 17.9, "2025-03-28")
        + biweekly("C1", "grocery store", "5411", 45, "2025-06-01")
        # C2: insurance 2025-10-09, mobile 2025-10-22, cloud 2025-11-01 -> insurance
        + monthly("C2", "phone contract", "4814", 47, "2025-01-25")
        + monthly("C2", "cloud backup", "5732", 6.8, "2025-09-02")
        + monthly("C2", "insurance monthly", "6300", 109, "2025-02-11")
        # C3: software starts in the Horizon; the biweekly music stream stopped in May -> software
        + series("C3", "saas billing", "5734", 39, "2025-10-10", 3)
        + series("C3", "audio streaming", "5812", 13.5, "2025-04-01", 5, every_days=14)
        # C4: its only stream stopped in the summer -> none
        + series("C4", "phone contract", "4814", 47, "2025-01-05", 7)
    )


_HORIZON_LABELS = {"C1": "gym", "C2": "insurance", "C3": "software", "C4": "none"}


def test_injecting_decoys_leaves_every_pseudo_label_unchanged():
    base = history(_horizon_histories())
    assert pseudo_labels(base, SHIFTED).to_dict() == _HORIZON_LABELS

    rng = np.random.default_rng(0)
    decoys = []
    for client in ["C1", "C2", "C3", "C4"]:
        for _ in range(rng.poisson(4.5) + 3):
            decoys.append(
                tx(
                    client,
                    pd.Timestamp("2024-12-01") + int(rng.integers(0, 395)) * DAY,
                    float(rng.choice([66, 17.9, 47, 6.8, 109, 39, 13.5, 67, 25, 120])),
                    str(rng.choice(DECOYS)),
                    str(rng.choice(DECOY_MCCS)),
                )
            )
    # decoys shadowing a stream's amount and MCC, early in the Horizon and before each label's payment
    shadows = [
        tx("C1", "2025-10-05", 17.9, "merchant charge", "5812"),
        tx("C1", "2025-10-06", 17.9, "pay dgtl order online", "5812"),
        tx("C2", "2025-10-04", 6.8, "card purchase digital", "5732"),
        tx("C2", "2025-10-05", 47, "digital order", "4814"),
        tx("C3", "2025-10-04", 13.5, "service payment", "5812"),
        tx("C3", "2025-10-05", 39, "merchant charge service", "5734"),
        tx("C4", "2025-10-04", 47, "billing merchant charge", "4814"),
        tx("C4", "2025-11-03", 47, "digital order plus", "4814"),
    ]
    decoys += shadows
    noisy = pd.concat([base, history(decoys)], ignore_index=True).sample(frac=1, random_state=1)
    pd.testing.assert_series_equal(pseudo_labels(noisy, SHIFTED), pseudo_labels(base, SHIFTED))

    # control: the same shadows with a Filler Description join a stream and do move labels
    fillers = history([{**d, "description": "subscription charge"} for d in shadows])
    moved = pseudo_labels(pd.concat([base, fillers], ignore_index=True), SHIFTED)
    assert moved["C1"] == "streaming" and moved["C2"] == "cloud"


# --- inputs at a Shifted Cutoff ------------------------------------------------------


def test_inputs_at_a_shifted_cutoff_use_only_the_transactions_before_it():
    from recurring_family.pseudo import milestone2_rule

    base = history(_horizon_histories())
    before = base[base["timestamp"] < SHIFTED]
    # the Horizon rewritten: every payment from the Shifted Cutoff on moves to another family and amount
    rewritten = base.copy()
    after = rewritten["timestamp"] >= SHIFTED
    rewritten.loc[after, ["description", "mcc", "amount"]] = ["cloud backup", "5732", 6.8]
    assert pseudo_labels(rewritten, SHIFTED).to_dict() != pseudo_labels(base, SHIFTED).to_dict()

    clients = pd.Index(["C1", "C2", "C3", "C4"], name="client_id")
    rule = milestone2_rule(SHIFTED)
    expected_streams = detect_streams(before, SHIFTED)
    expected_proba = rule.predict_proba(before, clients)
    for full in (base, rewritten):
        pd.testing.assert_frame_equal(detect_streams(full, SHIFTED), expected_streams)
        pd.testing.assert_frame_equal(rule.predict_proba(full, clients), expected_proba)
        pd.testing.assert_frame_equal(rule.proba_from_streams(detect_streams(full, SHIFTED), clients), expected_proba)
    # C1's gym and C2's insurance are Active Streams due inside the Horizon: the rule predicts from them
    predicted = expected_proba.idxmax(axis=1).to_dict()
    assert predicted["C1"] == "gym" and predicted["C2"] == "insurance"


def test_a_rule_saved_at_a_shifted_cutoff_reloads_at_that_cutoff(tmp_path):
    from recurring_family.pseudo import milestone2_rule
    from recurring_family.rules import RulesModel

    # insurance every month from January, stopping in September: due inside the Shifted Cutoff's
    # Horizon, long overdue at the real Cutoff
    base = history(series("C1", "insurance monthly", "6300", 109, "2025-01-10", 9))
    clients = pd.Index(["C1"], name="client_id")
    rule = milestone2_rule(SHIFTED)
    rule.save(tmp_path / "rule.json")
    loaded = RulesModel.load(tmp_path / "rule.json")
    assert loaded.predict_proba(base, clients).idxmax(axis=1).to_dict() == {"C1": "insurance"}
    assert milestone2_rule().predict_proba(base, clients).idxmax(axis=1).to_dict() == {"C1": "none"}
