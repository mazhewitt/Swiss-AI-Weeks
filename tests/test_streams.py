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
