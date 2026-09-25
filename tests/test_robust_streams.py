"""The opt-in noise-robust detector (ticket 24, `StreamParams(robust=True)`): Candidate Streams from amount and
periodicity alone, the family by a majority vote of evidence (family keyword 1, home MCC 0.5, Filler or Decoy
0), no-evidence streams dropped, Decoys joining on schedule only, and the default left untouched."""

import pandas as pd
import pytest

from recurring_family.none_model import NoneModel
from recurring_family.ranker import RankerModel
from recurring_family.streams import StreamParams, detect_stream_payments, detect_streams, pseudo_labels
from recurring_family.survival import SurvivalModel
from test_streams import DAY, frame, one, series, tx

ROBUST = StreamParams(robust=True)


def _noisy_gym(client="C1", amount=47.0, start="2025-04-05", n=8):
    """A gym stream booked as valid and test book one: most payments carry a Filler Description, and the two
    that name the family were booked on another MCC, so the default detector has no anchor to start it."""
    descriptions = ["member plan", "fitness club", "subscription charge", "member plan", "digital service",
                    "fitness club", "member plan", "member plan"]
    mccs = ["7997", "5411", "5812", "7997", "7997", "5411", "7997", "7997"]
    first = pd.Timestamp(start)
    return [tx(client, first + i * 30 * DAY, amount, descriptions[i], mccs[i]) for i in range(n)]


def test_robust_is_off_by_default():
    assert StreamParams().robust is False


def test_default_detector_splits_a_noisy_stream_the_robust_one_keeps_it_whole():
    rows = frame(_noisy_gym())
    default = detect_streams(rows)
    assert default.empty or default["n_payments"].max() < 8
    s = one(detect_streams(rows, params=ROBUST))
    assert (s.family, s.n_payments) == ("gym", 8)


def test_family_is_the_majority_vote_not_the_first_keyword():
    # two cloud keywords (2) and three software home MCCs with Filler Descriptions (1.5): cloud
    rows = [
        tx("C1", "2025-04-05", 20, "cloud backup", "5411"),
        tx("C1", "2025-05-05", 20, "member plan", "5734"),
        tx("C1", "2025-06-04", 20, "storage", "5411"),
        tx("C1", "2025-07-04", 20, "member plan", "5734"),
        tx("C1", "2025-08-03", 20, "subscription charge", "5734"),
    ]
    assert one(detect_streams(frame(rows), params=ROBUST)).family == "cloud"
    # two more software home MCCs tip it: 2.5 against 2
    rows += [tx("C1", "2025-09-02", 20, "member plan", "5734"), tx("C1", "2025-10-02", 20, "member plan", "5734")]
    assert one(detect_streams(frame(rows), params=ROBUST)).family == "software"


def test_home_mcc_alone_is_evidence():
    rows = series("C1", "subscription charge", "4814", 30, "2025-04-05", 5)
    s = one(detect_streams(frame(rows), params=ROBUST))
    assert (s.family, s.n_payments) == ("mobile", 5)


def test_a_stream_with_no_family_evidence_is_dropped():
    rows = series("C1", "member plan", "5411", 30, "2025-04-05", 6)
    assert detect_streams(frame(rows), params=ROBUST).empty


def test_shop_payments_never_form_or_join_a_robust_stream():
    rows = series("C1", "grocery store", "7997", 47, "2025-04-20", 6) + series("C1", "gym", "7997", 47, "2025-04-05", 3)
    s = one(detect_streams(frame(rows), params=ROBUST))
    assert (s.family, s.n_payments) == ("gym", 3)


def test_decoy_joins_only_on_the_schedule():
    rows = series("C1", "member plan", "7997", 47, "2025-04-05", 8)
    del rows[4]
    on_slot = tx("C1", pd.Timestamp("2025-04-05") + 120 * DAY, 47, "digital order", "5411")
    off_slot = tx("C1", pd.Timestamp("2025-04-05") + 105 * DAY, 47, "merchant charge", "5411")
    s = one(detect_streams(frame(rows + [on_slot, off_slot]), params=ROBUST))
    assert (s.family, s.n_payments) == ("gym", 8)


def test_decoys_alone_never_form_a_stream_and_carry_no_evidence():
    rows = series("C1", "digital order", "7997", 47, "2025-04-05", 6)
    assert detect_streams(frame(rows), params=ROBUST).empty


def test_two_monthly_streams_chained_by_amount_are_split_by_periodicity():
    gym = series("C1", "member plan", "7997", 93, "2025-04-05", 8)  # 98.5 / 93: within the amount tolerance
    ins = series("C1", "member plan", "6300", 98.5, "2025-04-20", 8)
    st = detect_streams(frame(gym + ins), params=ROBUST).sort_values("family")
    assert list(st["family"]) == ["gym", "insurance"]
    assert list(st["n_payments"]) == [8, 8]


def test_a_biweekly_stream_is_not_split():
    rows = series("C1", "phone contract", "4814", 20, "2025-06-02", 12, every_days=14)
    s = one(detect_streams(frame(rows), params=ROBUST))
    assert (s.family, s.n_payments) == ("mobile", 12)
    assert s.period_days == pytest.approx(14)


def test_music_and_streaming_still_split_by_hint():
    rows = series("C1", "member plan", "5812", 12.9, "2025-04-05", 4) + series("C1", "audio pass", "5812", 12.9, "2025-08-03", 2)
    assert one(detect_streams(frame(rows), params=ROBUST)).family == "music"


def test_payments_and_pseudo_labels_follow_the_option():
    rows = frame(_noisy_gym(start="2025-04-05", n=8))
    streams, paid = detect_stream_payments(rows, params=ROBUST)
    assert len(paid) == 8 and one(streams).n_payments == 8
    labels = pseudo_labels(rows, cutoff=pd.Timestamp("2025-09-01", tz="UTC"), params=ROBUST)
    assert labels["C1"] == "gym"


def _clients(n=24):
    rows, labels = [], {}
    for k in range(n):
        client = f"C{k:02d}"
        if k % 3 == 0:
            labels[client] = "none"
            rows += series(client, "grocery store", "5411", 40, "2025-06-01", 4)
        else:
            labels[client] = "gym"
            rows += _noisy_gym(client, start="2025-05-10", n=8)
    return frame(rows), pd.Series(labels, name="label")


def test_models_take_the_option_and_default_to_the_default_detector():
    tx_, labels = _clients()
    assert RankerModel().stream_params is None and SurvivalModel().stream_params is None
    robust = RankerModel(none_model=NoneModel(), stream_params=ROBUST).fit(tx_, labels)
    plain = RankerModel(none_model=NoneModel()).fit(tx_, labels)
    p_robust = robust.predict_proba(tx_, labels.index)
    p_plain = plain.predict_proba(tx_, labels.index)
    assert not p_robust.equals(p_plain)
    survival = SurvivalModel(order="unprojected-last", stream_params=ROBUST).fit(tx_, labels)
    assert survival.predict_proba(tx_, labels.index).shape == (len(labels), 8)


def test_a_saved_model_keeps_its_detector_setting(tmp_path):
    tx_, labels = _clients()
    for model, cls in ((RankerModel(stream_params=ROBUST), RankerModel), (SurvivalModel(stream_params=ROBUST), SurvivalModel)):
        model.fit(tx_, labels).save(tmp_path / "m.json")
        assert cls.load(tmp_path / "m.json").stream_params == ROBUST
    RankerModel().fit(tx_, labels).save(tmp_path / "d.json")
    assert "stream_params" not in (tmp_path / "d.json").read_text()
    assert RankerModel.load(tmp_path / "d.json").stream_params is None
