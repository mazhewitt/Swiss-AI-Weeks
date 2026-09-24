"""Seam 1: the E3 decision layer (per-class weights plus a `none` threshold) through the CLI."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from recurring_family import data, models

LABELS = ["cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none"]
FAMILIES = LABELS[:-1]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_labels(project, split, labels_by_client):
    pd.DataFrame(
        {
            "client_id": list(labels_by_client),
            "cutoff_date": "2026-01-01",
            "target_next_recurring_merchant": list(labels_by_client.values()),
        }
    ).to_csv(project.raw / f"{split}_labels.csv", index=False)


def read_split(project, capsys):
    capsys.readouterr()
    assert project.run("split") == 0
    return pd.read_csv(project.root / "artifacts" / "valid_split.csv", dtype=str)


def predictions(project, row):
    """The per-Client predictions of a logged run, committed next to the experiment log."""
    path = project.log.parent / "runs" / f"{row['run_id']}.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]


def evaluate(project, *args):
    assert project.run("evaluate", *args) == 0
    row = read_rows(project.log)[-1]
    return row, predictions(project, row)


def decision_file(project, name, weights=None, none_threshold=None):
    path = project.root / f"{name}.json"
    path.write_text(json.dumps({"weights": weights or {l: 1.0 for l in LABELS}, "none_threshold": none_threshold}))
    return str(path)


class FixtureProba(models.PriorModel):
    """Fixture probabilities: ignores what it is fitted on and answers from a per-Client table."""

    name = "fixture"
    table: dict = {}

    def predict_proba(self, transactions, clients):
        rows = [[self.table[c].get(l, 0.0) for l in LABELS] for c in clients]
        return pd.DataFrame(rows, index=pd.Index(clients, name="client_id"), columns=LABELS)


@pytest.fixture
def fixture_model(monkeypatch):
    monkeypatch.setattr(FixtureProba, "table", {})
    monkeypatch.setitem(models.MODELS, "fixture", FixtureProba)
    return FixtureProba


class Memoriser(models.PriorModel):
    """Recalls the label of every Client it was fitted on; uniform for anyone else."""

    name = "memoriser"

    def fit(self, transactions, labels):
        self.seen = dict(labels)
        return self

    def predict_proba(self, transactions, clients):
        rows = [
            [1.0 if self.seen.get(c) == l else 0.0 for l in LABELS] if c in self.seen else [1 / 8] * 8
            for c in clients
        ]
        return pd.DataFrame(rows, index=pd.Index(clients, name="client_id"), columns=LABELS)

    def save(self, path):
        path.write_text(json.dumps({"seen": self.seen}))

    @classmethod
    def load(cls, path):
        model = cls()
        model.seen = json.loads(path.read_text())["seen"]
        return model


@pytest.fixture
def memoriser(monkeypatch):
    monkeypatch.setitem(models.MODELS, "memoriser", Memoriser)
    return Memoriser


def big_train(project, n_none=20, n_gym=10, n_cloud=7):
    labels = {f"T{i:04d}": "none" for i in range(n_none)}
    labels.update({f"T{100 + i:04d}": "gym" for i in range(n_gym)})
    labels.update({f"T{200 + i:04d}": "cloud" for i in range(n_cloud)})
    write_labels(project, "train", labels)
    return labels


def big_valid(project, n_none=30, n_gym=10, n_cloud=10):
    labels = {f"VN{i:04d}": "none" for i in range(n_none)}
    labels.update({f"VG{i:04d}": "gym" for i in range(n_gym)})
    labels.update({f"VC{i:04d}": "cloud" for i in range(n_cloud)})
    write_labels(project, "valid", labels)
    return labels


def mirrored(project, capsys, model, groups):
    """Valid Clients in `groups` (label, probabilities, count); train is an exact copy of the
    selection set, so the decision layer's fitting data and the evaluated data are identical.
    Returns the selection set's true labels."""
    labels, table = {}, {}
    for g, (label, proba, count) in enumerate(groups):
        for i in range(count):
            client = f"V{g}{i:03d}"
            labels[client] = label
            table[client] = proba
    write_labels(project, "valid", labels)
    selection = read_split(project, capsys).query("set == 'selection'")["client_id"]
    write_labels(project, "train", {f"T{c}": labels[c] for c in selection})
    table.update({f"T{c}": table[c] for c in selection})
    model.table.update(table)
    return pd.Series({c: labels[c] for c in selection})


# --- fixed decisions: argmax and the none threshold ----------------------------


# Every row's argmax, by eye; ties go to the earlier label in the fixed label order.
ARGMAX_ROWS = {
    "C000011": ({"gym": 0.5, "none": 0.3, "cloud": 0.2}, "gym"),
    "C000012": ({"none": 0.6, "music": 0.3, "streaming": 0.1}, "none"),
    "C000013": ({"cloud": 0.4, "gym": 0.4, "none": 0.2}, "cloud"),  # tie -> cloud
    "C000014": ({"streaming": 0.35, "music": 0.35, "none": 0.3}, "music"),  # tie -> music
}


def test_uniform_weights_without_threshold_reproduce_argmax_exactly(fetched, fixture_model):
    fixture_model.table.update({c: p for c, (p, _) in ARGMAX_ROWS.items()})
    write_labels(fetched, "train", {c: "none" for c in ARGMAX_ROWS})
    assert fetched.run("train", "--model", "fixture") == 0

    _, argmax = evaluate(fetched, "--model", "fixture", "--split", "train")
    _, uniform = evaluate(fetched, "--model", "fixture", "--split", "train", "--decision", decision_file(fetched, "uniform"))
    assert sorted(uniform.index) == sorted(ARGMAX_ROWS)
    assert uniform.to_dict() == argmax.to_dict() == {c: label for c, (_, label) in ARGMAX_ROWS.items()}


def test_hand_set_weights_change_the_decision(fetched, fixture_model):
    fixture_model.table.update({c: p for c, (p, _) in ARGMAX_ROWS.items()})
    write_labels(fetched, "train", {c: "none" for c in ARGMAX_ROWS})
    assert fetched.run("train", "--model", "fixture") == 0
    weights = {**{l: 1.0 for l in LABELS}, "cloud": 3.0, "streaming": 2.0}
    _, weighted = evaluate(fetched, "--model", "fixture", "--split", "train", "--decision", decision_file(fetched, "w", weights))
    # cloud 0.2*3 = 0.6 beats gym 0.5; none 0.6 still beats streaming 0.2; streaming 0.7 beats music
    assert weighted.to_dict() == {"C000011": "cloud", "C000012": "none", "C000013": "cloud", "C000014": "streaming"}
    # with a none threshold (none when P(none) >= 0.5: only C000012) the weights still pick the family
    _, thresholded = evaluate(
        fetched, "--model", "fixture", "--split", "train", "--decision", decision_file(fetched, "wt", weights, 0.5)
    )
    assert thresholded.to_dict() == {"C000011": "cloud", "C000012": "none", "C000013": "cloud", "C000014": "streaming"}


THRESHOLD_ROWS = {
    "C000011": {"gym": 0.5, "none": 0.3, "cloud": 0.2},
    "C000012": {"none": 1.0},  # the model is sure there is no family at all
    "C000013": {"cloud": 0.9, "none": 0.1},
    "C000014": {"music": 0.2, "streaming": 0.1, "none": 0.7},
    "C000011X": {"software": 1.0},  # the model is sure of a family (P(none) = 0)
}


@pytest.fixture
def threshold_rows(fetched, fixture_model):
    fixture_model.table.update(THRESHOLD_ROWS)
    write_labels(fetched, "train", {c: "none" for c in THRESHOLD_ROWS})
    assert fetched.run("train", "--model", "fixture") == 0
    return fetched


def test_none_threshold_of_zero_yields_no_none(threshold_rows):
    _, decided = evaluate(threshold_rows, "--model", "fixture", "--split", "train", "--decision", decision_file(threshold_rows, "t0", none_threshold=0.0))
    assert "none" not in set(decided)
    # the family each Client gets is its best family
    assert decided.to_dict() == {
        "C000011": "gym", "C000012": "cloud", "C000013": "cloud", "C000014": "music", "C000011X": "software",
    }


def test_none_threshold_of_one_yields_all_none(threshold_rows):
    _, decided = evaluate(threshold_rows, "--model", "fixture", "--split", "train", "--decision", decision_file(threshold_rows, "t1", none_threshold=1.0))
    assert len(decided) == len(THRESHOLD_ROWS)
    assert set(decided) == {"none"}


def test_none_threshold_in_between_calls_none_on_confident_none(threshold_rows):
    # none when P(none) >= 1 - threshold: here P(none) >= 0.4
    _, decided = evaluate(threshold_rows, "--model", "fixture", "--split", "train", "--decision", decision_file(threshold_rows, "t6", none_threshold=0.6))
    assert decided.to_dict() == {
        "C000011": "gym", "C000012": "none", "C000013": "cloud", "C000014": "none", "C000011X": "software",
    }


# --- the tuned decision layer --------------------------------------------------


# argmax calls every cloud Client gym (needs a cloud weight) and every "unsure none"
# Client gym too (needs the none threshold or a none weight).
HARD_GROUPS = [
    ("none", {"none": 0.7, "gym": 0.2, "cloud": 0.1}, 10),
    ("none", {"gym": 0.46, "none": 0.44, "cloud": 0.1}, 10),
    ("gym", {"gym": 0.7, "none": 0.1, "cloud": 0.2}, 10),
    ("cloud", {"gym": 0.45, "cloud": 0.35, "none": 0.2}, 10),
]


def test_tuned_decision_fits_weights_and_threshold_that_beat_argmax(fetched, fixture_model, capsys):
    truth = mirrored(fetched, capsys, fixture_model, HARD_GROUPS)
    assert fetched.run("train", "--model", "fixture", "--decision", "tuned") == 0

    argmax_row, argmax = evaluate(fetched, "--model", "fixture")
    tuned_row, tuned = evaluate(fetched, "--model", "fixture", "--decision", "tuned")
    assert set(argmax) == {"gym", "none"}  # argmax never predicts cloud
    assert tuned.reindex(truth.index).to_dict() == truth.to_dict()
    assert float(tuned_row["macro_f1"]) == pytest.approx(3 / 8, abs=1e-4)  # three labels, all perfect
    assert float(tuned_row["macro_f1"]) > float(argmax_row["macro_f1"])
    assert tuned_row["model"] == "fixture+tuned"
    assert argmax_row["model"] == "fixture"


# Only the none threshold separates these: `none` must win at P(none) 0.40 against gym 0.60,
# yet lose at P(none) 0.35 against gym 0.20, which no none weight can do (it would need
# w_none / w_gym > 1.5 and < 0.58 at once). none iff P(none) >= 0.4 gets all three right.
THRESHOLD_ONLY_GROUPS = [
    ("none", {"none": 0.40, "gym": 0.60}, 10),
    ("gym", {"gym": 0.70, "none": 0.30}, 10),
    ("gym", {"none": 0.35, "gym": 0.20, "software": 0.15, "cloud": 0.10, "music": 0.10, "streaming": 0.10}, 10),
]


def test_tuned_decision_fits_a_none_threshold_where_no_weights_would_do(fetched, fixture_model, capsys):
    truth = mirrored(fetched, capsys, fixture_model, THRESHOLD_ONLY_GROUPS)
    assert fetched.run("train", "--model", "fixture", "--decision", "tuned") == 0
    _, argmax = evaluate(fetched, "--model", "fixture")
    tuned_row, tuned = evaluate(fetched, "--model", "fixture", "--decision", "tuned")
    assert set(argmax.reindex(truth.index)[truth == "none"]) == {"gym"}
    assert "none" in set(argmax.reindex(truth.index)[truth == "gym"])  # the unsure gym Clients
    assert tuned.reindex(truth.index).to_dict() == truth.to_dict()
    assert float(tuned_row["macro_f1"]) == pytest.approx(2 / 8, abs=1e-4)


def random_groups(seed):
    rng = np.random.default_rng(seed)
    groups = []
    for label in ["none", "none", "gym", "cloud", "music", "streaming"]:
        p = rng.dirichlet(np.ones(8) * 0.7)
        groups.append((label, dict(zip(LABELS, p)), 6))
    return groups


# argmax is perfect by margins so thin that moving any one weight a grid step, or using
# any none threshold, breaks some group: only argmax itself stays perfect.
PERFECT_ARGMAX = [
    ("none", {"none": 0.335, "gym": 0.333, "cloud": 0.332}, 10),
    ("gym", {"gym": 0.335, "none": 0.333, "cloud": 0.332}, 10),
    ("cloud", {"cloud": 0.335, "gym": 0.333, "none": 0.332}, 10),
]


@pytest.mark.parametrize("groups", [PERFECT_ARGMAX, HARD_GROUPS, *[random_groups(s) for s in range(4)]])
def test_tuned_decision_never_scores_below_argmax_on_its_fitting_data(fetched, fixture_model, capsys, groups):
    mirrored(fetched, capsys, fixture_model, groups)  # selection == the decision's fitting data
    assert fetched.run("train", "--model", "fixture", "--decision", "tuned") == 0
    argmax_row, _ = evaluate(fetched, "--model", "fixture")
    tuned_row, _ = evaluate(fetched, "--model", "fixture", "--decision", "tuned")
    assert float(tuned_row["macro_f1"]) >= float(argmax_row["macro_f1"])
    if groups is PERFECT_ARGMAX:
        assert float(tuned_row["macro_f1"]) == pytest.approx(3 / 8, abs=1e-4)


def test_tuned_decision_is_fitted_on_out_of_fold_probabilities(fetched, memoriser, capsys):
    # Leakage trap. In-sample the memoriser is perfect, so a decision fitted on in-sample
    # probabilities keeps argmax, which calls every uniform (unseen) Client `cloud`.
    # Out of fold it is uniform, so the tuned decision learns the best constant on the
    # train labels: `none`. Fitted on the evaluated (selection) Clients it would learn `gym`.
    big_train(fetched)  # 20 none, 10 gym, 7 cloud
    big_valid(fetched, n_none=5, n_gym=40, n_cloud=5)
    assert fetched.run("train", "--model", "memoriser", "--decision", "tuned") == 0
    _, argmax = evaluate(fetched, "--model", "memoriser")
    _, tuned = evaluate(fetched, "--model", "memoriser", "--decision", "tuned")
    assert set(argmax) == {"cloud"}
    assert set(tuned) == {"none"}
    assert len(tuned) == 35


def test_tuned_decision_refit_with_selection_learns_from_selection_labels_not_holdout(fetched, memoriser, capsys):
    # With --with-selection the out-of-fold fit covers train plus the selection set: the
    # selection set's gym Clients outnumber train's none Clients, so the best constant is gym.
    big_train(fetched)  # 20 none, 10 gym, 7 cloud
    big_valid(fetched, n_none=0, n_gym=50, n_cloud=0)  # selection: 35 gym
    assert fetched.run("train", "--model", "memoriser", "--decision", "tuned", "--with-selection") == 0
    row, tuned = evaluate(fetched, "--model", "memoriser", "--decision", "tuned", "--checkpoint")
    assert row["split"] == "holdout" and len(tuned) == 15
    assert set(tuned) == {"gym"}

    assert fetched.run("train", "--model", "memoriser", "--decision", "tuned") == 0
    _, train_only = evaluate(fetched, "--model", "memoriser", "--decision", "tuned", "--checkpoint")
    assert set(train_only) == {"none"}


def test_tuned_decision_is_never_scored_on_the_clients_it_was_fitted_on(fetched, capsys):
    big_train(fetched)
    assert fetched.run("train", "--model", "prior", "--decision", "tuned") == 0
    capsys.readouterr()
    assert fetched.run("evaluate", "--model", "prior", "--split", "train", "--decision", "tuned") != 0
    assert "fitted on" in capsys.readouterr().err
    assert not fetched.log.exists()
    # argmax on train stays possible, as before
    assert fetched.run("evaluate", "--model", "prior", "--split", "train") == 0


def test_prior_model_with_tuned_decision_predicts_every_client(fetched, capsys):
    big_train(fetched)
    big_valid(fetched)
    selection = read_split(fetched, capsys).query("set == 'selection'")["client_id"]
    assert fetched.run("train", "--model", "prior", "--decision", "tuned") == 0
    row, tuned = evaluate(fetched, "--model", "prior", "--decision", "tuned", "--change", "prior + E3")
    assert sorted(tuned.index) == sorted(selection)
    assert set(tuned) <= set(LABELS)
    assert row["model"] == "prior+tuned" and row["change"] == "prior + E3"

    assert fetched.run("submit", "--model", "prior", "--decision", "tuned", "--name", "e3") == 0
    out = fetched.root / "submissions" / "e3.csv"
    assert fetched.run("submit", "--check", str(out)) == 0
    submitted = pd.read_csv(out, dtype=str)
    assert submitted["client_id"].tolist() == ["C000004", "C000008", "C000010"]


def test_submit_applies_the_tuned_decision(fetched, fixture_model, capsys):
    mirrored(fetched, capsys, fixture_model, HARD_GROUPS)
    # the test Clients look like an unsure-none, a cloud and a gym Client
    fixture_model.table.update(
        {"C000004": HARD_GROUPS[1][1], "C000008": HARD_GROUPS[3][1], "C000010": HARD_GROUPS[2][1]}
    )
    assert fetched.run("train", "--model", "fixture", "--decision", "tuned") == 0
    assert fetched.run("submit", "--model", "fixture", "--decision", "tuned", "--name", "e3") == 0
    assert fetched.run("submit", "--model", "fixture", "--name", "argmax") == 0
    col = "predicted_next_recurring_merchant"
    tuned = pd.read_csv(fetched.root / "submissions" / "e3.csv", dtype=str).set_index("client_id")[col]
    argmax = pd.read_csv(fetched.root / "submissions" / "argmax.csv", dtype=str).set_index("client_id")[col]
    assert tuned.to_dict() == {"C000004": "none", "C000008": "cloud", "C000010": "gym"}
    assert argmax.to_dict() == {"C000004": "gym", "C000008": "gym", "C000010": "gym"}


def test_tuned_decision_needs_a_model_trained_with_it(fetched, capsys):
    big_train(fetched)
    assert fetched.run("train", "--model", "prior", "--decision", "tuned") == 0
    assert fetched.run("train", "--model", "prior") == 0  # retrained without: the old decision is stale
    capsys.readouterr()
    assert fetched.run("evaluate", "--model", "prior", "--decision", "tuned") != 0
    assert "--decision tuned" in capsys.readouterr().err
    assert not fetched.log.exists()


# --- leakage guards --------------------------------------------------------------


class LeakyPredictor(models.PriorModel):
    """Reads valid labels only when predicting, i.e. only while the decision layer is fitted."""

    name = "leaky_predictor"

    def predict_proba(self, transactions, clients):
        data.load_labels(self.raw, self.peek)
        return super().predict_proba(transactions, clients)


# a refit with --with-selection may read the selection set, but never the holdout or all of valid
@pytest.mark.parametrize(
    ("peek", "with_selection"),
    [("selection", False), ("holdout", False), ("valid", False), ("holdout", True), ("valid", True)],
)
def test_fitting_the_decision_layer_never_reads_valid_labels(fetched, monkeypatch, capsys, peek, with_selection):
    big_train(fetched)
    big_valid(fetched)
    monkeypatch.setattr(LeakyPredictor, "raw", fetched.raw, raising=False)
    monkeypatch.setattr(LeakyPredictor, "peek", peek, raising=False)
    monkeypatch.setitem(models.MODELS, "leaky_predictor", LeakyPredictor)
    flags = ["--with-selection"] if with_selection else []
    assert fetched.run("train", "--model", "leaky_predictor", *flags) == 0  # the model alone never predicts
    capsys.readouterr()
    assert fetched.run("train", "--model", "leaky_predictor", "--decision", "tuned", *flags) != 0
    err = capsys.readouterr().err
    assert "valid labels" in err and "training" in err


class TransactionRecorder(models.PriorModel):
    """Records whose transactions every fit and every prediction was given."""

    name = "recorder"
    seen: list = []

    def fit(self, transactions, labels):
        type(self).seen.append(set(transactions["client_id"]) | set(labels.index))
        return super().fit(transactions, labels)

    def predict_proba(self, transactions, clients):
        type(self).seen.append(set(transactions["client_id"]) | set(clients))
        return super().predict_proba(transactions, clients)


@pytest.mark.parametrize("with_selection", [False, True])
def test_tuned_decision_is_fitted_on_training_clients_never_the_holdout(fetched, monkeypatch, capsys, with_selection):
    monkeypatch.setattr(TransactionRecorder, "seen", [])
    monkeypatch.setitem(models.MODELS, "recorder", TransactionRecorder)
    big_valid(fetched)
    split = read_split(fetched, capsys)
    selection = set(split.query("set == 'selection'")["client_id"])
    holdout = set(split.query("set == 'holdout'")["client_id"])
    train = set(pd.read_csv(fetched.raw / "train_labels.csv", dtype=str)["client_id"])
    # the fixture's six train Clients are too few for the default 5 folds
    flags = ["--with-selection"] if with_selection else []
    assert fetched.run("train", "--model", "recorder", "--decision", "tuned", "--folds", "3", *flags) == 0
    seen = set().union(*TransactionRecorder.seen)
    assert len(TransactionRecorder.seen) == 1 + 3 * 2  # the model, then fit + predict per fold
    assert seen == (train | selection if with_selection else train)
    assert not seen & holdout
