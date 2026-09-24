"""The one CLI: fetch-data, split, streams, pseudo-labels, train, cv, features, evaluate, compare, submit."""

import argparse
import csv
import dataclasses
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data
from . import decision as decision_layer
from .comparison import Candidate, compare_candidates, summary_lines, to_markdown
from .config import CUTOFF, HORIZON, LABEL_COLUMN, LABELS, MERCHANT_FAMILIES, SHIFTED_CUTOFF
from .cross_validation import CV_FOLDS, out_of_fold_proba
from .evaluation import append_log, latest_fidelity, log_row, paired_bootstrap, previous_best, score
from .fetch import fetch_data
from .models import MODELS, predict_labels
from .pseudo import (
    FIDELITY_CANDIDATES, NONE_SHARE_TOLERANCE, PSEUDO_MIN_PAYMENTS, RULE_F1_TOLERANCE, fidelity_check, milestone2_rule,
)
from .ranker import PSEUDO_WEIGHT, pseudo_examples
from .rules import DEFAULT_NONE_GATE, DEFAULT_ORDERING, ORDERINGS
from .streams import StreamParams, cached_pseudo_labels, cached_streams
from .submission import InvalidSubmission, read_submission, validate, write_submission

DEFAULT_ZIP = Path("hackathons") / "2026" / "data" / "dataset.zip"


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "data" / "raw"
        self.artifacts = root / "artifacts"
        self.submissions = root / "submissions"
        self.log = root / "experiments" / "log.csv"
        # committed next to the log, so every logged run can be compared against in any clone
        self.runs = root / "experiments" / "runs"

    @property
    def streams(self) -> Path:
        return self.artifacts / "streams"

    @property
    def pseudo_labels(self) -> Path:
        return self.artifacts / "pseudo_labels"

    def pseudo_label_table(self, split: str, cutoff: pd.Timestamp, min_payments: int) -> Path:
        return self.pseudo_labels / f"{split}-{cutoff:%Y-%m-%d}-min{min_payments}.csv"

    def model(self, name: str) -> Path:
        return self.artifacts / f"{name}.json"

    def model_meta(self, name: str) -> Path:
        return self.artifacts / f"{name}.meta.json"

    def decision(self, name: str) -> Path:
        return self.artifacts / f"{name}.decision.json"

    def run_predictions(self, run_id: str) -> Path:
        return self.runs / f"{run_id}.csv"

    @property
    def valid_split(self) -> Path:
        return self.artifacts / "valid_split.csv"


def _load_model(paths: Paths, name: str):
    path = paths.model(name)
    if not path.exists():
        raise FileNotFoundError(f"no trained {name!r} model at {path}; run `train --model {name}` first")
    return MODELS[name].load(path)


def _load_decision(paths: Paths, model: str, choice: str) -> decision_layer.Decision | None:
    """None for plain argmax; the decision fitted by `train --decision tuned`; or a hand-set
    decision layer (weights and none_threshold) from a JSON file."""
    if choice == "argmax":
        return None
    path = paths.decision(model) if choice == "tuned" else Path(choice)
    if not path.exists():
        raise FileNotFoundError(
            f"no tuned decision layer for {model!r} at {path}; run `train --model {model} --decision tuned` first"
            if choice == "tuned" else f"no decision layer file at {path}"
        )
    return decision_layer.Decision.load(path)


def _decide(proba: pd.DataFrame, decision: decision_layer.Decision | None) -> pd.Series:
    return predict_labels(proba) if decision is None else decision.apply(proba)


def _decision_name(choice: str) -> str:
    return "" if choice == "argmax" else "+tuned" if choice == "tuned" else f"+{Path(choice).stem}"


def cmd_fetch_data(args, paths: Paths) -> int:
    zip_path = Path(args.zip) if args.zip else paths.root / DEFAULT_ZIP
    written = fetch_data(zip_path, paths.raw)
    print(f"fetch-data: {len(written)} file(s) written to {paths.raw}" + (f": {', '.join(written)}" if written else ""))
    for split in data.SPLITS:
        tx = data.load_transactions(paths.raw, split)
        first, last = (t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in (tx["timestamp"].min(), tx["timestamp"].max()))
        print(f"  {split}: {tx['client_id'].nunique()} Clients, {len(tx)} transactions, {first} .. {last}")
    return 0


def cmd_split(args, paths: Paths) -> int:
    split = data.valid_split(paths.raw)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    split.to_csv(paths.valid_split, index=False)
    counts = split["set"].value_counts()
    print(
        f"split: valid -> {counts.get('selection', 0)} selection, {counts.get('holdout', 0)} sealed holdout "
        f"Clients -> {paths.valid_split}"
    )
    return 0


def stream_param(text: str) -> tuple[str, object]:
    """Parse `NAME=VALUE` for one StreamParams field, typed like its default (tuples comma-separated)."""
    name, sep, value = text.partition("=")
    defaults = {f.name: f.default for f in dataclasses.fields(StreamParams)}
    if not sep or name not in defaults:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE with NAME one of {', '.join(defaults)}; got {text!r}")
    default = defaults[name]
    try:
        if isinstance(default, tuple):
            return name, tuple(float(v) for v in value.split(","))
        return name, type(default)(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad value for {name}: {value!r}") from None


def payment_count(text: str) -> int:
    """A positive whole number of payments (the `none`-gate threshold)."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number of payments; got {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive number of payments; got {text!r}")
    return n


def utc_date(text: str) -> pd.Timestamp:
    """A calendar date (YYYY-MM-DD), taken as midnight UTC."""
    try:
        return pd.Timestamp(pd.Timestamp(text).date(), tz="UTC")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date YYYY-MM-DD; got {text!r}") from None


def pseudo_source(text: str) -> tuple[str, pd.Timestamp]:
    """`SPLIT` or `SPLIT:YYYY-MM-DD`: a split's Clients Pseudo-Labelled at a Shifted Cutoff
    (default: the latest fully observed one)."""
    split, sep, cutoff = text.partition(":")
    if split not in data.SPLITS:
        raise argparse.ArgumentTypeError(
            f"expected SPLIT[:YYYY-MM-DD] with SPLIT one of {', '.join(data.SPLITS)}; got {text!r}"
        )
    return split, utc_date(cutoff) if sep else SHIFTED_CUTOFF


def positive_weight(text: str) -> float:
    try:
        weight = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive weight; got {text!r}") from None
    if not weight > 0 or weight == float("inf"):
        raise argparse.ArgumentTypeError(f"expected a positive weight; got {text!r}")
    return weight


def min_payments(text: str) -> int:
    n = payment_count(text)
    if n < 2:
        raise argparse.ArgumentTypeError(f"a Recurring Stream repeats: expected at least 2 payments; got {text!r}")
    return n


def payment_counts(text: str) -> tuple[int, ...]:
    """Comma-separated minimum-payments settings, e.g. 2,3,4."""
    try:
        values = tuple(sorted({int(v) for v in text.split(",")}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers of payments; got {text!r}") from None
    if any(v < 2 for v in values):
        raise argparse.ArgumentTypeError(f"every minimum must be at least 2 payments; got {text!r}")
    return values


def _settings_text(values: tuple[int, ...]) -> str:
    """2-10 for a run of consecutive settings, else 2,5."""
    if len(values) > 1 and values == tuple(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(v) for v in values)


def _pseudo_label_tables(args, paths: Paths, params: StreamParams, settings: tuple[int, ...]):
    """{min_payments: (labels, cached)} for the split, made without reading any label file."""
    try:
        with data.pseudo_labelling():
            return cached_pseudo_labels(
                paths.raw, args.split, paths.pseudo_labels / "cache", settings, cutoff=args.cutoff, params=params
            )
    except data.DataError:
        raise
    except ValueError as e:  # a Shifted Cutoff whose Horizon is not fully observed
        raise data.DataError(str(e)) from None


def cmd_pseudo_labels(args, paths: Paths) -> int:
    params = dataclasses.replace(StreamParams(), **dict(args.param))
    if args.fidelity and args.split != "train":
        raise data.DataError("the fidelity check compares with real labels, so it runs on --split train only")
    candidates = args.candidates or FIDELITY_CANDIDATES
    settings = candidates if args.fidelity else (args.min_payments or PSEUDO_MIN_PAYMENTS,)
    tables = _pseudo_label_tables(args, paths, params, settings)
    fidelity = _fidelity(paths, args.cutoff, tables) if args.fidelity else None
    min_payments = fidelity.chosen.min_payments if fidelity else settings[0]
    labels, cached = tables[min_payments]

    out = Path(args.out) if args.out else paths.pseudo_label_table(args.split, args.cutoff, min_payments)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"client_id": labels.index, "cutoff_date": f"{args.cutoff:%Y-%m-%d}", LABEL_COLUMN: labels.to_numpy()}
    ).to_csv(out, index=False)
    print(
        f"pseudo-labels: {args.split} at Shifted Cutoff {args.cutoff:%Y-%m-%d}, min_payments {min_payments} "
        f"({len(labels)} Clients) [{'cached' if cached else 'labelled'}] -> {out}"
    )
    counts = labels.value_counts()
    for label in LABELS:
        n = int(counts.get(label, 0))
        print(f"  {label:<10} {n:>6} {n / max(len(labels), 1):.4f}")
    if fidelity is not None:
        _report_fidelity(args, paths, candidates, fidelity)
    return 0


def _rule_predictions(paths: Paths, cutoff: pd.Timestamp, clients: pd.Index) -> pd.Series:
    """The milestone-2 rule at `cutoff`, from the (cached) train stream table detected at that Cutoff,
    which is built from the transactions before it only."""
    rule = milestone2_rule(cutoff)
    streams, _ = cached_streams(paths.raw, "train", paths.streams, cutoff=cutoff, params=rule.params)
    return predict_labels(rule.proba_from_streams(streams, clients))


def _fidelity(paths: Paths, cutoff: pd.Timestamp, tables):
    """The fidelity check on train: the only label file it reads is train's."""
    real = data.load_labels(paths.raw, "train").set_index("client_id")[LABEL_COLUMN]
    clients = pd.Index(real.index, name="client_id")
    return fidelity_check(
        real,
        _rule_predictions(paths, CUTOFF, clients),
        {m: labels for m, (labels, _) in tables.items()},
        _rule_predictions(paths, cutoff, clients),
    )


def _report_fidelity(args, paths: Paths, candidates: tuple[int, ...], fidelity) -> None:
    rule = milestone2_rule()
    rule_name = rule.name + rule.variant
    chosen = fidelity.chosen
    verdict = "PASS" if fidelity.passed else "FAIL"
    print(
        f"fidelity check: train at Shifted Cutoff {args.cutoff:%Y-%m-%d} against the real Cutoff, "
        f"rule {rule_name}"
    )
    print("  min_payments  none share  rule macro-F1  none gap  rule gap")
    for s in fidelity.settings:
        print(
            f"  {s.min_payments:<12}  {s.none_share:.4f}      {s.rule_macro_f1:.4f}         "
            f"{s.none_gap:+.4f}   {s.rule_gap:+.4f}  {'pass' if s.passed else 'fail'}"
        )
    none_text = (
        f"none share {chosen.none_share:.4f} vs real {fidelity.real_none_share:.4f}: "
        f"gap {chosen.none_gap:+.4f}, tolerance {NONE_SHARE_TOLERANCE:.2f}"
    )
    rule_text = (
        f"rule macro-F1 {chosen.rule_macro_f1:.4f} vs real {fidelity.real_rule_macro_f1:.4f}: "
        f"gap {chosen.rule_gap:+.4f}, tolerance {RULE_F1_TOLERANCE:.2f}"
    )
    print(f"  chosen min_payments {chosen.min_payments} (smallest worse gap relative to its tolerance)")
    print(f"  {none_text}")
    print(f"  {rule_text}")
    print(f"  {verdict}")

    overrides = ", ".join(f"{k}={v}" for k, v in args.param) or "defaults"
    change = (
        f"Pseudo-Label fidelity check: Shifted Cutoff {args.cutoff:%Y-%m-%d}, Horizon {HORIZON.days} days, "
        f"min_payments {chosen.min_payments} (chosen from {_settings_text(candidates)}), "
        f"labeller stream params {overrides}"
    )
    conclusion = f"{none_text}; {rule_text}; {verdict}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    row = log_row(
        chosen.rule_scores, change=change, model=rule_name, split="train", conclusion=conclusion,
        row_type="fidelity", run_id=run_id,
    )
    row.update({"delta": f"{chosen.rule_gap:.4f}", "verdict": verdict.lower()})
    append_log(paths.log, row)


def cmd_streams(args, paths: Paths) -> int:
    params = dataclasses.replace(StreamParams(), **dict(args.param))
    table, cached = cached_streams(paths.raw, args.split, paths.streams, params=params)
    n_clients = data.load_transactions(paths.raw, args.split)["client_id"].nunique()
    source = "cached" if cached else "detected"
    print(
        f"streams: {args.split} ({n_clients} Clients, {len(table)} Recurring Streams, "
        f"{int(table['active'].sum())} Active) [{source}]"
    )
    for family in MERCHANT_FAMILIES:
        fam = table[table["family"] == family]
        print(
            f"  {family:<10} streams {len(fam)}  active {int(fam['active'].sum())}  "
            f"clients {fam['client_id'].nunique()}"
        )
    return 0


def _training_data(paths: Paths, with_selection: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Train labels and transactions, plus the valid selection set's when refitting."""
    labels = data.load_labels(paths.raw, "train").set_index("client_id")[LABEL_COLUMN]
    transactions = data.load_transactions(paths.raw, "train")
    if with_selection:
        selection = data.load_labels(paths.raw, "selection").set_index("client_id")[LABEL_COLUMN]
        valid_tx = data.load_transactions(paths.raw, "valid")
        labels = pd.concat([labels, selection])
        transactions = pd.concat(
            [transactions, valid_tx[valid_tx["client_id"].isin(set(selection.index))]], ignore_index=True
        )
    return transactions, labels


def _fitted_on(with_selection: bool) -> list[str]:
    return ["train", "selection"] if with_selection else ["train"]


def _pseudo_training(args, paths: Paths) -> tuple[pd.DataFrame | None, dict | None]:
    """The Pseudo-Labelled training rows of `--pseudo` sources, and what they were: each source's split,
    Shifted Cutoff and Pseudo-Labelled Clients, the weight, the labeller's minimum payments and the
    latest fidelity check. Made from transactions only: no label file is read, so any split may be a
    source. Each Client's Candidate Streams come from its transactions before the Shifted Cutoff."""
    if not getattr(args, "pseudo", None):
        return None, None
    weight = PSEUDO_WEIGHT if args.pseudo_weight is None else args.pseudo_weight
    minimum = args.pseudo_min_payments or PSEUDO_MIN_PAYMENTS
    parts, sources = [], []
    for split, cutoff in args.pseudo:
        args_for_split = argparse.Namespace(split=split, cutoff=cutoff)
        [(labels, _)] = _pseudo_label_tables(args_for_split, paths, StreamParams(), (minimum,)).values()
        with data.pseudo_labelling():
            streams, _ = cached_streams(paths.raw, split, paths.streams, cutoff=cutoff)
        rows = pseudo_examples(streams, labels, cutoff)
        # the same Client may be a source at two Shifted Cutoffs, or also real-labelled: keep them apart
        rows["client_id"] = f"{split}@{cutoff:%Y-%m-%d}:" + rows["client_id"].astype(str)
        parts.append(rows)
        sources.append({"split": split, "cutoff": f"{cutoff:%Y-%m-%d}", "clients": int(len(labels))})
    info = {"sources": sources, "weight": weight, "min_payments": minimum, "fidelity": latest_fidelity(paths.log)}
    return pd.concat(parts, ignore_index=True), info


def _sources_text(pseudo: dict) -> str:
    return ";".join(f"{s['split']}@{s['cutoff']}" for s in pseudo["sources"])


def _pseudo_text(pseudo: dict) -> str:
    return (
        f"Pseudo-Labels {', '.join(_sources_text(pseudo).split(';'))}, weight {pseudo['weight']:g}, "
        f"min_payments {pseudo['min_payments']}"
    )


def _fidelity_note(pseudo: dict) -> str:
    fidelity = pseudo["fidelity"]
    if fidelity is None:
        return "no fidelity check logged: these Pseudo-Labels are not validated (run `pseudo-labels --fidelity`)"
    if fidelity["verdict"] == "pass":
        return f"the latest Pseudo-Label fidelity check {fidelity['run_id']} passed"
    return (
        f"the latest Pseudo-Label fidelity check {fidelity['run_id']} failed: training ran anyway, "
        "but this is not a validated setup"
    )


def _fidelity_column(pseudo: dict | None) -> str:
    if pseudo is None:
        return ""
    fidelity = pseudo["fidelity"]
    return "unchecked" if fidelity is None else f"{fidelity['verdict']} {fidelity['run_id']}"


def _new_model(args, pseudo: pd.DataFrame | None = None, info: dict | None = None):
    """A fresh model; the rule baseline takes its ordering rule, `none`-gate and stream parameters from
    `train`; the ranker pools any Pseudo-Labelled training rows into its fit."""
    if args.model == "ranker" and pseudo is not None:
        return MODELS["ranker"](pseudo=pseudo, pseudo_weight=info["weight"])
    if args.model != "rules":
        return MODELS[args.model]()
    params = dataclasses.replace(StreamParams(), **dict(args.param))
    return MODELS["rules"](ordering=args.ordering or DEFAULT_ORDERING, params=params, none_gate=args.none_gate)


def cmd_train(args, paths: Paths) -> int:
    pseudo, info = _pseudo_training(args, paths)
    with data.training_run(with_selection=args.with_selection):
        transactions, labels = _training_data(paths, args.with_selection)
        model = _new_model(args, pseudo, info).fit(transactions, labels)
        if args.decision == "tuned":
            # fitted on out-of-fold probabilities of the real-labelled training Clients only: Pseudo-Labelled
            # Clients join every fold's fit but are never scored, so none reaches the decision layer
            oof = out_of_fold_proba(lambda: _new_model(args, pseudo, info), transactions, labels, folds=args.folds)
            decision, fit_scores = decision_layer.fit(oof[list(LABELS)], labels)
    pseudo_clients = sum(s["clients"] for s in info["sources"]) if info else 0
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    model.save(paths.model(args.model))
    paths.model_meta(args.model).write_text(
        json.dumps(
            {
                "fitted_on": _fitted_on(args.with_selection),
                "training_clients": len(labels) + pseudo_clients,
                "pseudo": info,
            }
        )
    )
    fitted_on = " + ".join(_fitted_on(args.with_selection))
    if info is None:
        print(f"train: {args.model} fitted on {len(labels)} Clients ({fitted_on}) -> {paths.model(args.model)}")
    else:
        print(
            f"train: {args.model} fitted on {len(labels)} real-labelled + {pseudo_clients} Pseudo-Labelled Clients "
            f"({fitted_on}; {_pseudo_text(info)}) -> {paths.model(args.model)}"
        )
        print(f"  {_fidelity_note(info)}")
    if args.decision == "tuned":
        decision.save(paths.decision(args.model))
        print(
            f"  decision layer tuned on {args.folds}-fold out-of-fold probabilities of the real-labelled "
            f"Clients: macro-F1 "
            f"{fit_scores['argmax_macro_f1']:.4f} (argmax) -> {fit_scores['tuned_macro_f1']:.4f} (tuned)"
            f" -> {paths.decision(args.model)}"
        )
        threshold = decision.none_threshold
        print(
            "  weights " + ", ".join(f"{label} {w:.2f}" for label, w in decision.weights.items())
            + f"; none threshold {'off' if threshold is None else f'{threshold:.2f}'}"
        )
    else:
        paths.decision(args.model).unlink(missing_ok=True)  # it belonged to the replaced model
    return 0


def cmd_cv(args, paths: Paths) -> int:
    pseudo, info = _pseudo_training(args, paths)
    make_model = MODELS[args.model] if info is None else (lambda: _new_model(args, pseudo, info))
    with data.training_run(with_selection=args.with_selection):
        transactions, labels = _training_data(paths, args.with_selection)
        # folds are drawn over the real-labelled Clients only; Pseudo-Labelled ones join every fold's fit
        oof = out_of_fold_proba(make_model, transactions, labels, folds=args.folds)
    out = Path(args.out) if args.out else paths.artifacts / "oof" / f"{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out, index_label="client_id")
    if info is None:
        print(f"cv: {args.model} {args.folds}-fold out-of-fold probabilities for {len(oof)} Clients -> {out}")
    else:
        print(
            f"cv: {args.model} {args.folds}-fold out-of-fold probabilities for {len(oof)} real-labelled Clients "
            f"(every fold also fitted on {sum(s['clients'] for s in info['sources'])} Pseudo-Labelled Clients: "
            f"{_pseudo_text(info)}) -> {out}"
        )
        print(f"  {_fidelity_note(info)}")
    return 0


def cmd_features(args, paths: Paths) -> int:
    model = _load_model(paths, "lgbm")
    transactions = data.load_transactions(paths.raw, args.split)
    clients = pd.Index(sorted(transactions["client_id"].astype(str).unique()), name="client_id")
    rows = model.features(transactions, clients)
    # the Clients whose labels the model was fitted on get the out-of-fold rows it trained on:
    # encoded with the fitted description rates, their rows would carry their own labels
    meta = paths.model_meta("lgbm")
    with_selection = "selection" in (json.loads(meta.read_text())["fitted_on"] if meta.exists() else ["train"])
    if args.split == "train" or (args.split == "valid" and with_selection):
        with data.training_run(with_selection=with_selection):
            train_tx, train_labels = _training_data(paths, with_selection)
        trained = clients.intersection(train_labels.index)
        if len(trained):
            out_of_fold = model.training_features(train_tx, train_labels)
            rows = pd.concat([rows.drop(trained), out_of_fold.loc[trained]]).reindex(clients)
    out = Path(args.out) if args.out else paths.artifacts / "features" / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index_label="client_id")
    print(f"features: {args.split} ({len(rows)} Clients, {rows.shape[1]} features) -> {out}")
    return 0


def _best_predictions(paths: Paths, best: dict, clients: pd.Index) -> pd.Series:
    """The previous best run's per-Client predictions, which the paired bootstrap needs for
    exactly these Clients. Never fall back to a weaker run: that would misstate the verdict."""
    run_id = best.get("run_id") or ""
    path = paths.run_predictions(run_id) if run_id else None
    if path is None or not path.exists():
        raise data.DataError(
            f"the previous best run on {best['split']} ({run_id or 'no run id'}, "
            f"macro-F1 {best['macro_f1']}) has no saved predictions"
            + (f" at {path}" if path else "")
            + "; restore them (they are committed under experiments/runs/) before evaluating"
        )
    old = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]
    if set(old.index) != set(clients):
        raise data.DataError(
            f"the previous best run {run_id} on {best['split']} predicted for different Clients than this "
            "evaluation; a paired comparison is impossible"
        )
    return old.reindex(clients)


def _scored_clients(paths: Paths, split: str) -> pd.Index:
    """The Clients an evaluation scores, known without reading any valid labels."""
    if split == "train":
        ids = data.load_labels(paths.raw, "train")["client_id"]
    else:
        ids = data.valid_split(paths.raw).query("set == @split")["client_id"]
    return pd.Index(ids, name="client_id")


def cmd_evaluate(args, paths: Paths) -> int:
    model = _load_model(paths, args.model)
    split = "holdout" if args.checkpoint else args.split
    row_type = "checkpoint" if args.checkpoint else "evaluate"
    meta_path = paths.model_meta(args.model)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fitted_on = meta.get("fitted_on", ["train"])
    pseudo = meta.get("pseudo")
    if split == "selection" and split in fitted_on:
        raise data.DataError(
            f"{args.model} was fitted on {' + '.join(fitted_on)}; scoring it on the selection set is not honest"
        )
    clients = _scored_clients(paths, split)
    decision = _load_decision(paths, args.model, args.decision)
    if decision is not None:
        overlap = set(decision.fitted_on_clients) & set(clients)
        if overlap:
            raise data.DataError(
                f"the decision layer was fitted on {len(overlap)} of the {len(clients)} {split} Clients it would "
                "be scored on; score it on Clients outside its fitting data"
            )
    transactions = data.load_transactions(paths.raw, "train" if split == "train" else "valid")
    with data.predicting():
        proba = model.predict_proba(transactions, clients)
    # only now, with the predictions made, read the labels they are scored against
    with data.checkpoint() if args.checkpoint else data.scoring():
        labels = data.load_labels(paths.raw, split).set_index("client_id")[LABEL_COLUMN]
    predicted = _decide(proba, decision).reindex(labels.index)
    scores = score(labels, predicted)
    diagnostics = model.diagnostics(transactions, labels, predicted) if hasattr(model, "diagnostics") else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    comparison = None
    best = previous_best(paths.log, row_type=row_type, split=split)
    if best is not None:
        old = _best_predictions(paths, best, labels.index)
        comparison = {**paired_bootstrap(labels, predicted, old), "run_id": best["run_id"]}
    if args.proba:
        Path(args.proba).parent.mkdir(parents=True, exist_ok=True)
        proba[list(LABELS)].to_csv(args.proba, index_label="client_id")
    out = paths.run_predictions(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    predicted.rename("predicted").to_csv(out, index_label="client_id")

    name = args.model + getattr(model, "variant", "") + ("+pseudo" if pseudo else "") + _decision_name(args.decision)
    row = log_row(
        scores, change=args.change, model=name, split=split, conclusion=args.conclusion,
        row_type=row_type, run_id=run_id, comparison=comparison, diagnostics=diagnostics,
    )
    row["training_clients"] = meta.get("training_clients", "")
    if pseudo:
        row.update(
            {
                "pseudo_sources": _sources_text(pseudo),
                "pseudo_weight": f"{pseudo['weight']:g}",
                "pseudo_min_payments": pseudo["min_payments"],
                "fidelity": _fidelity_column(pseudo),
            }
        )
    append_log(paths.log, row)
    print(f"evaluate{' (checkpoint)' if args.checkpoint else ''}: {args.model} on {split} ({len(labels)} Clients)")
    if pseudo:
        print(f"  trained with {_pseudo_text(pseudo)}; {_fidelity_note(pseudo)}")
    if comparison is None:
        print("  no comparable previous best: verdict first")
    else:
        print(
            f"  vs best {comparison['run_id']}: delta {comparison['delta']:+.4f} "
            f"(95% {comparison['low']:+.4f} .. {comparison['high']:+.4f}) -> {row['verdict']}"
        )
    print(f"  macro-F1 {scores['macro_f1']:.4f}")
    for label in LABELS:
        print(f"  {label:<10} F1 {scores[f'f1_{label}']:.4f}")
    if diagnostics is not None:
        print(
            f"  coverage {diagnostics['coverage']:.4f} (true family among surviving streams), "
            f"selection accuracy {diagnostics['selection_accuracy']:.4f} (given coverage)"
        )
    return 0


def _logged_run(paths: Paths, run_id: str) -> dict:
    rows = []
    if paths.log.exists():
        with open(paths.log, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_id") == run_id]
    if not rows:
        raise data.DataError(f"no run {run_id} in the experiment log {paths.log}")
    return rows[-1]


def _saved_predictions(paths: Paths, run_id: str) -> pd.Series:
    path = paths.run_predictions(run_id)
    if not path.exists():
        raise data.DataError(
            f"run {run_id} has no saved predictions at {path}; restore them (they are committed under "
            "experiments/runs/) before comparing"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("client_id")["predicted"]


def cmd_compare(args, paths: Paths) -> int:
    rows = [_logged_run(paths, run_id) for run_id in args.run]
    sealed = [r["run_id"] for r in rows if r.get("split") == "holdout" or r.get("row_type") == "checkpoint"]
    if sealed:
        raise data.DataError(
            f"{', '.join(sealed)} scored the sealed holdout; candidates are chosen on the selection set only, "
            "and the sealed holdout is read only in a human-started checkpoint"
        )
    splits = sorted({r["split"] for r in rows})
    if len(splits) > 1:
        raise data.DataError(
            f"candidates must be scored on the same split for a paired comparison; got {', '.join(splits)}"
        )
    split = splits[0]
    if split not in ("selection", "train"):
        raise data.DataError(f"cannot compare runs on {split!r}: expected selection (or train) runs")
    candidates = [
        Candidate(r["run_id"], r["model"], r["change"], _saved_predictions(paths, r["run_id"])) for r in rows
    ]
    clients = set(candidates[0].predicted.index)
    for c in candidates[1:]:
        if set(c.predicted.index) != clients:
            raise data.DataError(
                f"{c.run_id} predicted for different Clients than {candidates[0].run_id}; "
                "a paired comparison is impossible"
            )
    # every prediction is already made and saved: only now read the labels they are scored against
    with data.scoring():
        labels = data.load_labels(paths.raw, split).set_index("client_id")[LABEL_COLUMN]
    if set(labels.index) != clients:
        raise data.DataError(f"the saved predictions are not for the {split} Clients; they cannot be scored")
    comparison = compare_candidates(labels, candidates, split)
    print("\n".join(summary_lines(comparison)))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(comparison, args.title))
        print(f"  -> {out}")
    return 0


def cmd_submit(args, paths: Paths) -> int:
    expected = data.load_sample_submission(paths.raw)["client_id"]
    if args.check:
        validate(read_submission(Path(args.check)), expected)
        print(f"submit: {args.check} is valid")
        return 0
    model = _load_model(paths, args.model)
    decision = _load_decision(paths, args.model, args.decision)
    transactions = data.load_transactions(paths.raw, "test")
    predicted = _decide(model.predict_proba(transactions, expected), decision)
    out = paths.submissions / f"{args.name}.csv"
    write_submission(predicted, expected, out)
    print(f"submit: wrote {out} ({len(expected)} Clients)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".", help="project root (default: current directory)")

    parser = argparse.ArgumentParser(prog="rf", description="Next Recurring Family pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", parents=[common], help="unpack the challenge zip into data/raw")
    p.add_argument("--zip", help=f"challenge zip (default: <root>/{DEFAULT_ZIP})")
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser("split", parents=[common], help="write the valid selection / sealed-holdout split")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("streams", parents=[common], help="detect (or load cached) Recurring Streams and summarise per family")
    p.add_argument("--split", choices=data.SPLITS, default="train")
    p.add_argument(
        "--param", type=stream_param, action="append", default=[], metavar="NAME=VALUE",
        help="override one stream detection parameter (repeatable), e.g. amount_tolerance=0.08",
    )
    p.set_defaults(func=cmd_streams)

    p = sub.add_parser(
        "pseudo-labels", parents=[common],
        help="write a split's Pseudo-Labels at a Shifted Cutoff (reads no label file); --fidelity checks them on train",
    )
    p.add_argument("--split", choices=data.SPLITS, default="train")
    p.add_argument(
        "--cutoff", type=utc_date, default=SHIFTED_CUTOFF, metavar="YYYY-MM-DD",
        help=f"the Shifted Cutoff (default {SHIFTED_CUTOFF:%Y-%m-%d}, the latest whose Horizon is fully observed)",
    )
    p.add_argument(
        "--min-payments", type=payment_count, metavar="N",
        help="payments a Recurring Stream needs for its Horizon payment to count "
        f"(default {PSEUDO_MIN_PAYMENTS}, the fidelity check's choice)",
    )
    p.add_argument(
        "--param", type=stream_param, action="append", default=[], metavar="NAME=VALUE",
        help="override one stream detection parameter of the labeller (repeatable)",
    )
    p.add_argument("--out", metavar="CSV", help="default: <root>/artifacts/pseudo_labels/<split>-<cutoff>-min<N>.csv")
    p.add_argument(
        "--fidelity", action="store_true",
        help="train only: compare the Pseudo-Label task with the real one, choose min_payments, log the result",
    )
    p.add_argument(
        "--candidates", type=payment_counts, metavar="N,N,...",
        help=f"--fidelity: the min_payments settings to choose among (default {_settings_text(FIDELITY_CANDIDATES)})",
    )
    p.set_defaults(func=cmd_pseudo_labels)

    p = sub.add_parser("train", parents=[common], help="fit a model on the train Clients")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--with-selection", action="store_true", help="also fit on the valid selection set (submission refit)")
    p.add_argument(
        "--ordering", choices=sorted(ORDERINGS),
        help=f"rules only: which surviving stream to predict (default: {DEFAULT_ORDERING}, the projected next payment)",
    )
    p.add_argument(
        "--param", type=stream_param, action="append", default=[], metavar="NAME=VALUE",
        help="rules only: override one stream detection parameter (repeatable)",
    )
    p.add_argument(
        "--none-gate", type=payment_count, nargs="?", const=DEFAULT_NONE_GATE, metavar="N",
        help=f"rules only: predict none when the Client's longest surviving stream has at most N payments "
        f"(off unless given; N defaults to {DEFAULT_NONE_GATE})",
    )
    p.add_argument(
        "--decision", choices=("argmax", "tuned"), default="argmax",
        help="tuned: also fit the E3 decision layer on out-of-fold probabilities",
    )
    p.add_argument("--folds", type=int, default=CV_FOLDS, help="folds for the tuned decision layer's out-of-fold fit")
    p.add_argument(
        "--pseudo", type=pseudo_source, action="append", default=[], metavar="SPLIT[:YYYY-MM-DD]",
        help="ranker only: also fit on this split's Clients Pseudo-Labelled at a Shifted Cutoff "
        f"(default {SHIFTED_CUTOFF:%Y-%m-%d}); repeatable. They are never scored out of fold",
    )
    p.add_argument(
        "--pseudo-weight", type=positive_weight, metavar="W",
        help=f"--pseudo: sample weight of the Pseudo-Labelled Clients (default {PSEUDO_WEIGHT}; real ones weigh 1)",
    )
    p.add_argument(
        "--pseudo-min-payments", type=min_payments, metavar="N",
        help=f"--pseudo: payments a stream needs for its Horizon payment to count (default {PSEUDO_MIN_PAYMENTS})",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("cv", parents=[common], help="stratified k-fold out-of-fold probabilities")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--with-selection", action="store_true", help="cross-validate over train plus the selection set")
    p.add_argument("--folds", type=int, default=CV_FOLDS)
    p.add_argument("--out", metavar="CSV", help="default: <root>/artifacts/oof/<model>.csv")
    p.add_argument(
        "--pseudo", type=pseudo_source, action="append", default=[], metavar="SPLIT[:YYYY-MM-DD]",
        help="ranker only: also fit on this split's Clients Pseudo-Labelled at a Shifted Cutoff "
        f"(default {SHIFTED_CUTOFF:%Y-%m-%d}); repeatable. They are never scored out of fold",
    )
    p.add_argument(
        "--pseudo-weight", type=positive_weight, metavar="W",
        help=f"--pseudo: sample weight of the Pseudo-Labelled Clients (default {PSEUDO_WEIGHT}; real ones weigh 1)",
    )
    p.add_argument(
        "--pseudo-min-payments", type=min_payments, metavar="N",
        help=f"--pseudo: payments a stream needs for its Horizon payment to count (default {PSEUDO_MIN_PAYMENTS})",
    )
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser(
        "features", parents=[common], help="write the E2 feature rows the trained lgbm model scores for a split"
    )
    p.add_argument("--split", choices=data.SPLITS, default="train")
    p.add_argument("--out", metavar="CSV", help="default: <root>/artifacts/features/<split>.csv")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("evaluate", parents=[common], help="score a trained model and log the run")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--split", choices=("selection", "train"), default="selection")
    p.add_argument(
        "--checkpoint", action="store_true",
        help="human-only milestone checkpoint: score the sealed holdout (ignores --split)",
    )
    p.add_argument("--change", default="", help="what changed in this run")
    p.add_argument("--conclusion", default="", help="what the run tells us")
    p.add_argument("--proba", metavar="CSV", help="also write per-Client label probabilities to this CSV")
    p.add_argument(
        "--decision", default="argmax", metavar="argmax|tuned|JSON",
        help="argmax (default); tuned: the decision layer fitted by `train --decision tuned`; "
        "or a JSON file with hand-set weights and none_threshold",
    )
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "compare", parents=[common],
        help="pick a milestone candidate among logged runs by paired bootstrap (ties go to the simpler one)",
    )
    p.add_argument(
        "--run", action="append", required=True, metavar="RUN_ID",
        help="a logged run to compare (repeatable, at least two), simplest candidate first",
    )
    p.add_argument("--out", metavar="MD", help="also write the comparison as Markdown to this file")
    p.add_argument("--title", default="Candidate comparison", help="the Markdown heading")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("submit", parents=[common], help="write or check a submission CSV")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=sorted(MODELS))
    group.add_argument("--check", metavar="CSV", help="validate an existing submission file")
    p.add_argument("--name", default="submission", help="file name under submissions/ (without .csv)")
    p.add_argument(
        "--decision", default="argmax", metavar="argmax|tuned|JSON",
        help="argmax (default); tuned: the decision layer fitted by `train --decision tuned`; "
        "or a JSON file with hand-set weights and none_threshold",
    )
    p.set_defaults(func=cmd_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.model != "rules" and (args.ordering or args.param or args.none_gate):
        parser.error("--ordering, --param and --none-gate apply to --model rules only")
    if args.command in ("train", "cv"):
        if args.pseudo and args.model != "ranker":
            parser.error("--pseudo applies to --model ranker only")
        if not args.pseudo and (args.pseudo_weight is not None or args.pseudo_min_payments is not None):
            parser.error("--pseudo-weight and --pseudo-min-payments apply to --pseudo only")
        if len(set(args.pseudo)) < len(args.pseudo):
            parser.error("give each Pseudo-Label source (split and Shifted Cutoff) once")
    if args.command == "compare" and (len(args.run) < 2 or len(set(args.run)) < len(args.run)):
        parser.error("give at least two distinct --run candidates")
    if args.command == "pseudo-labels":
        if args.fidelity and args.min_payments is not None:
            parser.error("--fidelity chooses min_payments itself; give the settings to choose among with --candidates")
        if not args.fidelity and args.candidates is not None:
            parser.error("--candidates applies to --fidelity only")
    paths = Paths(Path(args.root))
    try:
        return args.func(args, paths)
    except InvalidSubmission as e:
        print(f"invalid submission: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, data.DataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
