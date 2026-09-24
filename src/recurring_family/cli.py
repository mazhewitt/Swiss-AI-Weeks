"""The one CLI: fetch-data, split, streams, train, cv, evaluate, submit."""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data
from . import decision as decision_layer
from .config import LABEL_COLUMN, LABELS, MERCHANT_FAMILIES
from .cross_validation import CV_FOLDS, out_of_fold_proba
from .evaluation import append_log, log_row, paired_bootstrap, previous_best, score
from .fetch import fetch_data
from .models import MODELS, predict_labels
from .streams import cached_streams
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


def cmd_streams(args, paths: Paths) -> int:
    table, cached = cached_streams(paths.raw, args.split, paths.streams)
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


def cmd_train(args, paths: Paths) -> int:
    with data.training_run(with_selection=args.with_selection):
        transactions, labels = _training_data(paths, args.with_selection)
        model = MODELS[args.model]().fit(transactions, labels)
        if args.decision == "tuned":
            # fitted on out-of-fold probabilities of the training Clients only
            oof = out_of_fold_proba(MODELS[args.model], transactions, labels, folds=args.folds)
            decision, fit_scores = decision_layer.fit(oof[list(LABELS)], labels)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    model.save(paths.model(args.model))
    paths.model_meta(args.model).write_text(json.dumps({"fitted_on": _fitted_on(args.with_selection)}))
    print(
        f"train: {args.model} fitted on {len(labels)} Clients ({' + '.join(_fitted_on(args.with_selection))})"
        f" -> {paths.model(args.model)}"
    )
    if args.decision == "tuned":
        decision.save(paths.decision(args.model))
        print(
            f"  decision layer tuned on {args.folds}-fold out-of-fold probabilities: macro-F1 "
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
    with data.training_run(with_selection=args.with_selection):
        transactions, labels = _training_data(paths, args.with_selection)
        oof = out_of_fold_proba(MODELS[args.model], transactions, labels, folds=args.folds)
    out = Path(args.out) if args.out else paths.artifacts / "oof" / f"{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out, index_label="client_id")
    print(f"cv: {args.model} {args.folds}-fold out-of-fold probabilities for {len(oof)} Clients -> {out}")
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
    meta = paths.model_meta(args.model)
    fitted_on = json.loads(meta.read_text())["fitted_on"] if meta.exists() else ["train"]
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

    row = log_row(
        scores, change=args.change, model=args.model + _decision_name(args.decision), split=split, conclusion=args.conclusion,
        row_type=row_type, run_id=run_id, comparison=comparison,
    )
    append_log(paths.log, row)
    print(f"evaluate{' (checkpoint)' if args.checkpoint else ''}: {args.model} on {split} ({len(labels)} Clients)")
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
    p.set_defaults(func=cmd_streams)

    p = sub.add_parser("train", parents=[common], help="fit a model on the train Clients")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--with-selection", action="store_true", help="also fit on the valid selection set (submission refit)")
    p.add_argument(
        "--decision", choices=("argmax", "tuned"), default="argmax",
        help="tuned: also fit the E3 decision layer on out-of-fold probabilities",
    )
    p.add_argument("--folds", type=int, default=CV_FOLDS, help="folds for the tuned decision layer's out-of-fold fit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("cv", parents=[common], help="stratified k-fold out-of-fold probabilities")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--with-selection", action="store_true", help="cross-validate over train plus the selection set")
    p.add_argument("--folds", type=int, default=CV_FOLDS)
    p.add_argument("--out", metavar="CSV", help="default: <root>/artifacts/oof/<model>.csv")
    p.set_defaults(func=cmd_cv)

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
    args = build_parser().parse_args(argv)
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
