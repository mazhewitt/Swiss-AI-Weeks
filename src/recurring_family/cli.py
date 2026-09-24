"""The one CLI: fetch-data, train, evaluate, submit."""

import argparse
import sys
from pathlib import Path

from . import data
from .config import LABEL_COLUMN, LABELS
from .evaluation import append_log, log_row, score
from .fetch import fetch_data
from .models import MODELS, predict_labels
from .submission import InvalidSubmission, read_submission, validate, write_submission

DEFAULT_ZIP = Path("hackathons") / "2026" / "data" / "dataset.zip"


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "data" / "raw"
        self.artifacts = root / "artifacts"
        self.submissions = root / "submissions"
        self.log = root / "experiments" / "log.csv"

    def model(self, name: str) -> Path:
        return self.artifacts / f"{name}.json"


def _load_model(paths: Paths, name: str):
    path = paths.model(name)
    if not path.exists():
        raise FileNotFoundError(f"no trained {name!r} model at {path}; run `train --model {name}` first")
    return MODELS[name].load(path)


def cmd_fetch_data(args, paths: Paths) -> int:
    zip_path = Path(args.zip) if args.zip else paths.root / DEFAULT_ZIP
    written = fetch_data(zip_path, paths.raw)
    print(f"fetch-data: {len(written)} file(s) written to {paths.raw}" + (f": {', '.join(written)}" if written else ""))
    for split in data.SPLITS:
        tx = data.load_transactions(paths.raw, split)
        first, last = (t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in (tx["timestamp"].min(), tx["timestamp"].max()))
        print(f"  {split}: {tx['client_id'].nunique()} Clients, {len(tx)} transactions, {first} .. {last}")
    return 0


def cmd_train(args, paths: Paths) -> int:
    labels = data.load_labels(paths.raw, "train").set_index("client_id")[LABEL_COLUMN]
    transactions = data.load_transactions(paths.raw, "train")
    model = MODELS[args.model]().fit(transactions, labels)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    model.save(paths.model(args.model))
    print(f"train: {args.model} fitted on {len(labels)} train Clients -> {paths.model(args.model)}")
    return 0


def cmd_evaluate(args, paths: Paths) -> int:
    model = _load_model(paths, args.model)
    labels = data.load_labels(paths.raw, args.split).set_index("client_id")[LABEL_COLUMN]
    transactions = data.load_transactions(paths.raw, args.split)
    predicted = predict_labels(model.predict_proba(transactions, labels.index))
    scores = score(labels, predicted.reindex(labels.index))
    append_log(
        paths.log,
        log_row(scores, change=args.change, model=args.model, split=args.split, conclusion=args.conclusion),
    )
    print(f"evaluate: {args.model} on {args.split} ({len(labels)} Clients)")
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
    transactions = data.load_transactions(paths.raw, "test")
    predicted = predict_labels(model.predict_proba(transactions, expected))
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

    p = sub.add_parser("train", parents=[common], help="fit a model on the train Clients")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[common], help="score a trained model and log the run")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--split", choices=data.LABELLED_SPLITS, default="valid")
    p.add_argument("--change", default="", help="what changed in this run")
    p.add_argument("--conclusion", default="", help="what the run tells us")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("submit", parents=[common], help="write or check a submission CSV")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=sorted(MODELS))
    group.add_argument("--check", metavar="CSV", help="validate an existing submission file")
    p.add_argument("--name", default="submission", help="file name under submissions/ (without .csv)")
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
