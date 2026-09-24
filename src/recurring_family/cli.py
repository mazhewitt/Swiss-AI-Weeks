"""The one CLI: fetch-data, streams, train, evaluate, submit."""

import argparse
import dataclasses
import sys
from pathlib import Path

from . import data
from .config import LABEL_COLUMN, LABELS, MERCHANT_FAMILIES
from .evaluation import append_log, log_row, score
from .fetch import fetch_data
from .models import MODELS, predict_labels
from .streams import StreamParams, cached_streams
from .submission import InvalidSubmission, read_submission, validate, write_submission

DEFAULT_ZIP = Path("hackathons") / "2026" / "data" / "dataset.zip"


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "data" / "raw"
        self.artifacts = root / "artifacts"
        self.submissions = root / "submissions"
        self.log = root / "experiments" / "log.csv"

    @property
    def streams(self) -> Path:
        return self.artifacts / "streams"

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
    proba = model.predict_proba(transactions, labels.index)
    if args.proba:
        Path(args.proba).parent.mkdir(parents=True, exist_ok=True)
        proba[list(LABELS)].to_csv(args.proba, index_label="client_id")
    predicted = predict_labels(proba)
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

    p = sub.add_parser("streams", parents=[common], help="detect (or load cached) Recurring Streams and summarise per family")
    p.add_argument("--split", choices=data.SPLITS, default="train")
    p.add_argument(
        "--param", type=stream_param, action="append", default=[], metavar="NAME=VALUE",
        help="override one stream detection parameter (repeatable), e.g. amount_tolerance=0.08",
    )
    p.set_defaults(func=cmd_streams)

    p = sub.add_parser("train", parents=[common], help="fit a model on the train Clients")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[common], help="score a trained model and log the run")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--split", choices=data.LABELLED_SPLITS, default="valid")
    p.add_argument("--change", default="", help="what changed in this run")
    p.add_argument("--conclusion", default="", help="what the run tells us")
    p.add_argument("--proba", metavar="CSV", help="also write per-Client label probabilities to this CSV")
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
