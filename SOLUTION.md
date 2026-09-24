# Solution: Next Recurring Family

Our entry for the 2026 challenge (`hackathons/2026/challenge.md`). Glossary: `CONTEXT.md`. Decisions: `docs/adr/`. The organisers' `README.md` and `hackathons/` are never edited.

## Setup

```sh
uv sync
uv run rf fetch-data          # unpacks hackathons/2026/data/dataset.zip into data/raw/ (idempotent)
```

## Pipeline

```sh
uv run rf train --model prior
uv run rf evaluate --model prior --change "what changed" --conclusion "what we learned"
uv run rf submit --model prior --name milestone1-all-none
uv run rf submit --check submissions/milestone1-all-none.csv
```

- `streams --split valid` detects (or loads cached) Recurring Streams and prints a per-family summary. `--param NAME=VALUE` (repeatable) overrides one stream detection parameter, e.g. `--param amount_tolerance=0.08`. Each parameter set is cached separately under `artifacts/streams/`, and a change to the detector code, family table or raw data rebuilds.
- `train --model rules` is E1, the rule baseline: among each Client's Active Streams whose projected next payment falls within the Horizon, it predicts the family of the one due soonest, else `none`. `--ordering most_recent|longest` picks by another rule, and `--param NAME=VALUE` overrides stream detection parameters; both are saved with the model.
- `train --model lgbm` fits E2: LightGBM multiclass (balanced class weights, default hyperparameters) on one feature row per Client, built only from its Recurring Streams (ADR 0001): the top three Active Stream slots, a block per Merchant Family and the `none` signals. `features --split test` writes the rows the trained model scores for a split (same columns for every split); for the Clients it was fitted on (train, plus the selection set after `--with-selection`) those are the out-of-fold rows it trained on, so no row carries its own Client's label. `cv --model lgbm` saves its out-of-fold probabilities.
- `evaluate` reports macro-F1 over the eight allowed labels and per-family F1, and appends one row to `experiments/log.csv`. For E1 the row also records coverage (how often the true family is among the surviving streams, over Clients whose label is a family) and selection accuracy given coverage.
- E3 decision layer, for any model: `train --model <m> --decision tuned` also fits one weight per label plus a `none` threshold by grid search on the training Clients' out-of-fold probabilities (never valid labels). `evaluate` and `submit` then take `--decision tuned` (default `argmax`), or `--decision <file>.json` with hand-set `weights` and `none_threshold` (`none` when P(none) >= 1 - threshold; 0 never predicts `none`, 1 always does). A tuned decision is refused on Clients it was fitted on.
- `submit` writes `submissions/<name>.csv` for the test Clients and refuses files with missing, extra or duplicate Client IDs or labels outside the allowed set.

## Layout

| Path | Contents | In git |
|---|---|---|
| `data/raw/`, `data/interim/` | challenge data, caches | no |
| `artifacts/` | fitted models | no |
| `submissions/` | submission CSVs | yes |
| `experiments/log.csv` | experiment log | yes |

## Tests

```sh
uv run pytest                 # fast suite, fixture data only
uv run pytest -m slow         # real-data checks (needs fetch-data)
```
