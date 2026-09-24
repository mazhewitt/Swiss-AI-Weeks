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

- `evaluate` reports macro-F1 over the eight allowed labels and per-family F1, and appends one row to `experiments/log.csv`.
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
