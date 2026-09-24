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
