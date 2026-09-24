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

The milestone-2 submission is the gated rule, reproduced exactly by:

```sh
uv run rf train --model rules --none-gate
uv run rf submit --model rules --name milestone2_rules_none_gate_v2
```

- `streams --split valid` detects (or loads cached) Recurring Streams and prints a per-family summary. `--param NAME=VALUE` (repeatable) overrides one stream detection parameter, e.g. `--param amount_tolerance=0.08`. Each parameter set is cached separately under `artifacts/streams/`, and a change to the detector code, family table or raw data rebuilds.
- `pseudo-labels --split <split> --cutoff <date>` writes each Client's Pseudo-Label at a Shifted Cutoff (default 2025-10-03, the latest whose 90-day Horizon is fully observed) to `artifacts/pseudo_labels/<split>-<cutoff>-min<N>.csv` (`client_id, cutoff_date, target_next_recurring_merchant`, like a label file) and prints the per-label counts and shares. It works for every split, valid and test included, because it reads transactions only: any label read while it runs is refused. `--min-payments N` is the payments a Recurring Stream needs for its Horizon payment to count (default 4, chosen by the fidelity check); `--param NAME=VALUE` overrides the labeller's stream detection. Tables are cached under `artifacts/pseudo_labels/cache/` by split, Shifted Cutoff, detector version and labeller parameters.
- `pseudo-labels --split train --fidelity` is the fidelity check: for each `min_payments` in `--candidates` (default 2-10) it compares the Pseudo-Label `none` share with the real train share, and the milestone-2 rule's macro-F1 against Pseudo-Labels (its inputs are the transactions before the Shifted Cutoff only) with its macro-F1 against the real labels. It chooses the setting with the smallest worse gap relative to the tolerances (0.05 each), prints PASS or FAIL, writes that setting's table and appends a `fidelity` row to the log. On the real train split it fails (the `fidelity` row of 2026-09-24).
- `train --model rules` is E1, the rule baseline: among each Client's Active Streams whose projected next payment falls within the Horizon, it predicts the family of the one due soonest, else `none`. `--none-gate [N]` adds the milestone-2 `none`-gate: a Client whose longest surviving stream has at most N payments (default 4) gets `none`; it is off unless given. `--ordering most_recent|longest` picks by another rule, and `--param NAME=VALUE` overrides stream detection parameters; all are saved with the model, and the log's model column names the variant (`rules+gate4`).
- `train --model lgbm` fits E2: LightGBM multiclass (balanced class weights, default hyperparameters) on one feature row per Client, built only from its Recurring Streams (ADR 0001): the top three Active Stream slots, a block per Merchant Family and the `none` signals. `features --split test` writes the rows the trained model scores for a split (same columns for every split); for the Clients it was fitted on (train, plus the selection set after `--with-selection`) those are the out-of-fold rows it trained on, so no row carries its own Client's label. `cv --model lgbm` saves its out-of-fold probabilities.
- `train --model ranker` fits the Stream Ranker: every detected Recurring Stream is a Candidate Stream, and a LightGBM binary model scores whether it carries the Client's Next Recurring Family, from stream-table features only (ADR 0001). Each family takes its best candidate's score, `none` is one minus the best score, and the row is normalised; a Client without streams is `none`. It works with `cv`, `evaluate`, `submit` and `--decision tuned` like any model.
- `train` and `cv --model ranker --pseudo SPLIT[:YYYY-MM-DD]` (repeatable) also fit on a split's Clients Pseudo-Labelled at a Shifted Cutoff (default 2025-10-03). Their Candidate Streams come from their transactions before it and their labels from `pseudo-labels`, so no label file is read and any split may be a source; a train Client may appear with its real label and as a Pseudo-Labelled Client. `--pseudo-weight W` (default 0.5) is their sample weight (real-labelled Clients weigh 1) and `--pseudo-min-payments N` (default 4) the labeller's setting. Pseudo-Labelled Clients join every fold's fit but are never scored, so the out-of-fold rows and the tuned decision layer are real-labelled only. The model metadata and the `evaluate` log row (`training_clients`, `pseudo_sources`, `pseudo_weight`, `pseudo_min_payments`, `fidelity`) record what it was trained on; `fidelity` is the latest fidelity check's verdict and run (`fail ...` today), or `unchecked`. Training still runs when that check failed; the model column reads `ranker+pseudo`.
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
