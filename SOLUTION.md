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

The milestone-2 submission is the gated rule, reproduced exactly by (`join_strays=false`: the stream detector before ticket 11, which lets stray Filler Description payments join a stream on schedule):

```sh
uv run rf train --model rules --none-gate --param join_strays=false
uv run rf submit --model rules --name milestone2_rules_none_gate_v2
```

The Day-2 12:00 milestone submission is made by `bash scripts/day2_noon_milestone.sh` (it refits the rule with `--param join_strays=false`, the detector it was made with): `rf compare` over the rule, ranker and ranker-with-Pseudo-Labels runs on the selection set found a three-way tie, so the simplest candidate, the gated rule, was refit on train plus the selection set (`submissions/day2_noon_rules_gate4.csv`, with the comparison beside it in `day2_noon_rules_gate4.md`).

- `streams --split valid` detects (or loads cached) Recurring Streams and prints a per-family summary. `--param NAME=VALUE` (repeatable) overrides one stream detection parameter, e.g. `--param amount_tolerance=0.08`. Each parameter set is cached separately under `artifacts/streams/`, and a change to the detector code, family table or raw data rebuilds.
- `pseudo-labels --split <split> --cutoff <date>` writes each Client's Pseudo-Label at a Shifted Cutoff (default 2025-10-03, the latest whose 90-day Horizon is fully observed) to `artifacts/pseudo_labels/<split>-<cutoff>-min<N>.csv` (`client_id, cutoff_date, target_next_recurring_merchant`, like a label file) and prints the per-label counts and shares. It works for every split, valid and test included, because it reads transactions only: any label read while it runs is refused. `--min-payments N` is the payments a Recurring Stream needs for its Horizon payment to count (default 4, chosen by the fidelity check); `--param NAME=VALUE` overrides the labeller's stream detection. Tables are cached under `artifacts/pseudo_labels/cache/` by split, Shifted Cutoff, detector version and labeller parameters.
- `pseudo-labels --split train --fidelity` is the fidelity check: for each `min_payments` in `--candidates` (default 2-10) it compares the Pseudo-Label `none` share with the real train share, and the milestone-2 rule's macro-F1 against Pseudo-Labels (its inputs are the transactions before the Shifted Cutoff only) with its macro-F1 against the real labels. It chooses the setting with the smallest worse gap relative to the tolerances (0.05 each), prints PASS or FAIL, writes that setting's table and appends a `fidelity` row to the log. On the real train split it fails (the `fidelity` row of 2026-09-24).
- `train --model rules` is E1, the rule baseline: among each Client's Active Streams whose projected next payment falls within the Horizon, it predicts the family of the one due soonest, else `none`. `--none-gate [N]` adds the milestone-2 `none`-gate: a Client whose longest surviving stream has at most N payments (default 4) gets `none`; it is off unless given. `--ordering most_recent|longest` picks by another rule, and `--param NAME=VALUE` overrides stream detection parameters; all are saved with the model, and the log's model column names the variant (`rules+gate4`).
- `train --model lgbm` fits E2: LightGBM multiclass (balanced class weights, default hyperparameters) on one feature row per Client, built only from its Recurring Streams (ADR 0001): the top three Active Stream slots, a block per Merchant Family and the `none` signals. `features --split test` writes the rows the trained model scores for a split (same columns for every split); for the Clients it was fitted on (train, plus the selection set after `--with-selection`) those are the out-of-fold rows it trained on, so no row carries its own Client's label. `cv --model lgbm` saves its out-of-fold probabilities.
- `train --model ranker` fits the Stream Ranker: every detected Recurring Stream is a Candidate Stream, and a LightGBM binary model scores whether it carries the Client's Next Recurring Family, from stream-table features only (ADR 0001). Each family takes its best candidate's score, `none` is one minus the best score, and the row is normalised; a Client without streams is `none`. It works with `cv`, `evaluate`, `submit` and `--decision tuned` like any model.
- `train` and `cv --model ranker --pseudo SPLIT[:YYYY-MM-DD]` (repeatable) also fit on a split's Clients Pseudo-Labelled at a Shifted Cutoff (default 2025-10-03). Their Candidate Streams come from their transactions before it and their labels from `pseudo-labels`, so no label file is read and any split may be a source; a train Client may appear with its real label and as a Pseudo-Labelled Client. `--pseudo-weight W` (default 0.5) is their sample weight (real-labelled Clients weigh 1) and `--pseudo-min-payments N` (default 4) the labeller's setting. Pseudo-Labelled Clients join every fold's fit but are never scored, so the out-of-fold rows and the tuned decision layer are real-labelled only. The model metadata and the `evaluate` log row (`training_clients`, `pseudo_sources`, `pseudo_weight`, `pseudo_min_payments`, `fidelity`) record what it was trained on; `fidelity` is the latest fidelity check's verdict and run (`fail ...` today), or `unchecked`. Training still runs when that check failed; the model column reads `ranker+pseudo`.
- `train --model blend` mixes the gated rule and the Stream Ranker as `rule_weight * rule + (1 - rule_weight) * ranker` (`--rule-weight W`, default 0.5). The rule inside it takes `--none-gate`, `--ordering` and `--param`, and the ranker takes `--pseudo`, in `train` and `cv` alike. The saved blend is `artifacts/blend.json` plus its parts beside it (`blend.rules.json`, `blend.ranker.json`). `scripts/blend_weight_sweep.py` chooses the weight on train out-of-fold probabilities only (`experiments/blend_weight_sweep.csv`: 0.1 is best, +0.002 over the ranker alone). On the selection set the chosen blend scores 0.5726, a tie with the ranker with Pseudo-Labels (0.5735), so the blend is kept as a logged comparison only.
- `train` and `cv --model ranker --none-model` add a Client-level `none` model: a LightGBM binary model fitted on the real-labelled Clients with Candidate Streams (never on Pseudo-Labelled ones) gives P(`none`), and each family keeps its share of the ranker's family scores. Its features come from the stream table and the member payments and matched refunds of detected streams (ADR 0001), plus the ranker's own `none`, cross-fitted for training Clients. `scripts/none_model_variants.py` chose the feature set on train out-of-fold (`experiments/none_model_variants.csv`: nested tuned macro-F1 0.5699 -> 0.6095). On the selection set it scores 0.5764, a tie with the ranker with Pseudo-Labels; the model column reads `ranker+none+pseudo+tuned`. `scripts/day2_final_ranker_none.sh` writes the refit candidate `submissions/day2_final_ranker_none.csv`.
- `evaluate` reports macro-F1 over the eight allowed labels and per-family F1, and appends one row to `experiments/log.csv`. For E1 the row also records coverage (how often the true family is among the surviving streams, over Clients whose label is a family) and selection accuracy given coverage.
- E3 decision layer, for any model: `train --model <m> --decision tuned` also fits one weight per label plus a `none` threshold by grid search on the training Clients' out-of-fold probabilities (never valid labels). `evaluate` and `submit` then take `--decision tuned` (default `argmax`), or `--decision <file>.json` with hand-set `weights` and `none_threshold` (`none` when P(none) >= 1 - threshold; 0 never predicts `none`, 1 always does). A tuned decision is refused on Clients it was fitted on.
- `compare --run RUN --run RUN ...` picks a milestone candidate among logged runs, simplest first: it scores each run's committed predictions on the same Clients, gives each its macro-F1 with a 95% bootstrap interval, compares every pair with the paired bootstrap (deltas under 0.03 are ties) and names the simplest candidate that no other candidate beats. `--out file.md` also writes the comparison as Markdown. It refuses sealed-holdout runs, runs on different splits and runs without saved predictions, and reads the selection labels only after every prediction is loaded.
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
