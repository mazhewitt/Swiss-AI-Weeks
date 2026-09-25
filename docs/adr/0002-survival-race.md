# The Survival Race: one model for `none` and the family

Ticket 13. It is added beside the Stream Ranker, and it does not replace it.

## Context

The Stream Ranker scores each Candidate Stream on its own, and a separate Client-level model predicts `none`. We broke down where the best such model (v2, run `20260925T001755-e309a3`, 0.5871 on selection) loses macro-F1 on the selection set. Fixing each bucket of errors in turn would gain:

```mermaid
flowchart LR
    A["v2 on 700 selection Clients<br/>423 right, 277 wrong<br/>macro-F1 0.587"] --> D1["True family's stream<br/>was detected: 142"]
    A --> D2["True family never<br/>detected: 64"]
    A --> D3["Truth none,<br/>a family predicted: 71"]
    D1 --> P1["Picked a different<br/>detected stream: 111<br/><b>+0.19 if fixed</b>"]
    D1 --> P2["Called it none: 31<br/>+0.05 if fixed"]
    D2 --> P3["+0.08 (wrong family) and<br/>+0.03 (called none) if fixed;<br/>many are brand-new streams"]
    D3 --> P4["+0.06 if fixed"]
    style P1 fill:#fde2c8,stroke:#d9822b,stroke-width:2px
```

The largest loss is choosing among streams we *did* detect. In 84 of the 111 Clients, v2 picked the stream projected to pay first. The true stream was projected a median of 7 days later, which is too far to be schedule jitter on monthly streams.

On train, the labels fit a simple process. Each Active Stream independently survives the Cutoff with probability about 0.5. The label is the soonest survivor, or `none` if no stream survives.

| Active families | Clients | Predicted none / 1st / 2nd | Observed none / 1st / 2nd |
|---|---|---|---|
| 3 | 305 | 0.125 / 0.50 / 0.25 | 0.128 / 0.48 / 0.22 |
| 4 | 78 | 0.06 / 0.50 / 0.25 | 0.064 / 0.46 / 0.19 |

The scripts are in `experiments/analysis/survival/`. This is a pattern found in train labels, the kind a subscription-churn analyst would look for. We did not reverse-engineer the data generator: that was ruled out as against the spirit of the challenge.

## Decision

`--model survival` (`src/recurring_family/survival.py`) models that process directly. It sits beside v2 and starts from the same Candidate Streams:

```mermaid
flowchart TB
    T["Client's card payments"] --> S["Stream detector<br/>(rules, ADR 0001)"]
    S --> C["Candidate Streams<br/>16 stream-table features each"]
    subgraph V2["v2: ranker + none model"]
        direction TB
        R["Stream Ranker<br/>score per stream"] --> F["family share<br/>= best score / sum"]
        N["Client-level none model<br/>104 features + cross-fitted ranker score"] --> M["P(family) = share × (1 − P(none))"]
        F --> M
        PL["Pseudo-Labels<br/>shifted cutoff"] -.-> R
    end
    subgraph SR["Survival Race"]
        direction TB
        SV["Survival model<br/>s = P(stream keeps paying)"] --> RC["Race in projected<br/>payment order"]
        RC --> PR["P(next) per stream,<br/>P(none) = all stop"]
    end
    C --> R
    C --> N
    C --> SV
    M --> DL["Tuned decision layer<br/>per-label weights + none threshold"]
    PR --> DL
    DL --> O["Next Recurring Family<br/>or none"]
```

- **The survival model.** A LightGBM binary model gives every Candidate Stream a survival probability s. Its features are the ranker's 16 stream-table features and nothing else (ADR 0001).
- **The race.** Each Client's streams are ordered by projected next payment. With the order fixed:
  - P(stream i is next) = s_i × Π_{j before i} (1 − s_j)
  - P(`none`) = Π_j (1 − s_j)
  - A family's probability is the sum over its streams. A Client with no Candidate Stream is `none`.

  A worked example: one Client, three streams, each asked in turn whether it keeps paying. The four outcomes sum to 1.

  ```mermaid
  flowchart LR
      G{"gym<br/>due in 4 days<br/>keeps paying?<br/>s = 0.6"} -- "yes 0.6" --> PG["<b>gym</b><br/>0.60"]
      G -- "no 0.4" --> Mu{"music<br/>due in 12 days<br/>s = 0.5"}
      Mu -- "yes 0.5" --> PM["<b>music</b><br/>0.4 × 0.5 = 0.20"]
      Mu -- "no 0.5" --> Cl{"cloud<br/>due in 20 days<br/>s = 0.3"}
      Cl -- "yes 0.3" --> PC["<b>cloud</b><br/>0.4 × 0.5 × 0.3 = 0.06"]
      Cl -- "no 0.7" --> PN["<b>none</b><br/>0.4 × 0.5 × 0.7 = 0.14"]
  ```
- **Training.** One Client's log-likelihood splits into plain binary rows:
  - the label's stream gets target 1;
  - every stream ordered before it gets target 0;
  - streams after it are **censored** and get no row, because nobody saw whether they would have paid;
  - a `none` Client gives every stream target 0.

  ```mermaid
  flowchart LR
      subgraph L["Client labelled music: streams in race order"]
          direction LR
          A["gym<br/>due in 4 days"] --> B["music<br/>due in 12 days"] --> C["cloud<br/>due in 20 days"]
      end
      A -.- RA["row, target 0<br/>due first, but it stopped"]
      B -.- RB["row, target 1<br/>it paid: the label"]
      C -.- RC["no row: censored<br/>nobody saw whether it would pay"]
      style RA fill:#f8d7da,stroke:#c0392b
      style RB fill:#d4edda,stroke:#2e7d32
      style RC fill:#eeeeee,stroke:#999999,stroke-dasharray: 4 3
  ```

  So the fit is an ordinary LightGBM fit on a filtered set of rows: 3,913 of the 6,507 train candidates. The Stream Ranker, in these terms, trains those later streams as negatives.
- **Decision layer.** The same tuned per-label weights and `none` threshold as every other model.
- **What was left out.** No Pseudo-Labels, no Client-level `none` model and no churn features.

## Results

| | Train out-of-fold, nested tuned | Selection (700) | Test (leaderboard) |
|---|---|---|---|
| v2: ranker + `none` model + Pseudo-Labels | 0.6082 | 0.5871 | **0.6059** |
| Survival Race (unprojected-last order) | 0.6239 | **0.5903** | not yet scored |
| Survival Race (monthly-slot order) | 0.6323 | 0.5862 | – |

v2 also scored 0.5960 on the sealed holdout.

Every difference in the table is a tie under our rule (a delta under 0.03). The Survival Race agrees with v2 on 83% of selection Clients and 84% of test Clients.

It gains most where v2 was weakest: mobile F1 rises from 0.52 to 0.70 and insurance from 0.55 to 0.61. It is lower on cloud and software. The probabilities are calibrated. The predicted `none` rate matches the observed rate for Clients with 1, 2 and 3 active families (0.35/0.32, 0.23/0.22, 0.14/0.13). With 4 families it overpredicts: 0.11 against 0.06.

## Trade-offs

**One model, not two.**
- For: a Client's family and `none` probabilities come from one process, so they are consistent by construction. We need no cross-fitting of one model's score into another, and have fewer moving parts.
- Against: it assumes streams survive independently once their features are known. The 4-family overprediction suggests survival is correlated within a Client: Clients who cancel tend to cancel several subscriptions. A Client-level frailty term would model that. It is not built.

**Censoring instead of negatives.**
- For: it is the right likelihood. A stream behind the label's stream is not evidence of a stop.
- Against: it trains on 40% fewer rows than the ranker.

**Unexplained Clients are dropped.** 87 train Clients (4%) have a label family that is not among their Candidate Streams: a brand-new or undetected stream.
- For: their race order is unknown, so they can't give honest rows.
- Against: the model gives such families zero probability. About 6% of labels are families never seen in the history, and no stream-based model can predict them.

**Ranker features only.**
- For: adding the churn features (member-payment gaps, refunds, Client churn history) raised the survival model's AUC from 0.86 to 0.88 but lowered nested macro-F1 (0.6171 against 0.6239). Those are also the features that drift between train and test (ticket 09's review). The 16 ranker features are the most drift-robust set we have.
- Against: some real churn signal is left unused.

**No Pseudo-Labels.**
- For: at a Shifted Cutoff almost no stream stops (ticket 08), so Pseudo-Labels would teach s ≈ 1.
- Against: less training data than the ranker gets.

**A hard race order.**
- For: it is simple and exact given the order.
- Against: projected dates have jitter, so a soft race that integrates over each stream's schedule uncertainty would be more faithful. Not built.

**Single-payment streams have no projected date.**
- The default is monthly-slot: last payment + 30.4 days, rolled forward past the Cutoff, used only to place the stream in the order. It is best on train (nested +0.008, a tie).
- The submitted candidate uses the original order, unprojected-last: these streams race last, in stream-table order. It is best on selection (+0.004, a tie). We fixed the rule before running selection, and by that rule the submitted file stayed.
- `scripts/day2_final_survival.sh` pins `--race-order unprojected-last`, and every run records its order in the log's model column (`survival` or `survival+monthly-slot`).

**Train gains shrink on selection, again.**
- The +0.016 over v2 on train became +0.003 on selection. That is the same pattern as the `none` model's, and it matches the drift between train and valid/test.
- The Survival Race needs fewer parts and fewer drifting features, yet only ties v2. So the two are uploaded as distinct candidates, not one replacing the other.

## Consequences

- **Its own model.** The Survival Race lives beside the Stream Ranker as `--model survival`, and the ranker's code and outputs are unchanged. Tests pin the building of the training rows, the race's closed form, every race order through fit and predict, the label guards and Decoy injection.
- **Candidates.** `submissions/day2_final_ranker_none_v2.csv` (v2) and `submissions/day2_final_survival.csv` (Survival Race) are the two candidates. Only the best upload counts, and they disagree on 16% of test Clients.
- **Open follow-ups:**
  - a Client-level frailty term;
  - a soft race order;
  - an average of v2's and the Survival Race's probabilities;
  - Pseudo-Labels with an `is_pseudo` feature (ticket 13, variant 3, not run).
