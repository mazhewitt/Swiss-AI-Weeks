# Stream ranker with Pseudo-Labels from Shifted Cutoffs

**Status:** ready-for-agent

## Problem Statement

Our best submission is a hand-written rule (E1 plus a `none`-gate) at 0.533 macro-F1 on train. The learned model (E2, LightGBM on per-Client stream features) scored only 0.472 on the selection set, partly because one of its features, the family-description share, drifts between train and test.

A ceiling analysis on train shows where the rule loses:

- 47% of Clients have more than one live Recurring Stream due within the Horizon, and "soonest projected next payment" is a weak guess among them. Perfect picking among those candidates would give 0.698.
- Perfect `none` detection alone would give 0.628, and the two together 0.856.
- The label's Merchant Family is in *some* detected Recurring Stream for 94% of Clients who pay one, but in the live, due-within-Horizon candidate set only 76% of the time. The candidate filter throws away answers.
- Only 6% of labels are a family the Client has never paid before.

We also have 2,000 labelled Clients but 14,000 more unlabelled histories: the 10,000 unlabelled Clients plus the valid and test histories. Every history runs to 2025-12-31, so moving the Cutoff back 90 days lets us observe what each Client actually paid next, which gives far more examples of stream choice and of churn to `none`.

## Solution

Two new capabilities, each usable on its own:

1. **A Stream Ranker model.** Every Recurring Stream of a Client becomes a Candidate Stream. A LightGBM model scores how likely each Candidate Stream is to carry the Client's Next Recurring Family, using features that describe the stream relative to the Client's other streams. Scores are turned into one probability per allowed label, including `none`, and the existing decision layer (E3) tunes the final label for macro-F1. It plugs into the existing `train`, `cv`, `evaluate` and `submit` commands as `--model ranker`.
2. **Pseudo-Labels from Shifted Cutoffs.** A labeller finds, for any Client and any Cutoff, the family of the first Recurring Stream payment in the following Horizon, or `none`. A new `pseudo-labels` command writes these for any split at a Shifted Cutoff and runs a fidelity check that shows whether the Pseudo-Labels behave like the real ones. The ranker can then train on Pseudo-Labels pooled with the real labels.

The milestone-2 rule is registered as a model (the `rules` model with the `none`-gate as an option) and logged on the selection set, so the ranker has an honest previous-best to beat.

## User Stories

### Baseline

1. As a team member, I want the milestone-2 rule available as `train/evaluate --model rules` with a `none`-gate option, so that the submission we uploaded is reproducible from the CLI.
2. As a team member, I want the `none`-gate threshold (the longest live stream's payment count at or below which we predict `none`) to be a parameter with default 4, so that the loop and experiments can tune it.
3. As a team member, I want the rule with and without the gate logged on the selection set, so that every later experiment compares against the real previous best.
4. As a team member, I want the E1 slow reference test to reflect the current detector (0.5365 ± 0.02 on the selection set), so that the test guards against regressions rather than against an out-of-date number.

### Pseudo-Labels

5. As a team member, I want a labeller that returns, for every Client in a transaction table, the Merchant Family of the first Recurring Stream payment strictly after a given Cutoff and within the Horizon, or `none`, so that I can make labels without label files.
6. As a team member, I want the labeller to find Recurring Streams using all of the Client's history, both before and after the Shifted Cutoff, so that a stream that starts inside the Horizon still counts once it has repeated.
7. As a team member, I want the minimum number of payments a stream needs for its Horizon payment to count to be a parameter, so that the fidelity check can pick it.
8. As a team member, I want Decoy Transactions, one-off payments and refunds inside the Horizon never to become a Pseudo-Label, so that Pseudo-Labels follow the Next Recurring Family definition.
9. As a team member, I want a payment with a Filler Description inside a real stream to count as that stream's family, so that noisy descriptions don't make Pseudo-Labels `none`.
10. As a team member, I want a Client with no qualifying payment in the Horizon to get `none`, so that churn is represented.
11. As a team member, I want the labeller to refuse a Shifted Cutoff whose Horizon runs past the end of the known history (2025-12-31), so that no Pseudo-Label is built from a partly observed Horizon.
12. As a team member, I want the features and the Candidate Streams at a Shifted Cutoff built only from transactions before that Cutoff, so that nothing from the Pseudo-Label's Horizon leaks into its inputs.
13. As a team member, I want `rf pseudo-labels --split <split> --cutoff <date>` to write a Pseudo-Label table for any split, including unlabeled, valid and test, so that I can inspect them and train on them.
14. As a team member, I want Pseudo-Label generation never to read a label file, so that it is allowed on valid and test without unlocking any label guard.
15. As a team member, I want the command to print the Pseudo-Label distribution, so that I can see the `none` share and each family's share at a glance.
16. As a team member, I want a fidelity check that, on train, compares the Pseudo-Label `none` share with the real one and the milestone-2 rule's macro-F1 against Pseudo-Labels at the Shifted Cutoff with its macro-F1 against real labels at the real Cutoff, so that I know whether the Pseudo-Label task resembles the real task.
17. As a team member, I want the fidelity check to state pass or fail against explicit tolerances (`none` share within 0.05 of the real share; rule macro-F1 within 0.05 of its real score), so that the decision to use Pseudo-Labels is not a judgement call.
18. As a team member, I want the fidelity result appended to the experiment log with its parameters, so that the decision is traceable.
19. As a team member, I want Pseudo-Label tables cached by split, Shifted Cutoff, detector version and labeller parameters, so that repeated training runs don't re-detect streams for 14,000 Clients.

### Stream Ranker

20. As a team member, I want `train --model ranker` to fit on the train Clients, so that the ranker plugs into the pipeline like E1 and E2.
21. As a team member, I want every detected Recurring Stream to be a Candidate Stream, not just Active Streams due within the Horizon, so that the 18% of answers the old filter dropped are reachable.
22. As a team member, I want each Candidate Stream described by its family, projected days to next payment, its rank by that among the Client's streams, payment count, days since first and last payment, period, gap regularity, amount stability, amount and refund rate, so that the model can learn which stream comes next.
23. As a team member, I want Client-level context on every candidate row (number of streams, number of Active Streams, longest Active Stream, earliest Active Stream start), so that the model can learn when a Client is likely `none`.
24. As a team member, I want no ranker feature to use the family-description share or any per-transaction description statistic, so that the drift that sank E2 cannot recur (ADR 0001).
25. As a team member, I want the ranker's per-Client output to be one probability per allowed label summing to 1, so that the decision layer, CV and evaluation work unchanged.
26. As a team member, I want a Client with no Candidate Streams to get a `none`-dominated probability row rather than an error, so that every Client gets a prediction.
27. As a team member, I want a Client with two streams in the same family to have that family's probability built from both, so that duplicates don't split the family's score.
28. As a team member, I want `cv --model ranker` to produce out-of-fold probabilities with Clients (not candidate rows) as the fold unit, so that no Client's candidates straddle folds.
29. As a team member, I want `train --model ranker --decision tuned` to fit per-class weights and the `none` threshold on out-of-fold probabilities, so that macro-F1 is optimised the same way as for E2.
30. As a team member, I want `evaluate --model ranker` to log the run on the selection set with a paired bootstrap against the previous best, so that improvements are judged honestly.
31. As a team member, I want `submit --model ranker` to write a valid submission for the test Clients, so that the ranker can go straight to a milestone.
32. As a team member, I want the saved ranker to reload and give identical probabilities, so that submission and evaluation use the same model.

### Ranker with Pseudo-Labels

33. As a team member, I want `train` and `cv` to accept Pseudo-Label sources (splits and Shifted Cutoffs), so that the ranker can learn from far more Clients.
34. As a team member, I want Pseudo-Labelled Clients pooled with the real-labelled Clients under a weight parameter (default 0.5), so that the real labels still anchor the model and the weight can be tuned.
35. As a team member, I want Pseudo-Labelled rows never to enter the validation fold of cross-validation or the decision layer's fit, so that out-of-fold scores and the tuned `none` threshold reflect real labels only.
36. As a team member, I want a Client whose real label is used in training also to be allowed as a Pseudo-Labelled Client at a Shifted Cutoff, so that train histories contribute twice at different Cutoffs.
37. As a team member, I want the experiment log row to record the Pseudo-Label sources, Shifted Cutoffs, weight and total training Clients, so that runs can be compared.
38. As a team member, I want Pseudo-Label training to remain possible even if the fidelity check fails, with the failed check noted on the log row, so that we can still measure it but nobody mistakes it for a validated setup.

### Milestone

39. As a team member, I want the best of rule, ranker and ranker-with-Pseudo-Labels on the selection set, judged by paired bootstrap, refit on train plus selection and written as a submission, so that the Day-2 12:00 milestone uses the strongest honest candidate.
40. As a team member, I want the submission file and the command that produced it committed, so that the repository link in the form reproduces it.

## Implementation Decisions

- **Terms.** Candidate Stream, Shifted Cutoff and Pseudo-Label are added to the domain glossary. The labeller implements the Next Recurring Family definition at any Cutoff.
- **Labeller lives with stream detection.** A new public function beside the stream detector takes a transaction table, a Cutoff and a Horizon and returns a label per Client. It runs stream detection over the full history so it can decide which Horizon payments belong to a Recurring Stream. That needs the detector to expose per-payment stream membership internally, without changing the stream table's columns.
- **Default Shifted Cutoff** is the real Cutoff minus 90 days (2025-10-03 UTC), the latest date whose Horizon is fully observed. Earlier Shifted Cutoffs are allowed but have shorter histories, which makes them less like the real task.
- **Inputs at a Shifted Cutoff** are transactions strictly before it. Stream detection and candidate features are rebuilt on that truncated history with the Shifted Cutoff as the Cutoff.
- **Pseudo-Label sources may include valid and test histories.** This uses only transactions, never labels, so it is the same information the submission model sees at predict time. Selection and holdout Clients can therefore appear as Pseudo-Labelled Clients. That is acceptable because their Pseudo-Labels come from 2025 payments, not from their post-Cutoff labels.
- **Fidelity check** runs on train at the Shifted Cutoff. It passes when the `none` share is within 0.05 of the real train share (about 0.30) and the milestone-2 rule's Pseudo-Label macro-F1 is within 0.05 of its real train score (0.533). Its minimum-payments parameter is chosen by this check.
- **Ranker formulation.** One row per Candidate Stream, target = "this stream's family is the Client's label". A LightGBM binary classifier, using Clients as the grouping unit for every fold split. Per-Client probabilities: each family's score is the highest candidate score in that family; the `none` score is the probability that no candidate carries the label (one minus the highest candidate score); the row is then normalised to sum to 1. The decision layer (E3) then tunes per-class weights and the `none` threshold on out-of-fold rows.
- **Ranker features** come only from the stream table (ADR 0001): per-candidate fields, their rank among the Client's streams, and the Client-level context listed in the stories. No description-share features and no label-derived encodings.
- **Pooling** Pseudo-Labelled Clients with real ones uses a sample weight (default 0.5). Pseudo rows are excluded from every validation fold and from decision-layer fitting.
- **Caching.** Pseudo-Label tables and truncated stream tables are cached with keys that include the split, the Shifted Cutoff, the detector's code version and the labeller parameters, following the existing stream cache.
- **Registry.** The ranker registers as `ranker` in the model registry and follows the existing model protocol (fit, predict_proba, save, load). Pseudo-Label options go through `train` and `cv` flags and are saved in the model's metadata.
- **Baseline.** The rules model from the MVP's ticket 05 is finished here with a `none`-gate parameter. Its slow reference is amended to 0.5365 ± 0.02 on the selection set.
- **Evaluation discipline is unchanged.** Selection set for decisions, sealed holdout only in human-started checkpoint mode, paired bootstrap against the previous best on the same split, deltas under 0.03 are ties, and per-run predictions committed.

## Testing Decisions

- Good tests assert external behaviour: labels, probability tables, CLI outputs, log rows, refusals. They don't assert internal feature columns or intermediate frames.
- **Seam 1 (the `rf` CLI against the fixture data)** covers:
  - `pseudo-labels` output and its fidelity report
  - `train`, `cv`, `evaluate` and `submit` with `--model ranker`, with and without Pseudo-Label sources
  - that Pseudo-Label generation reads no label file on valid or test
  - that pseudo rows are absent from out-of-fold output
  - that a Client with no streams still gets a valid row
  - that the ranker save/load round trip gives identical probabilities
  - that the `rules` model's `none`-gate changes predictions for a Client whose longest live stream is short
  - Prior art: the existing pipeline CLI and LightGBM pipeline tests.
- **Seam 2 (the stream module's public functions)** covers the labeller with hand-built histories:
  - biweekly and monthly streams due at different dates in the Horizon, where the earlier one wins
  - a stream that stopped before the Shifted Cutoff, giving `none`
  - a new stream that starts inside the Horizon and repeats
  - a Decoy Transaction or refund as the first Horizon payment, which is ignored
  - a Filler Description inside a stream, which counts
  - no Horizon payments, giving `none`
  - a Shifted Cutoff past the observable end, which is refused
  - Decoy injection that leaves Pseudo-Labels unchanged (mirrors the ADR 0001 detector test)
  - Prior art: the existing stream detection tests.
- **Slow, marked real-data tests:**
  - the fidelity check runs on train and reports its figures
  - the ranker trains and scores on the selection set within the evaluation-time budget

## Out of Scope

- The Opus self-improvement loop. It gets its own spec and will tune the ranker's parameters, the Pseudo-Label weight and the Shifted Cutoffs.
- Sequence models (transformers or RNNs over payment histories), LLM-based classifiers and Open JEV.
- Predicting brand-new streams whose family never appears in the history (6% of labels).
- Ensembling several ranker variants. This is a later candidate.
- Changing the E2 model. The ranker supersedes it and E2 stays as a logged comparison.
- The MVP's ticket 07 (milestone-2 path). It is overtaken by this spec's milestone ticket.

## Further Notes

- The ceiling figures (0.628, 0.698, 0.856) and candidate coverage (76% against 94%) were measured on train with the hardened detector. They motivate the design but are not targets.
- Deadline: the Day-2 12:00 CEST milestone should carry the best candidate from this spec, and the Day-2 17:30 submission goes to the jury.
- If the fidelity check fails, the ranker still ships on real labels alone. Pseudo-Labels then become the self-improvement loop's first investigation.
