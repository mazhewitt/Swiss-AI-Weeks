# Spec: Next Recurring Family MVP

Status: ready-for-agent

## Problem Statement

We are a two-member team (one human and Claude with its subagents) entering the UBS Swiss{ai}Weeks 2026 challenge. For each Client we must predict the Next Recurring Family: the Merchant Family of the first Recurring Stream payment in the 90-day Horizon after the 2026-01-01 Cutoff, or `none`. Rank is the best macro-F1 across four milestone submissions. The final milestone also goes to a jury that wants model comparison, limitations and interpretability.

There is no code yet. The data is deliberately hard:

- Descriptions are ambiguous across MCCs.
- Filler Descriptions sit inside real streams.
- Decoy Transactions rise sharply from train (0.16 per Client) to valid (3.1) and test (4.5).
- The label is defined by future behaviour, including cancellations. The best hand rule reaches only about 0.48 macro-F1.

We need a pipeline that produces valid submissions early, measures every change honestly on data that resembles test, and leaves a clear record of what worked for the jury.

## Solution

A small Python package with one command-line tool that:

- fetches the data
- detects each Client's Recurring Streams in a way that Decoy Transactions cannot affect
- trains and evaluates three increasingly capable predictors
- writes validated submission files

The three predictors are built in order:

- **E1, rule baseline (milestone 1).** Take Active Streams with 3+ payments and predict the Merchant Family whose projected next payment is soonest after the Cutoff, else `none`.
- **E2, LightGBM on stream features (milestone 2).** Learns the family and the `none` boundary from per-stream and per-family features, including the signals that separate `none` Clients (short, late-starting streams).
- **E3, decision layer (milestone 2b or 3).** Per-class weights and a `none` threshold, fitted on out-of-fold probabilities to maximise macro-F1 rather than taking the argmax.

Every run appends a row to an experiment log with valid macro-F1, per-family F1, and whether the change beat the previous best by more than bootstrap noise. That log becomes the jury story.

## User Stories

1. As the team, I want a single command that unpacks the challenge data into a local, git-ignored raw area, so that nobody commits 209 MB of data and setup is repeatable.
2. As the team, I want data fetching to be idempotent, so that re-running it never corrupts or duplicates the raw data.
3. As the team, I want the organisers' files (the challenge material and upstream README) left untouched, so that pulling upstream changes into our fork never conflicts.
4. As the team, I want typed loaders for transactions, labels and the sample submission for every split, so that every component reads data the same way with UTC timestamps.
5. As the team, I want the Cutoff, Horizon length and allowed label set defined once, so that no component can disagree about them.
6. As the team, I want a training run to fail loudly if it tries to read the valid labels, so that model selection can never leak evaluation data.
7. As the team, I want Recurring Streams detected from each Client's outgoing card payments, so that every prediction rests on the same stream view.
8. As the team, I want each Recurring Stream assigned exactly one Merchant Family from MCC plus description, so that ambiguous descriptions like "premium plan" resolve to software under MCC 5734 and to streaming or music under 5812.
9. As the team, I want music and streaming streams on MCC 5812 separated by description hints and then amount, so that the two most-confused families are split the way the data supports.
10. As the team, I want Filler Descriptions kept inside the stream they belong to, so that streams do not fragment when a payment carries a generic description.
11. As the team, I want Decoy Transactions excluded from streams, so that stream output is the same whether a history has train-like or test-like decoy rates.
12. As the team, I want refunded stream payments kept in the stream with a refund rate recorded, so that fully refunded streams (which still carry their family's label) are not lost.
13. As the team, I want transfers and shop-description payments never treated as streams, so that everyday spending cannot masquerade as recurring payments.
14. As the team, I want each stream to report its period, amount, amount stability, timing regularity, payment count, first and last payment, and projected next payment, so that both the rules and the model have what they need.
15. As the team, I want a projected next payment that is overdue at the Cutoff rolled forward by whole periods, so that a stream due just before the Cutoff still counts in the Horizon.
16. As the team, I want stream detection to ignore anything after the Cutoff, so that no feature can peek into the future.
17. As the team, I want an Active Stream flag (3+ payments, last paid within about 1.6 periods of the Cutoff), so that stopped streams are distinguishable from live ones.
18. As the team, I want stream thresholds (amount tolerance, active window, minimum payments) held in one parameter object, so that they can be swept without code changes.
19. As the team, I want the E1 rule baseline to predict for every Client in a split, including Clients with no streams, so that its output is always a complete submission.
20. As the team, I want E1 to report its coverage (how often the true family is among surviving streams) and selection accuracy, so that we can see whether errors come from detection or from choosing among streams.
21. As the team, I want an all-`none` submission produced by the foundation, so that milestone 1 has a valid safety net even if E1 is late.
22. As the team, I want macro-F1 computed over the fixed eight-label set, so that a family we never predict counts as zero instead of silently disappearing.
23. As the team, I want per-family F1 reported alongside macro-F1, so that we see which families drag the score down.
24. As the team, I want each new result compared to the previous best with a paired bootstrap, so that we do not chase differences that are noise. Deltas under about 0.03 count as ties.
25. As the team, I want every train or evaluate run to append one row to the experiment log (date, change, valid macro-F1, per-family F1, conclusion), so that the jury story writes itself.
26. As the team, I want the experiment log written only on the main line of work, so that parallel agents never conflict on it.
27. As the team, I want a submission writer that rejects missing, extra or duplicate Client IDs and labels outside the allowed set, so that every uploaded file is valid.
28. As the team, I want submissions committed with the code that produced them, so that every milestone file can be traced and reproduced.
29. As the team, I want E2 to build one feature row per Client from their streams: the top three Active Streams, a block per Merchant Family, and the `none` signals (longest active stream length, earliest active stream start, share of family-specific descriptions). This lets the model learn what the rule cannot.
30. As the team, I want features built only from Recurring Streams and never from raw transaction counts, so that the decoy shift between train and test cannot move them (ADR 0001).
31. As the team, I want the same feature schema produced for every split, so that a model trained on one split scores any other.
32. As the team, I want unseen descriptions encoded as unknown, so that test-only noise cannot break feature building.
33. As the team, I want E2 trained with balanced classes and default hyperparameters first, so that the first number is honest before any tuning.
34. As the team, I want E2 to save out-of-fold probabilities, so that E3 can be fitted without touching valid.
35. As the team, I want E2 accepted only if it beats E1 and no Merchant Family loses more than 0.05 F1, so that an average gain does not hide a collapsed class.
36. As the team, I want E3 to fit per-class weights and a `none` threshold on out-of-fold probabilities, so that the final decisions target macro-F1 rather than accuracy.
37. As the team, I want E3 with uniform weights and no threshold to reproduce plain argmax exactly, so that the decision layer can never be worse by construction.
38. As the team, I want model selection on the valid selection set (the roughly 700 valid Clients outside the sealed holdout) for milestones 1–2, then 5-fold stratified cross-validation over train plus the valid selection set from milestone 3, so that later tuning uses the data most like test without overfitting a small holdout.
39. As the team, I want milestone submission models refit on train plus the valid selection set, and only the final milestone-4 model refit on all of valid (after the sealed holdout's last checkpoint scoring), so that the model learns from the test-like decoy distribution without spending the holdout early.
40. As the human team member, I want to review valid scores and confirm before any milestone submission file is generated, so that nothing is uploaded without my say.
41. As the human team member, I want to upload submissions to the form myself, so that the external submission stays under my control.
42. As a jury member, I want to see the rule baseline, the learned model and the decision layer compared on the same evaluation with per-family F1, so that I can judge what each approach contributed.
43. As a jury member, I want the limitations stated (the label depends on future cancellations, the decoy shift, the ~0.48 rule ceiling), so that I trust the reported numbers.
44. As a future contributor, I want the domain terms in the glossary used consistently in code and docs, so that "stream", "family" and "decoy" mean one thing everywhere.
45. As a future contributor, I want a slow test that checks E1 on real valid selection data lands within ±0.04 of 0.479, so that regressions in stream detection are caught against reality, not just fixtures.
46. As the team, I want about 300 valid Clients sealed into a holdout before any tuning happens, so that we have one honest estimate left after many rounds of selection.
47. As the team, I want the sealed holdout split deterministically (fixed seed, stratified by label), so that every run and every agent sees the same split.
48. As the team, I want any read of sealed-holdout labels outside an explicit checkpoint evaluation to fail loudly, so that neither a person nor an agent can tune against it by accident.
49. As the human team member, I want to trigger sealed-holdout scoring myself at milestone checkpoints, so that I can tell whether gains on the selection data are real or overfitted.
50. As the team, I want a full evaluation run (stream detection cached, E2 cross-validation, E3 fit, scoring) to finish in about two minutes or less, so that a later self-improvement loop can run many rounds in a day.

## Implementation Decisions

**Stack and layout**
- Python 3.12 managed with uv, using pandas, LightGBM, scikit-learn and pytest.
- One installable package with a single CLI offering `fetch-data`, `train`, `evaluate` and `submit`. Notebooks are for exploration only.
- The project lives at the repository root beside the organisers' files, which are never edited. A short solution README is added at the root.
- Raw and interim data are git-ignored. Submission files are committed.

**Modules**
- **Configuration.** Owns the Cutoff (2026-01-01), the 90-day Horizon and the eight allowed labels.
- **Data access.** Owns loaders per split, the deterministic split of valid into a selection set and a sealed holdout of about 300 Clients (fixed seed, stratified by label), and the guards that raise if valid labels are read during a training run or sealed-holdout labels are read outside an explicit checkpoint evaluation.
- **Stream detection.** Takes one or more Clients' transactions and the Cutoff, and returns a table of Recurring Streams. Each stream carries its Client, Merchant Family, period, median amount, amount variation, gap regularity, payment count, first and last payment dates, projected next payment, active flag and refund rate. The Merchant Family mapping lives inside this module: a fixed lookup from MCC plus description keywords, with the music/streaming split by description hint and then amount. The lookup is taken from the fact-finding family table. The detector's thresholds live in one parameter object.
- **Rules (E1).** Picks one stream per Client by an ordering rule (default: soonest projected next payment within the Horizon among Active Streams), else `none`. Reports coverage and selection accuracy.
- **Features (E2).** Builds one row per Client from the stream table: the top three Active Stream slots, a per-family block and the `none` signals.
- **Model (E2).** LightGBM multiclass wrapper with balanced class weights. It saves out-of-fold probabilities.
- **Decision layer (E3).** Per-class weights plus a `none` threshold, fitted on out-of-fold probabilities by grid search to maximise macro-F1.
- **Evaluation.** Macro-F1 over the fixed label set, per-family F1, paired bootstrap against the previous best, and appending rows to the experiment log. It also has a separate checkpoint mode, started only by the human, that scores the sealed holdout and records it in the log as its own row type. The stream table for each split is cached, so that a full evaluation run takes about two minutes or less.
- **Submission.** Writes and validates the CSV against the sample submission's Client IDs and the allowed labels.

**Behaviour**
- Only outgoing card payments are stream candidates. Transfers, shop descriptions and service-fee rows are excluded.
- Payments are grouped per Client and family, then clustered by amount (a log-amount gap of about 6% splits clusters).
- Decoy Transactions are excluded because their descriptions and scattered MCCs never meet the family evidence a stream needs.
- Filler Descriptions join a stream when their amount and MCC fit it.
- Refunds do not end or shrink a stream. They are counted in its refund rate.
- Amounts are compared in native currency, since 99.3% of streams stay in one currency.
- Only monthly and biweekly periods exist in the data. Doubled gaps from missed payments must not change a stream's period.
- Every model feature comes from the stream table, never from raw transactions (ADR 0001).

**Evaluation protocol**
- Valid is split once, before any tuning, into a selection set (~700 Clients) and a sealed holdout (~300 Clients).
- Milestones 1–2: train on train, select on the valid selection set.
- From milestone 3: 5-fold stratified cross-validation over train plus the valid selection set.
- The sealed holdout is scored only at human-triggered milestone checkpoints. If gains on the selection set do not appear on it, we treat that as overfitting and stop tuning in that direction.
- Milestone submissions are refit on train plus the valid selection set. The final milestone-4 model is refit on all of valid after the holdout's last checkpoint scoring.
- Deltas under about 0.03 macro-F1 are treated as ties.

**Milestones and time boxes**
- An all-`none` safety submission comes out of the foundation.
- E1 targets milestone 1. If it is not ready by 11:15 on Day 1, the safety file stands.
- E2 targets milestone 2. If it is not ready by 15:30, milestone 2 is E1 with swept stream thresholds.
- E3 follows at milestone 2b or 3.

**Build process**
- Tickets live under the MVP feature folder, and each is implemented test-first by a subagent.
- **Order:**
  1. The scaffold is built first.
  2. Data access, evaluation/submission, stream detection and the family mapping are built in parallel, in isolated worktrees.
  3. Then E1.
  4. Then features, then the model, then the decision layer.
- **Review:** each ticket gets an adversarial-critic review that attacks tests that still pass with a stub, shuffled labels or leakage. A ticket gets at most two fix rounds before it is escalated to the human.
- **Merging:** the workflow merges and commits after a clean review and a green test suite. It pauses for the human before generating any milestone submission.
- **Agent routing:** every agent runs on Opus. Mechanical tickets (scaffold, data access, evaluation/submission, decision layer) run at low effort. No more than 10 agents run at once.

## Testing Decisions

**What a good test is:** it exercises external behaviour through a seam, uses small hand-built fixtures whose expected output a human can check by eye, and would fail if the behaviour were stubbed, the labels shuffled, or future data leaked in. Tests never assert on private helpers or intermediate data structures.

**Seam 1: the pipeline, through the CLI, against fixture data.** A fixture data folder mirrors the real file layout with a handful of Clients. Tests cover:
- idempotent data fetching
- macro-F1 and per-family F1, including a never-predicted family scoring zero
- submission validation rejecting missing, extra or duplicate IDs and bad labels
- the valid-label leak guard raising during training
- the sealed-holdout guard raising on any label read outside checkpoint mode, and checkpoint mode succeeding
- the selection/holdout split being identical across runs, disjoint, covering all of valid, and stratified by label
- an experiment-log row being appended per run
- E1, E2 and E3 each producing a complete, valid prediction for every Client, including Clients with no streams
- E2 reaching F1 ≥ 0.95 on a fixture with easily separable families
- a fixture with a leakage trap that fails unless out-of-fold fitting is used
- E3 with uniform weights reproducing argmax; a `none` threshold of 0 yielding no `none` and of 1 yielding all `none`

**Seam 2: stream detection.** Tests cover:
- the same Recurring Streams before and after injecting Decoy Transactions at test-like rates
- Filler Descriptions staying in their stream
- identical output when post-Cutoff transactions are appended
- "premium plan" resolving to software under MCC 5734 and to streaming or music under 5812
- music versus streaming split by amount
- an overdue projected payment rolled forward into the Horizon
- a missed payment not doubling the period
- refunded payments kept with a refund rate
- transfers and shop payments never forming streams

**Slow real-data checks.** Two marked tests, run on request:
- E1's macro-F1 on the valid selection set lands within ±0.04 of 0.479. The tolerance is wider than ±0.03 because the reference number was measured on all 1,000 valid Clients.
- A full evaluation run on the real data finishes within about two minutes.

**No seams** for features, the model wrapper or the decision layer. They are covered through Seam 1.

**Prior art.** There are no tests in the repository yet. The fact-finding reference scripts and data facts in the MVP research folder give expected behaviour and numbers to derive fixtures from.

## Out of Scope

- Pseudo-labels generated with shifted Cutoffs on the unlabelled set (a Day 2 "should", with its own spec).
- Learning the family mapping from labels, and clustering streams to resolve ambiguous descriptions (the fixed family table makes this unnecessary for the MVP).
- Synthetic decoy augmentation (a fallback only if the gap between train and valid scores grows).
- Open Jev stream naming, pretrained sequence models and TS-JEPA ("could" items).
- Hyperparameter tuning beyond defaults and the E3 decision layer.
- SHAP, confidence intervals and error decomposition for the jury write-up (a separate Day 2 ticket reading saved predictions).
- Automating the submission form. The human uploads by hand.
- The self-improvement loop, in which Opus agents propose, implement and evaluate changes against a fixed acceptance check. It gets its own spec once E2 has landed. This spec only prepares for it with the sealed holdout, the guards and the evaluation time budget.
- Changes to the organisers' files.

## Further Notes

- Glossary: CONTEXT.md. Architecture decision: ADR 0001 (features only from Recurring Streams).
- Data facts, including the family table, stream statistics, label and `none` semantics and the decoy shift, are recorded in the MVP research folder with reference analysis scripts. Those scripts are throwaway analysis code: use them for numbers and behaviour, not as the implementation.
- The label is not fully recoverable from history: the best rule scores about 0.48, and 73% of `none` Clients still have Active Streams. Expect learned `none` detection to be where E2 gains most.
- Open question for the organisers (ask at the Q&A): may code be built before Day 1?
- Milestone deadlines (CEST): Day 1 12:00 and 17:00, Day 2 12:00 and 17:30 (jury).
