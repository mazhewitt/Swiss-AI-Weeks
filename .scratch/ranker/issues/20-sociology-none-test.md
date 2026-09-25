# 20: Do sociological churn mechanisms predict `none`? (pre-registered diagnostic)

Status: ready-for-agent

**Why:** the user asked whether sociology could improve our `none` signal. The churn literature names mechanisms we have not tested as such:
- financial stress cuts discretionary spending first;
- people run "subscription audits" and cancel several at once;
- price rises trigger cancellation;
- discretionary subscriptions go before essential ones.

**Prior evidence against:**
- the train labels fit independent stream survival (about 0.5), which argues against correlated audits;
- classic churn signals were near chance (`experiments/analysis/churn/`);
- ticket 18's raw block already held income and spend windows.

**Expected gain:** below +0.01. This ticket tests the theory cleanly either way. **Diagnostic only; no build.**

## Features (fixed before any label is looked at)

Per Client, from history before the Cutoff (2026-01-01):

1. **`income_trend`:** log((inbound top-up/salary amount in the last 90 days + 1) / (mean 90-day inbound amount over the 270 days before that + 1)).
2. **`spend_contraction`:** the same log ratio for outgoing card spend. Negative means spending is shrinking.
3. **`essential_share`:** the share of card spend in the last 180 days at essential MCCs. The MCC list comes from the standard MCC meanings (groceries, supermarkets, pharmacies, utilities, fuel, public transport), fixed in code before any label is read, and never adjusted afterwards.
4. **`price_rise`:** the maximum over Active Streams of (last amount / median of that stream's earlier amounts − 1), clipped at 0.
5. **`discretionary_share`:** the share of the Client's Active Streams whose family is streaming, music or gym.
6. **`recent_stops`:** the number of the Client's streams whose last payment fell within the 180 days before the Cutoff and which are no longer active at the Cutoff. It uses the existing stream table's definition of ended or inactive.

## Test

- **Base:** the logit of the hail mary's train-only P(none).
  - Train: 5-fold out-of-fold, the average of `artifacts/hailmary/rehearsal/oof_{v2,surv}.csv`.
  - Selection: the average of `artifacts/hailmary/rehearsal/target_{v2,surv}.csv`.
- **Model:** logistic regression, standardised inputs, L2 with C = 1. Base only, against base + 6. This keeps it interpretable: report each coefficient's sign against the sociological prediction.
- **Two samples, the same procedure:**
  - **train:** 2,000 Clients, train labels, repeated 5×5 stratified CV (seeds 0–4), with out-of-fold scores averaged over the repeats;
  - **selection:** the 700 Clients inside `data.training_run(with_selection=True)`, the same CV.
- **Measures, per sample:**
  - the `none` AUC of base and of base + 6;
  - a paired DeLong test of the difference;
  - each feature's standalone AUC;
  - the coefficients.
- **Gate:** the difference is at least +0.01 **and** its paired DeLong 95% CI excludes 0 **on both train and selection** → recommend a build as a new ticket. Otherwise, the sociological route is closed in writing.
- **Also report, label-free:** each feature's train-vs-test adversarial AUC.

## Rules

- The sealed holdout is never read.
- LightGBM is not needed.
- Scripts go in `experiments/analysis/sociology/`, with caches git-ignored.

## Result

(to fill)
