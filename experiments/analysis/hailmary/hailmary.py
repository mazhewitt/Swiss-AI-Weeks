"""Ticket 14, the hail mary: the average of v2 and the Survival Race, label-shift EM and self-training on the
target Clients' transactions (never their labels).

Two domains:

- `rehearsal`: fitted on train only; the target is the 700 selection Clients (valid transactions of the
  selection set only, never a sealed-holdout Client's). Scored on the selection labels in `score`, the only
  stage that reads them, inside `data.scoring()` and after every prediction was made.
- `final`: fitted on train plus selection (as the day-2 refit scripts do); the target is the 1,000 test Clients.

Stages (each caches under artifacts/hailmary/<domain>/, so they run as separate processes in parallel):

    oof <domain> <v2|surv>        5-fold out-of-fold probabilities on the domain's labelled Clients (`rf cv` folds)
    fit <domain> <v2|surv>        fit on the labelled Clients, predict the target Clients
    selftrain <domain> <v2|surv|both> <q>
                                  self-training, cross-fitted over two fixed halves of the target
    score                         rehearsal only: score every configuration, apply the ticket's upload rule
    submit                        final only: write submissions/day2_final_hailmary.csv for the chosen configuration

    uv run python experiments/analysis/hailmary/hailmary.py <stage> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recurring_family import data
from recurring_family import decision as decision_layer
from recurring_family.cli import Paths, _pseudo_training, _training_data
from recurring_family.config import LABEL_COLUMN, LABELS, SHIFTED_CUTOFF
from recurring_family.cross_validation import out_of_fold_proba
from recurring_family.evaluation import paired_bootstrap, score
from recurring_family.none_model import NoneModel
from recurring_family.ranker import RankerModel
from recurring_family.submission import write_submission
from recurring_family.survival import SurvivalModel

ROOT = Path(".")
PATHS = Paths(ROOT)
OUT = ROOT / "experiments" / "analysis" / "hailmary"
DOMAINS = ("rehearsal", "final")
QS = (0.3, 0.6, 1.0)
REFITS = ("surv", "both")
SEED = 0
V2_SELECTION_RUN = "20260925T001755-e309a3"  # 0.5871
SURVIVAL_SELECTION_RUN = "20260925T080905-fd1dc5"  # 0.5903
V2_SELECTION = 0.5871
SUBMISSION = "day2_final_hailmary"


def cache(domain: str) -> Path:
    path = PATHS.artifacts / "hailmary" / domain
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- data ---------------------------------------------------------------------------------------------


def labelled(domain: str) -> tuple[pd.DataFrame, pd.Series]:
    """The domain's labelled Clients: train (rehearsal) or train plus selection (final)."""
    with data.training_run(with_selection=domain == "final"):
        return _training_data(PATHS, with_selection=domain == "final")


def target(domain: str) -> tuple[pd.DataFrame, pd.Index]:
    """The target Clients' transactions (no labels): selection (rehearsal) or test (final)."""
    if domain == "rehearsal":
        ids = data.valid_split(PATHS.raw).query("set == 'selection'")["client_id"]
        tx = data.load_transactions(PATHS.raw, "valid")
        tx = tx[tx["client_id"].isin(set(ids))].reset_index(drop=True)  # no sealed-holdout Client's history
    else:
        ids = data.load_sample_submission(PATHS.raw)["client_id"]
        tx = data.load_transactions(PATHS.raw, "test")
    return tx, pd.Index(ids.astype(str), name="client_id")


# --- models -------------------------------------------------------------------------------------------

_PSEUDO = None


def v2_factory():
    """v2 exactly as scripts/day2_final_ranker_none_v2.sh trains it: the ranker with Pseudo-Labels (train and
    unlabeled at the Shifted Cutoff, weight 0.5, min_payments 4) and its Client-level none model."""
    global _PSEUDO
    if _PSEUDO is None:
        settings = argparse.Namespace(
            model="ranker", pseudo=[("train", SHIFTED_CUTOFF), ("unlabeled", SHIFTED_CUTOFF)], pseudo_weight=0.5,
            pseudo_min_payments=4,
        )
        _PSEUDO = _pseudo_training(settings, PATHS)
    pseudo, info = _PSEUDO
    return lambda: RankerModel(pseudo=pseudo, pseudo_weight=info["weight"], none_model=NoneModel())


def surv_factory():
    """The Survival Race as scripts/day2_final_survival.sh trains it."""
    return lambda: SurvivalModel(order="unprojected-last")


FACTORIES = {"v2": v2_factory, "surv": surv_factory}


# --- stages -------------------------------------------------------------------------------------------


def stage_oof(domain: str, model: str) -> None:
    make = FACTORIES[model]()
    tx, labels = labelled(domain)
    with data.training_run(with_selection=domain == "final"):
        oof = out_of_fold_proba(make, tx, labels)
    oof.to_csv(cache(domain) / f"oof_{model}.csv")
    print(f"oof {domain} {model}: {len(oof)} Clients", flush=True)


def stage_fit(domain: str, model: str) -> None:
    make = FACTORIES[model]()
    tx, labels = labelled(domain)
    with data.training_run(with_selection=domain == "final"):
        fitted = make().fit(tx, labels)
    target_tx, clients = target(domain)
    with data.predicting():
        proba = fitted.predict_proba(target_tx, clients)[list(LABELS)]
    proba.to_csv(cache(domain) / f"target_{model}.csv", index_label="client_id")
    print(f"fit {domain} {model}: predicted {len(proba)} target Clients", flush=True)


def read_proba(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    return df[list(LABELS)]


def decision_and_prior(domain: str) -> tuple[decision_layer.Decision, pd.Series]:
    """The E3 decision fitted on the averaged out-of-fold probabilities, and their mean (EM's source prior).
    Reads the domain's labelled Clients' labels only (train, or train plus selection in `final`)."""
    oof = (read_proba(cache(domain) / "oof_v2.csv") + read_proba(cache(domain) / "oof_surv.csv")) / 2
    _, labels = labelled(domain)
    labels = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    decision, _ = decision_layer.fit(oof, labels.reindex(oof.index))
    return decision, oof.mean()


def label_shift_em(proba: pd.DataFrame, prior: pd.Series, tol: float = 1e-6, iters: int = 1000) -> pd.DataFrame:
    """Saerens et al. (2002): re-estimate the target's label mix from the model's own probabilities and
    reweight each row by target prior / source prior."""
    prior = prior[list(LABELS)]
    mix = prior.copy()
    for _ in range(iters):
        adjusted = proba * (mix / prior)
        adjusted = adjusted.div(adjusted.sum(axis=1), axis=0)
        new = adjusted.mean()
        done = float((new - mix).abs().max()) < tol
        mix = new
        if done:
            break
    adjusted = proba * (mix / prior)
    return adjusted.div(adjusted.sum(axis=1), axis=0)


def averaged(domain: str, v2: pd.DataFrame | None = None, surv: pd.DataFrame | None = None) -> pd.DataFrame:
    v2 = read_proba(cache(domain) / "target_v2.csv") if v2 is None else v2
    surv = read_proba(cache(domain) / "target_surv.csv") if surv is None else surv
    return (v2 + surv.reindex(v2.index)) / 2


def em_rule() -> bool:
    """Rule 1, decided in the rehearsal's `score` stage."""
    path = OUT / "results.json"
    if not path.exists():
        raise SystemExit("run `score` on the rehearsal first: it decides whether EM is used")
    return bool(json.loads(path.read_text())["em_used"])


def self_labels(proba: pd.DataFrame, decision: decision_layer.Decision, q: float) -> pd.Series:
    """The top q of target Clients by the probability of their decided label, labelled with it."""
    decided = decision.apply(proba)
    confidence = pd.Series(
        proba.to_numpy()[np.arange(len(proba)), [list(LABELS).index(l) for l in decided]], index=proba.index
    )
    order = confidence.sort_index().sort_values(ascending=False, kind="stable")
    keep = order.index[: int(round(q * len(order)))]
    return decided.loc[keep].astype(str)


def halves(clients: pd.Index) -> list[pd.Index]:
    ids = np.array(sorted(map(str, clients)), dtype=object)
    perm = np.random.default_rng(SEED).permutation(ids)
    mid = len(perm) // 2
    return [pd.Index(sorted(perm[:mid]), name="client_id"), pd.Index(sorted(perm[mid:]), name="client_id")]


def stage_selftrain(domain: str, refit: str, q: float, use_em: bool | None = None) -> None:
    use_em = em_rule() if use_em is None else use_em
    decision, prior = decision_and_prior(domain)
    base = averaged(domain)
    start = label_shift_em(base, prior) if use_em else base
    pseudo = self_labels(start, decision, q)
    tx, labels = labelled(domain)
    target_tx, clients = target(domain)
    models = ["v2", "surv"] if refit == "both" else [refit]
    out = {m: [] for m in models}
    for h, half in enumerate(halves(clients)):
        other = pseudo[~pseudo.index.isin(set(half))]
        extra_tx = target_tx[target_tx["client_id"].astype(str).isin(set(other.index))].copy()
        extra_tx["client_id"] = ("self:" + extra_tx["client_id"].astype(str)).astype("string")
        extra = pd.Series(other.to_numpy(), index=pd.Index("self:" + other.index, name="client_id"))
        fit_tx = pd.concat([tx, extra_tx], ignore_index=True)
        fit_labels = pd.concat([labels.astype(str), extra])
        for m in models:
            with data.training_run(with_selection=domain == "final"):
                fitted = FACTORIES[m]()().fit(fit_tx, fit_labels)
            with data.predicting():
                proba = fitted.predict_proba(target_tx[target_tx["client_id"].astype(str).isin(set(half))], half)
            out[m].append(proba[list(LABELS)])
            print(f"selftrain {domain} {refit} q={q}: half {h}, {m} fitted with {len(extra)} self-labels", flush=True)
    for m in models:
        pd.concat(out[m]).reindex(clients).to_csv(
            cache(domain) / f"st_{refit}_q{q}_{m}{'_em' if use_em else ''}.csv", index_label="client_id"
        )


def configuration(domain: str, name: str, use_em: bool) -> pd.DataFrame:
    """The final probabilities of one configuration: `A`, or `st_<refit>_q<q>`."""
    decision, prior = decision_and_prior(domain)
    if name == "A":
        proba = averaged(domain)
    else:
        _, refit, qtext = name.split("_")
        tag = "_em" if use_em else ""
        c = cache(domain)
        surv = read_proba(c / f"{name}_surv{tag}.csv")
        v2 = read_proba(c / f"{name}_v2{tag}.csv") if refit == "both" else None
        proba = averaged(domain, v2, surv)
    return label_shift_em(proba, prior) if use_em else proba


def stage_score() -> None:
    domain = "rehearsal"
    decision, prior = decision_and_prior(domain)
    base = averaged(domain)
    single = {m: read_proba(cache(domain) / f"target_{m}.csv") for m in ("v2", "surv")}
    single_decisions = {m: decision_layer.fit(read_proba(cache(domain) / f"oof_{m}.csv"), _labels(domain))[0] for m in single}
    predictions = {
        "A": decision.apply(base),
        "A+EM": decision.apply(label_shift_em(base, prior)),
        **{f"{m} alone": single_decisions[m].apply(p) for m, p in single.items()},
    }
    # the self-training configurations under both EM tags, wherever their files exist
    missing = {False: [], True: []}
    for use_em in (False, True):
        for refit in REFITS:
            for q in QS:
                name = f"st_{refit}_q{q}"
                if not (cache(domain) / f"{name}_surv{'_em' if use_em else ''}.csv").exists():
                    missing[use_em].append(name)
                    continue
                predictions[name + ("+EM" if use_em else "")] = decision.apply(configuration(domain, name, use_em))
    for name, p in predictions.items():
        assert p.notna().all() and len(p) == 700, f"{name}: incomplete predictions"
    # every prediction is made; only now read the selection labels
    with data.scoring():
        truth = data.load_labels(PATHS.raw, "selection").set_index("client_id")[LABEL_COLUMN]
    truth.index = truth.index.astype(str)
    runs = {
        name: pd.read_csv(PATHS.run_predictions(run), dtype=str, keep_default_na=False).set_index("client_id")["predicted"]
        .reindex(truth.index)
        for name, run in (("v2", V2_SELECTION_RUN), ("surv", SURVIVAL_SELECTION_RUN))
    }
    assert all(r.notna().all() for r in runs.values())
    raw: dict[str, float] = {}

    def row(name: str, predicted: pd.Series) -> dict:
        predicted = predicted.reindex(truth.index)
        s = score(truth, predicted)
        raw[name] = s["macro_f1"]
        cmp = paired_bootstrap(truth, predicted, runs["v2"])
        (OUT / "predictions").mkdir(parents=True, exist_ok=True)
        predicted.rename("predicted").to_csv(
            OUT / "predictions" / f"{name.replace('+', '_').replace(' ', '_')}.csv", index_label="client_id"
        )
        return {
            "config": name, "macro_f1": round(s["macro_f1"], 4), "vs_v2_delta": round(cmp["delta"], 4),
            "vs_v2_low": round(cmp["low"], 4), "vs_v2_high": round(cmp["high"], 4),
            "agree_v2": round(float((predicted == runs["v2"]).mean()), 3),
            **{f"f1_{l}": round(s[f"f1_{l}"], 4) for l in LABELS},
        }

    rows = {name: row(name, p) for name, p in predictions.items()}
    # checks: each single model reproduces its logged selection run exactly
    for m in single:
        same = float((predictions[f"{m} alone"].reindex(truth.index) == runs[m]).mean())
        rows[f"{m} alone"]["agree_logged_run"] = same
        assert same == 1.0, f"{m} alone does not reproduce its logged run ({same:.3f} agreement)"
    eps = 1e-9
    use_em = raw["A+EM"] - raw["A"] >= 0.01 - eps  # rule 1
    tag = "+EM" if use_em else ""
    st = [n for n in rows if n.startswith("st_") and n.endswith("+EM") == use_em]
    base_name = "A" + tag
    if missing[use_em]:
        chosen, why = None, f"incomplete: missing self-training runs {missing[use_em]}; rules 2-3 not applied"
    else:
        best = max(st, key=lambda n: raw[n])
        if raw[best] >= V2_SELECTION - 0.01 - eps:
            chosen, why = best, "rule 2: the best self-training configuration, not clearly worse than v2 (optimistic: best of 6)"
        elif raw[base_name] >= V2_SELECTION - eps:
            chosen, why = base_name, "rule 3: no self-training configuration qualified; the average does"
        else:
            chosen, why = None, "rule 4: nothing qualified; the user decides"
    versus_a = None
    if chosen is not None and chosen != base_name:
        c = paired_bootstrap(truth, predictions[chosen].reindex(truth.index), predictions[base_name].reindex(truth.index))
        versus_a = {k: round(v, 4) for k, v in c.items()}
    result = {
        "em_used": use_em, "complete": not missing[use_em], "chosen": chosen, "why": why,
        "chosen_vs_average": versus_a, "rows": list(rows.values()),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    for r in rows.values():
        print(f"{r['config']:<22} {r['macro_f1']:.4f}  vs v2 {r['vs_v2_delta']:+.4f} ({r['vs_v2_low']:+.4f} .. {r['vs_v2_high']:+.4f})  agree {r['agree_v2']}")


def _labels(domain: str) -> pd.Series:
    _, labels = labelled(domain)
    return pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))


def stage_submit() -> None:
    result = json.loads((OUT / "results.json").read_text())
    chosen = result["chosen"]
    if not result.get("complete") or chosen is None:
        raise SystemExit("the rehearsal chose nothing (rule 4): the user decides")
    use_em = result["em_used"]
    name = chosen.removesuffix("+EM")
    domain = "final"
    decision, _ = decision_and_prior(domain)
    proba = configuration(domain, name, use_em)
    predicted = decision.apply(proba)
    expected = data.load_sample_submission(PATHS.raw)["client_id"]
    out = PATHS.submissions / f"{SUBMISSION}.csv"
    write_submission(predicted, expected, out)
    proba.to_csv(cache(domain) / "submitted_proba.csv", index_label="client_id")
    print(f"submit: {chosen} -> {out} ({len(expected)} Clients); label mix:")
    print(predicted.value_counts().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("oof", "fit"):
        s = sub.add_parser(stage)
        s.add_argument("domain", choices=DOMAINS)
        s.add_argument("model", choices=sorted(FACTORIES))
    s = sub.add_parser("selftrain")
    s.add_argument("domain", choices=DOMAINS)
    s.add_argument("refit", choices=REFITS)
    s.add_argument("q", type=float, choices=QS)
    s.add_argument("--em", choices=("rule", "on", "off"), default="rule")
    sub.add_parser("score")
    sub.add_parser("submit")
    a = p.parse_args()
    if a.stage == "oof":
        stage_oof(a.domain, a.model)
    elif a.stage == "fit":
        stage_fit(a.domain, a.model)
    elif a.stage == "selftrain":
        stage_selftrain(a.domain, a.refit, a.q, None if a.em == "rule" else a.em == "on")
    elif a.stage == "score":
        stage_score()
    else:
        stage_submit()


if __name__ == "__main__":
    main()
