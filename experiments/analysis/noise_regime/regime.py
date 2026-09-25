"""Ticket 23, step 2: recreate test's description and MCC noise in train, and see what it costs.

The corruption parameters are `noise_params.json`, fixed by `measure.py` (step 1) before this script ran.

Stages (each writes under `cache/`, git-ignored, so they run as separate processes in parallel):

    corrupt           train's transactions at doses 0, 0.5 and 1 (seed 0), plus what changed
    oof <dose> <v2|surv>
                      hail mary configuration A's 5-fold out-of-fold probabilities (`rf cv` folds) on the
                      corrupted train: v2 (ranker + none model + Pseudo-Labels) and the hard Survival Race
    score             nested tuned E3 per dose on the average of v2 and the race (ticket 17's Gate A)
    drift             train-vs-test adversarial AUC of the v2 none model's stream features, per dose
    closure           step 1's measurement on the corrupted train: does 1x look like test?

Rules: train labels only (read inside `data.training_run()`); valid and test labels are never read. Test is
read as transactions only, through `data.load_transactions`.

**Corruption.** The payments of every stream of two or more payments that the default detector finds in
train's whole history (Cutoff 2026-01-01) are the stream payments. Each gets four uniform draws from one
seed-0 generator, in transaction order, whatever the dose: u_desc, u_desc_pick, u_mcc, u_mcc_pick. At dose
d its description is replaced when u_desc < d * rate[F].description, by the draw of u_desc_pick from
draws[F].description (inverse CDF); its MCC likewise. So the 0.5x corruption is a subset of the 1x one.
Amounts, dates, labels and every other transaction are unchanged.

**Pseudo-Labels.** v2 pools Pseudo-Labelled rows of train and unlabeled at the Shifted Cutoff (2025-10-03,
min_payments 4, weight 0.5). The train part is made from the corrupted train (detected afresh from the
DataFrame, never from the on-disk caches, which are keyed on the raw file's size and mtime and would serve
clean tables); the unlabeled part is left clean and comes from its cache. The Survival Race uses no
Pseudo-Labels.

    uv run python experiments/analysis/noise_regime/regime.py <stage> ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recurring_family import data  # noqa: E402
from recurring_family import decision as decision_layer  # noqa: E402
from recurring_family.cli import _labeller, _pseudo_training  # noqa: E402
from recurring_family.config import CUTOFF, LABEL_COLUMN, LABELS, SHIFTED_CUTOFF  # noqa: E402
from recurring_family.cross_validation import out_of_fold_proba  # noqa: E402
from recurring_family.evaluation import paired_bootstrap, score  # noqa: E402
from recurring_family.none_model import GROUP_A, GROUP_B, NoneModel, client_features  # noqa: E402
from recurring_family.ranker import pseudo_examples, RankerModel  # noqa: E402
from recurring_family.streams import StreamParams, _detect_per_client, detect_stream_payments, detect_streams, pseudo_label_sweep  # noqa: E402
from recurring_family.survival import SurvivalModel  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hailmary", Path(__file__).resolve().parent.parent / "hailmary" / "hailmary.py"
)
hm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hm)

RAW = Path("data/raw")
OUT = Path("experiments/analysis/noise_regime")
CACHE = OUT / "cache"
DOSES = ("0", "0.5", "1")
SEED = 0
PSEUDO_MIN_PAYMENTS = 4
PSEUDO_WEIGHT = 0.5


def update_results(key: str, value) -> None:
    path = OUT / "results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[key] = value
    path.write_text(json.dumps(result, indent=1))


def tx_path(dose: str) -> Path:
    return CACHE / f"train_dose{dose}.pkl"


def content_hash(tx: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(tx[sorted(tx.columns)], index=False).to_numpy().tobytes()).hexdigest()[:12]


# --- corruption ---------------------------------------------------------------------------------------


def _pick(u: float, dist: dict) -> str:
    keys = list(dist)
    cdf = np.cumsum([dist[k] for k in keys])
    return keys[min(int(np.searchsorted(cdf / cdf[-1], u, side="right")), len(keys) - 1)]


def stage_corrupt() -> None:
    params = json.loads((OUT / "noise_params.json").read_text())
    clean = data.load_transactions(RAW, "train")
    rows, fams = [], []
    for _, pay, srows, membership, _ in _detect_per_client(clean, CUTOFF, StreamParams()):
        family = {r["stream_id"]: r["family"] for r in srows if r["n_payments"] >= 2}
        for idx, m in zip(pay.index, membership):
            if m in family:
                rows.append(idx)
                fams.append(family[m])
    order = np.argsort(rows, kind="stable")
    rows, fams = np.asarray(rows)[order], np.asarray(fams, dtype=object)[order]
    rng = np.random.default_rng(SEED)
    u = rng.random((len(rows), 4))  # u_desc, u_desc_pick, u_mcc, u_mcc_pick
    desc_draw = np.array([_pick(p, params["draws"][f]["description"]) for p, f in zip(u[:, 1], fams)], dtype=object)
    mcc_draw = np.array([_pick(p, params["draws"][f]["mcc"]) for p, f in zip(u[:, 3], fams)], dtype=object)
    rate_d = np.array([params["rates"][f]["description"] for f in fams])
    rate_m = np.array([params["rates"][f]["mcc"] for f in fams])
    CACHE.mkdir(parents=True, exist_ok=True)
    info = {"stream_payments": int(len(rows)), "train_transactions": int(len(clean)), "doses": {}}
    for dose in DOSES:
        d = float(dose)
        tx = clean.copy()
        hit_d, hit_m = u[:, 0] < d * rate_d, u[:, 2] < d * rate_m
        tx.loc[rows[hit_d], "description"] = desc_draw[hit_d]
        tx.loc[rows[hit_m], "mcc"] = mcc_draw[hit_m]
        tx = tx.astype(data.TRANSACTION_DTYPES)
        changed_d = (tx["description"] != clean["description"]).sum()
        changed_m = (tx["mcc"] != clean["mcc"]).sum()
        tx.to_pickle(tx_path(dose))
        info["doses"][dose] = {
            "description_replaced": int(hit_d.sum()), "mcc_replaced": int(hit_m.sum()),
            "description_changed": int(changed_d), "mcc_changed": int(changed_m),
            "content_hash": content_hash(tx),
        }
        print(f"dose {dose}: {info['doses'][dose]}", flush=True)
    assert info["doses"]["0"]["content_hash"] == content_hash(clean)
    update_results("corruption", info)


# --- configuration A on the corrupted train -----------------------------------------------------------


def load_tx(dose: str) -> pd.DataFrame:
    return pd.read_pickle(tx_path(dose))


def train_labels() -> pd.Series:
    with data.training_run():
        labels = data.load_labels(RAW, "train").set_index("client_id")[LABEL_COLUMN]
    return labels


def pseudo_rows(tx: pd.DataFrame) -> pd.DataFrame:
    """v2's Pseudo-Labelled rows: train's from `tx` (the corrupted train, detected afresh), unlabeled's clean
    from its cache, in the order `rf` pools them (train, then unlabeled)."""
    labeller = _labeller(PSEUDO_MIN_PAYMENTS, None, None)
    with data.pseudo_labelling():
        labels = pseudo_label_sweep(tx, (labeller,), SHIFTED_CUTOFF)[labeller]
        streams = detect_streams(tx, cutoff=SHIFTED_CUTOFF)
    train = pseudo_examples(streams, labels, SHIFTED_CUTOFF)
    train["client_id"] = f"train@{SHIFTED_CUTOFF:%Y-%m-%d}:" + train["client_id"].astype(str)
    settings = argparse.Namespace(
        model="ranker", pseudo=[("unlabeled", SHIFTED_CUTOFF)], pseudo_weight=PSEUDO_WEIGHT,
        pseudo_min_payments=PSEUDO_MIN_PAYMENTS,
    )
    unlabeled, info = _pseudo_training(settings, hm.PATHS)
    assert info["weight"] == PSEUDO_WEIGHT
    return pd.concat([train, unlabeled], ignore_index=True), labels


def stage_oof(dose: str, model: str) -> None:
    tx = load_tx(dose)
    labels = train_labels()
    extra = {}
    if model == "v2":
        pseudo, plabels = pseudo_rows(tx)
        if dose == "0":  # the bypass reproduces the pooled rows `rf` builds from its caches, exactly
            reference, _ = _pseudo_training(
                argparse.Namespace(model="ranker", pseudo=[("train", SHIFTED_CUTOFF), ("unlabeled", SHIFTED_CUTOFF)],
                                   pseudo_weight=PSEUDO_WEIGHT, pseudo_min_payments=PSEUDO_MIN_PAYMENTS), hm.PATHS)
            pd.testing.assert_frame_equal(pseudo.reset_index(drop=True), reference.reset_index(drop=True))
        extra = {
            "pseudo_rows": int(len(pseudo)), "train_pseudo_none_share": round(float((plabels == "none").mean()), 4),
            "train_pseudo_label_mix": plabels.value_counts().to_dict(),
        }
        make = lambda: RankerModel(pseudo=pseudo, pseudo_weight=PSEUDO_WEIGHT, none_model=NoneModel())
    else:
        make = lambda: SurvivalModel(order="unprojected-last")
    with data.training_run():
        oof = out_of_fold_proba(make, tx, labels)
    oof.to_csv(CACHE / f"oof_{model}_dose{dose}.csv")
    (CACHE / f"oof_{model}_dose{dose}.json").write_text(json.dumps({"content_hash": content_hash(tx), **extra}, indent=1))
    print(f"oof dose {dose} {model}: {len(oof)} Clients {extra}", flush=True)


# --- scoring ------------------------------------------------------------------------------------------


def read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    return df[["fold", *LABELS]]


def averaged_oof(v2_path: Path, surv_path: Path) -> pd.DataFrame:
    v2, surv = read_oof(v2_path), read_oof(surv_path)
    assert v2.index.equals(surv.index) and (v2["fold"] == surv["fold"]).all()
    out = (v2[list(LABELS)] + surv[list(LABELS)]) / 2
    out.insert(0, "fold", v2["fold"])
    return out


def nested_e3(oof: pd.DataFrame, labels: pd.Series, predictions: dict | None = None, name: str = "") -> dict:
    """Ticket 17's Gate A: tuned E3 fitted on four folds' out-of-fold probabilities, applied to the fifth."""
    nested = pd.Series(index=oof.index, dtype=object)
    folds = []
    for k in sorted(oof["fold"].unique()):
        held = oof[oof["fold"] == k][list(LABELS)]
        rest = oof[oof["fold"] != k][list(LABELS)]
        d = decision_layer.fit(rest, labels.reindex(rest.index))[0]
        nested.loc[held.index] = d.apply(held)
        folds.append(round(score(labels[held.index], nested[held.index])["macro_f1"], 4))
    s = score(labels, nested)
    if predictions is not None:
        predictions[name] = nested
    tuned = decision_layer.fit(oof[list(LABELS)], labels)[0].apply(oof[list(LABELS)])
    argmax = oof[list(LABELS)].idxmax(axis=1)
    return {
        "nested_e3_macro_f1": round(s["macro_f1"], 4), "folds": folds,
        "tuned_in_sample_macro_f1": round(score(labels, tuned)["macro_f1"], 4),
        "argmax_macro_f1": round(score(labels, argmax)["macro_f1"], 4),
        "f1": {l: round(s[f"f1_{l}"], 4) for l in LABELS},
        "none_share_predicted": round(float((nested == "none").mean()), 4),
    }


def stage_score() -> None:
    labels = train_labels()
    labels = pd.Series(labels.astype(str).to_numpy(), index=labels.index.astype(str))
    out, predictions = {}, {}
    reference = averaged_oof(hm.cache("rehearsal") / "oof_v2.csv", hm.cache("rehearsal") / "oof_surv.csv")
    out["hailmary_A_cached"] = nested_e3(reference, labels.reindex(reference.index))
    for dose in DOSES:
        oof = averaged_oof(CACHE / f"oof_v2_dose{dose}.csv", CACHE / f"oof_surv_dose{dose}.csv")
        out[dose] = nested_e3(oof, labels.reindex(oof.index), predictions, dose)
        for m in ("v2", "surv"):
            single = read_oof(CACHE / f"oof_{m}_dose{dose}.csv")
            out[dose][f"{m}_alone_nested_e3"] = nested_e3(single, labels.reindex(single.index))["nested_e3_macro_f1"]
        meta = json.loads((CACHE / f"oof_v2_dose{dose}.json").read_text())
        out[dose]["pseudo"] = {k: v for k, v in meta.items() if k != "content_hash"}
        print(dose, out[dose]["nested_e3_macro_f1"], out[dose]["folds"], flush=True)
    for m in ("v2", "surv"):  # dose 0 reproduces the hail mary's cached out-of-fold probabilities
        mine, cached = read_oof(CACHE / f"oof_{m}_dose0.csv"), read_oof(hm.cache("rehearsal") / f"oof_{m}.csv")
        out[f"dose0_{m}_max_abs_diff_vs_hailmary_cache"] = float((mine - cached.reindex(mine.index)).abs().max().max())
    base = out["0"]["nested_e3_macro_f1"]
    out["drop_vs_clean"] = {d: round(out[d]["nested_e3_macro_f1"] - base, 4) for d in DOSES}
    # paired bootstrap over train Clients (same folds, same Clients): dose against clean
    truth = labels.reindex(predictions["0"].index)
    out["paired_bootstrap_vs_clean"] = {
        d: {k: round(v, 4) for k, v in paired_bootstrap(truth, predictions[d], predictions["0"]).items()} for d in DOSES[1:]
    }
    update_results("step2_nested_e3", out)
    print(json.dumps(out["drop_vs_clean"]), json.dumps(out["paired_bootstrap_vs_clean"]))


# --- drift and closure --------------------------------------------------------------------------------


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, list[str]]:
    """Ticket 11's shift check: LightGBM train-vs-other classifier, 5-fold, 3 seeds, mean AUC."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = pd.concat([a, b]).astype(float)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    make = lambda: lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, verbose=-1, n_jobs=1)
    aucs = []
    for seed in range(3):
        p = cross_val_predict(make(), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    m = make().fit(x, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=x.columns).sort_values(ascending=False)
    return round(float(np.mean(aucs)), 4), list(imp.index[:5])


def stage_drift() -> None:
    test = data.load_transactions(RAW, "test")
    feats = {"test": client_features(*detect_stream_payments(test))}
    for dose in DOSES:
        feats[dose] = client_features(*detect_stream_payments(load_tx(dose)))
    out = {"method": "LightGBM train-vs-test, 5-fold x 3 seeds (ticket 11's shift11.py), Clients with a stream, v2 none model's stream features"}
    for dose in DOSES:
        row = {}
        for name, cols in (("A+B", GROUP_A + GROUP_B), ("group A", GROUP_A), ("group B", GROUP_B)):
            auc, top = adversarial_auc(feats[dose][cols], feats["test"][cols])
            row[name] = {"auc": auc, "top": top}
        row["means"] = {c: round(float(feats[dose][c].mean()), 4) for c in ("max_missed_rate", "n_short", "n_ended", "n_active_streams")}
        out[dose] = row
        print(dose, {k: v["auc"] for k, v in row.items() if k != "means"}, row["means"], flush=True)
    out["test_means"] = {c: round(float(feats["test"][c].mean()), 4) for c in ("max_missed_rate", "n_short", "n_ended", "n_active_streams")}
    update_results("step2_drift", out)


def stage_closure() -> None:
    from schedule import schedule_payments, summarise

    out = {}
    for dose in DOSES[1:]:
        tx = load_tx(dose)
        payments, streams, table = schedule_payments(tx)
        s = summarise(payments, streams, table, tx["client_id"].nunique())
        out[dose] = {"all": s["all"], "fragmentation": s["fragmentation"],
                     "mcc_noise_given_description": s["mcc_noise_given_description"],
                     "description_noise_dispersion": s["description_noise_dispersion"]}
        print(dose, s["all"]["description_noise"], s["all"]["mcc_noise"], s["fragmentation"], flush=True)
    update_results("step2_closure", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("corrupt")
    s = sub.add_parser("oof")
    s.add_argument("dose", choices=DOSES)
    s.add_argument("model", choices=("v2", "surv"))
    for stage in ("score", "drift", "closure"):
        sub.add_parser(stage)
    a = p.parse_args()
    if a.stage == "oof":
        stage_oof(a.dose, a.model)
    else:
        {"corrupt": stage_corrupt, "score": stage_score, "drift": stage_drift, "closure": stage_closure}[a.stage]()


if __name__ == "__main__":
    main()
