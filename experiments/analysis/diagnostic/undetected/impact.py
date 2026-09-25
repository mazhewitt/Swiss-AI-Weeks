"""Macro-F1 impact of 'odd recent stream payments' as a none signal, on top of train OOF probabilities
of v2 and the survival race (experiments/analysis/survival/oof_*.csv). Nested 5-fold: stack + threshold tuned inside."""
from lib import *
from recurring_family.config import CUTOFF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
m = pd.read_pickle(OUT / "members.pkl"); _, y, st = load()
LAB = list(FAMS) + ["none"]
DAY = pd.Timedelta(days=1)
m["mb"] = ((CUTOFF - m.timestamp) / DAY // 30.44).astype(int)
rec, old = m[m.mb <= 5], m[m.mb >= 8]
f = pd.DataFrame(index=y.index)
f["rec_odd"] = rec.groupby("client_id").odd.mean()
f["rec_off"] = rec.groupby("client_id").off_home.mean()
f["rec_n_odd"] = rec.groupby("client_id").odd.sum()
f["old_odd"] = old.groupby("client_id").odd.mean()
f["rec_minus_old"] = f.rec_odd - f.old_odd.fillna(f.old_odd.mean())
f = f.fillna({"rec_n_odd": 0})
f.to_pickle(OUT / "odd_client_feats.pkl")
yn = (y == "none").astype(int)
has = f.rec_odd.notna()
for c in ["rec_odd", "rec_off", "rec_n_odd", "rec_minus_old"]:
    print(c, "AUC for none (clients with recent stream payments)", round(roc_auc_score(yn[has], f.loc[has, c].fillna(0)), 3))
# rule precision table
for t in (0.2, 0.3, 0.4, 0.5):
    s = has & (f.rec_odd >= t); print(f"rec_odd>={t}: clients={s.sum()} none_prec={yn[s].mean():.3f} share_of_none={yn[s].sum()/yn.sum():.3f}")
for k in (1, 2, 3, 4):
    s = f.rec_n_odd >= k; print(f"rec_n_odd>={k}: clients={s.sum()} none_prec={yn[s].mean():.3f}")
def mf1(p, pn, t):
    fam = p[list(FAMS)].idxmax(axis=1)
    pred = np.where(pn > t, "none", fam)
    return f1_score(y[p.index], pred, labels=LAB, average="macro")
rng = np.random.RandomState(0)
for name in ["oof_baseline_v2", "oof_S-full"]:
    P = pd.read_csv(f"../../../analysis/survival/{name}.csv", index_col=0).loc[y.index]
    folds = P.fold.to_numpy()
    base_pn = P["none"]
    X = pd.DataFrame({"lg": np.log(base_pn.clip(1e-4, 1 - 1e-4) / (1 - base_pn.clip(1e-4, 1 - 1e-4))),
                      "rec_odd": f.rec_odd.fillna(0), "rec_off": f.rec_off.fillna(0), "rec_n": np.log1p(f.rec_n_odd),
                      "has": has.astype(float), "rmo": f.rec_minus_old.fillna(0)})
    grid = np.linspace(0.05, 0.95, 91)
    out = {}
    for variant, cols in [("base", None), ("stack", list(X.columns)), ("contrast_only", ["lg", "rmo", "has"])]:
        preds, pn_all = pd.Series(index=y.index, dtype=object), pd.Series(index=y.index, dtype=float)
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            if cols is None:
                pn_tr, pn_te = base_pn[tr], base_pn[te]
            else:
                lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr][cols], yn[tr])
                # inner cross-fit for the training fold's stacked p_none
                pn_tr = pd.Series(np.nan, index=y.index[tr])
                inner = folds[tr]
                for j in np.unique(inner):
                    a, b = inner != j, inner == j
                    pn_tr[b] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr][cols][a], yn[tr][a]).predict_proba(X[tr][cols][b])[:, 1]
                pn_te = pd.Series(lr.predict_proba(X[te][cols])[:, 1], index=y.index[te])
            # renormalise family probs to share of (1-pn) is not needed for argmax
            t = max(grid, key=lambda t: mf1(P[tr], pn_tr, t))
            fam = P[te][list(FAMS)].idxmax(axis=1)
            preds[te] = np.where(pn_te > t, "none", fam); pn_all[te] = pn_te
        out[variant] = f1_score(y, preds, labels=LAB, average="macro")
        print(name, variant, "nested macro-F1", round(out[variant], 4), "none AUC", round(roc_auc_score(yn, pn_all), 3),
              "per-class F1", dict(zip(LAB, np.round(f1_score(y, preds, labels=LAB, average=None), 3))))
    # bootstrap-free paired delta sign by fold
    print(name, "delta", round(out["stack"] - out["base"], 4))
