"""Robustness of the hidden-series marker to the valid/test Filler Description shift, simulated on train only:
replace the description of a random share q of stream member payments by a Filler Description (keeping MCC and
amount), re-detect, and re-measure the marker. q=0.33 brings the live-stream family-description share from ~0.69
to ~0.46, the level seen in the cached valid stream table. Also tests an 'off-stream' variant of the marker that
ignores orphan clusters whose amount fits one of the Client's streams."""
import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from recurring_family.config import CUTOFF
from recurring_family.streams import _detect_per_client, StreamParams, _evidence
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); lab = D["labels"]; tx = D["tx"]; members = D["members"]
FILLERS = np.array(["member plan", "subscription charge", "digital service"])
TOL = 0.06
def marker(t):
    rows, fams = [], []
    for client, pays, srows, membership, _ in _detect_per_client(t, CUTOFF, StreamParams()):
        live = [r for r in srows if r["active"]]
        fams.extend(r["family_description_share"] for r in live)
        kinds = np.array([_evidence(d, m)[0] for d, m in zip(pays.description, pays.mcc)])
        o = pays[(kinds != "drop") & (membership < 0)]
        la_s = [(np.log(r["median_amount"])) for r in srows if r["n_payments"] >= 2]
        mx = n2 = mx_off = n2_off = mx_sc = n2_sc = 0
        if len(o):
            order = np.argsort(np.log(o.amount.to_numpy())); la = np.log(o.amount.to_numpy())[order]
            mccs = o.mcc.astype(str).to_numpy()[order]
            br = np.concatenate([[0], np.cumsum(np.diff(la) > TOL)])
            for k in np.unique(br):
                cl = la[br == k]; sz = len(cl)
                mx = max(mx, sz); n2 += sz >= 2
                top = pd.Series(mccs[br == k]).value_counts().iloc[0] / sz
                if top < 0.75:  # scattered MCCs: no single MCC carries 3/4 of the cluster
                    mx_sc = max(mx_sc, sz); n2_sc += sz >= 2
                if not any(abs(np.median(cl) - s) <= 2 * TOL for s in la_s):
                    mx_off = max(mx_off, sz); n2_off += sz >= 2
        rows.append(dict(client_id=client, n_live=len(live), hidden_max_n=mx, hidden_n2=n2, off_max_n=mx_off, off_n2=n2_off, sc_max_n=mx_sc, sc_n2=n2_sc))
    M = pd.DataFrame(rows).set_index("client_id").reindex(lab.index).fillna(0)
    M["none"] = (lab == "none").astype(int)
    return M, float(np.mean(fams))
res = []; saved = {}
stream_idx = np.concatenate([m["idx"] for m in members])
for q in [0.0, 0.33, 0.5]:
    t = tx.copy()
    rng = np.random.default_rng(0)
    pick = stream_idx[rng.random(len(stream_idx)) < q]
    t.loc[pick, "description"] = rng.choice(FILLERS, len(pick))
    M, fds = marker(t)
    saved[q] = M
    L = M[M.n_live > 0]
    for name, col, thr in [("hidden_n2>=1", "hidden_n2", 1), ("hidden_max_n>=3", "hidden_max_n", 3),
                           ("off-stream n2>=1", "off_n2", 1), ("off-stream max_n>=3", "off_max_n", 3),
                           ("scattered-MCC n2>=1", "sc_n2", 1), ("scattered-MCC max_n>=3", "sc_max_n", 3)]:
        flag = L[col] >= thr
        res.append(dict(q=q, live_fam_desc_share=fds, marker=name, live_clients=len(L), flagged=int(flag.sum()),
                        precision=L.none[flag].mean(), recall_live_none=flag[L.none == 1].mean(),
                        false_flags_family=int((flag & (L.none == 0)).sum()), auc_count=roc_auc_score(L.none, L[col])))
    print(pd.DataFrame(res).tail(6).round(3).to_string(index=False), flush=True)
pickle.dump(saved, open(OUT / "simulated_markers.pkl", "wb"))
R = pd.DataFrame(res); R.to_csv(OUT / "simulate_fillers.csv", index=False)
print(R.round(3).to_string(index=False))
