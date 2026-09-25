"""Upper bound: macro-F1 if every never-detected label were predicted right (S-full OOF, in-sample tuned none threshold)."""
from lib import *
from sklearn.metrics import f1_score
_, y, st = load(); LAB = list(FAMS) + ["none"]
fams = st.groupby("client_id")["family"].agg(set)
und = [c for c in y.index if y[c] != "none" and y[c] not in fams.get(c, set())]
P = pd.read_csv("../../../analysis/survival/oof_S-full.csv", index_col=0).loc[y.index]
best = max(((f1_score(y, np.where(P["none"] > t, "none", P[list(FAMS)].idxmax(axis=1)), labels=LAB, average="macro"), t) for t in np.linspace(.05, .95, 91)))
pred = pd.Series(np.where(P["none"] > best[1], "none", P[list(FAMS)].idxmax(axis=1)), index=y.index)
print("base", round(best[0], 4), "undetected predicted as:", pred[und].value_counts().to_dict())
fix = pred.copy(); fix[und] = y[und]
print("ceiling all undetected fixed", round(f1_score(y, fix, labels=LAB, average="macro"), 4))
fix2 = pred.copy(); fix2[[c for c in und if pred[c] != "none"]] = "none"
print("undetected wrong-family preds -> none", round(f1_score(y, fix2, labels=LAB, average="macro"), 4))
