import pickle, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from recurring_family.streams import _evidence, _words, _FAMILY_WORDS
OUT = Path(__file__).parent
D = pickle.load(open(OUT / "prep.pkl", "rb")); tx, lab, st, pays = D["tx"], D["lab"], D["streams"], D["pays"]
y = lab.set_index("client_id")["target_next_recurring_merchant"].astype(str)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 50)
fams = st.groupby("client_id")["family"].agg(lambda s: set(map(str, s)))
det = y.index.map(lambda c: y[c] in fams.get(c, set()))
undet = y[(y != "none") & ~det]
print("clients", len(y), "none", (y=="none").mean(), "family labels", (y!="none").sum(), "undetected", len(undet), len(undet)/len(y))
print(undet.value_counts())
print(tx.type.value_counts(), tx.direction.value_counts(), tx.currency.value_counts())
print(tx.description.value_counts().to_string())
