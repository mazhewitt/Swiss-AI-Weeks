import pickle, numpy as np, pandas as pd
from pathlib import Path
D=pickle.load(open(Path(__file__).parent/"prep.pkl","rb"))
tx=D["tx"]
pd.set_option("display.max_rows",400,"display.width",200)
g=tx.groupby(["type","direction","description"]).size().sort_values(ascending=False)
print(len(g)); print(g.to_string())
