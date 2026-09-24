"""Word vocabulary over all train and unlabeled transaction descriptions: is there any cancellation-like word?"""
import re, collections, pickle
from pathlib import Path
from recurring_family import data
D=pickle.load(open(Path(__file__).parent/"prep.pkl","rb"))
tx=D["tx"]
c=collections.Counter()
for d in tx.description.astype(str): c.update(re.findall(r"[a-z0-9]+",d.lower()))
print(len(c)); print(sorted(c.items(), key=lambda kv: -kv[1]))
u=data.load_transactions(Path("data/raw"),"unlabeled")
cu=collections.Counter()
for d in u.description.astype(str): cu.update(re.findall(r"[a-z0-9]+",d.lower()))
print("unlabeled-only words:", {w:n for w,n in cu.items() if w not in c})
print("types", tx.groupby(["type","direction"]).size().to_dict())
print("currencies", tx.currency.value_counts().to_dict(), "fee>0", (tx.fee>0).mean())
