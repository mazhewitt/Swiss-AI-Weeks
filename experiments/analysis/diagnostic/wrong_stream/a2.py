import pandas as pd, numpy as np
C="cache/"
paid=pd.read_pickle(C+"paid.pkl"); st=pd.read_pickle(C+"streams.pkl")
p=paid.merge(st[["client_id","stream_id","family","period_days","n_payments","active"]],on=["client_id","stream_id"])
p["dom"]=p.timestamp.dt.day; p["wd"]=p.timestamp.dt.weekday; p["hour"]=p.timestamp.dt.hour
print("period dist:\n",st.period_days.round().value_counts().head(10))
print("weekday dist:",p.wd.value_counts(normalize=True).sort_index().round(3).tolist())
print("hour dist:",p.hour.value_counts(normalize=True).sort_index().round(3).tolist())
m=p[(p.period_days.between(26,35))&(p.n_payments>=3)]
g=m.groupby(["client_id","stream_id"])
dstd=g.dom.agg(lambda d: np.median(np.abs(d-np.median(d))))
print("monthly streams:",len(dstd)," DOM MAD dist:",dstd.describe().round(2).to_dict())
print("share with DOM MAD<=1:",(dstd<=1).mean().round(3),"<=3:",(dstd<=3).mean().round(3))
# gaps
m=m.sort_values(["client_id","stream_id","timestamp"])
m["gap"]=m.groupby(["client_id","stream_id"]).timestamp.diff().dt.total_seconds()/86400
print("monthly gap quantiles:",m.gap.quantile([.05,.1,.25,.5,.75,.9,.95]).round(2).to_dict())
print("gap value counts:",m.gap.round().value_counts().head(15).to_dict())
# gap vs calendar month length
m["prevmonthlen"]=(m.timestamp - m.gap*pd.Timedelta(days=1)).dt.days_in_month
print(m.groupby("prevmonthlen").gap.median())
# dom change: same dom as prev?
m["pdom"]=m.groupby(["client_id","stream_id"]).dom.shift()
print("same DOM as prev:",(m.dom==m.pdom).mean().round(3), " |diff|<=1:",((m.dom-m.pdom).abs()<=1).mean().round(3))
# per-stream jitter: gap residual vs 30.44
r=(m.gap-30.44)
print("gap-30.44 abs quantiles",r.abs().quantile([.25,.5,.75,.9]).round(2).to_dict())
# autocorrelation of residuals within stream (random walk vs anchored)
m["r"]=m.gap-m.groupby(["client_id","stream_id"]).gap.transform("median")
m["rprev"]=m.groupby(["client_id","stream_id"]).r.shift()
print("corr successive gap residuals", m[["r","rprev"]].dropna().corr().iloc[0,1].round(3))
print("period by family:\n",st[st.n_payments>=3].groupby("family").period_days.describe()[["count","mean","50%","min","max"]].round(1))
