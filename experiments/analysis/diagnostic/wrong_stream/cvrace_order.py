import numpy as np, pandas as pd
import cvrace as cr
d=cr.d
# last + fixed 30.44 (rolled); singles: monthly slot from their one payment
d["o_fixed"]=np.where(d.n_payments>=2, d.p_3044, (-d.days_since_last)%30.44)
d["o_fixed_single_last"]=d.p_3044
d["o_last31"]=((-d.days_since_last)%31)
d["o_orig_single_slot"]=np.where(d.n_payments>=2, d.days_to_next, (-d.days_since_last)%30.44)
d["o_callast"]=np.where(d.n_payments>=2, d.p_callast, (-d.days_since_last)%30.44)
for o in ["o_fixed","o_fixed_single_last","o_last31","o_orig_single_slot","o_callast"]:
    cr.run(cr.BASE,o,tag=f"base, order {o}")
cr.run(cr.BASE+["n_ns_sub","n_ns_sub_90"],"o_fixed",tag="+hidden series, order o_fixed")
