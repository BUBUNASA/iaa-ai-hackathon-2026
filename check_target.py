import pandas as pd, numpy as np
from prep import *
pd.set_option("display.width", 250, "display.max_columns", 60)

claims = load_claims()
print("claims net_incurred<0:", (claims.net_incurred < 0).sum(), "| recoveries>incurred:", (claims.recoveries > claims.incurred_to_date).sum())
for ln in LINES:
    pol = load_policies(ln)
    df, un = attach_claims(pol, claims, ln)
    n_cl = (claims.line == ln).sum()
    print(f"\n=== {ln}: claims {n_cl}, unmatched {len(un)} ({len(un)/n_cl:.2%}), dup matches {df.attrs['duplicate_matches']}")
    if len(un):
        print(" unmatched loss years:", un.loss_date.dt.year.value_counts().sort_index().to_dict())
        print(" unmatched policy in policies:", un.policy_id.isin(pol.policy_id).mean().round(3))
    g = df.groupby("year").agg(expo=("exposure", "sum"), n=("claim_count", "sum"), cost=("net_incurred", "sum"),
                               open_n=("open_count", "sum"), prem=("written_premium", "sum"))
    g["freq"] = g.n / g.expo
    g["cost_per_expo"] = g.cost / g.expo
    g["sev"] = g.cost / g.n
    g["open_share"] = g.open_n / g.n
    g["LR_proxy"] = g.cost / g.prem
    print(g.round(3))
    r = df.net_incurred / df.exposure
    print(" zero-cost rows share:", (df.net_incurred == 0).mean().round(4),
          "| top-0.1% rows share of total cost:", (df.net_incurred.nlargest(int(len(df) * .001)).sum() / df.net_incurred.sum()).round(3),
          "| max row cost:", df.net_incurred.max().round(0))
    print(" written_premium vs exposure corr (premium annualised?):", df[["written_premium", "exposure"]].corr().iloc[0, 1].round(3),
          "| prem per expo mean:", (df.written_premium / df.exposure).mean().round(1))
