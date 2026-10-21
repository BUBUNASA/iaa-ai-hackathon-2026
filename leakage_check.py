"""Leakage investigation: does customer_service_contacts (and other in-period fields) reflect claims in the same row?"""
import pandas as pd, numpy as np
from prep import *
from train import fit_eval, d2
from features import build, feature_columns
pd.set_option("display.width", 250, "display.max_columns", 40)

SUSPECT = ["customer_service_contacts", "missed_payments_12m", "days_quote_to_inception", "discount_applied", "written_premium"]
for ln in LINES:
    train, test, _ = build(ln)
    print(f"\n===== {ln}")
    g = train.groupby(train.customer_service_contacts.clip(upper=6)).agg(
        rows=("row_id", "size"), expo=("exposure", "sum"), n=("claim_count", "sum"), cost=("net_capped", "sum"))
    g["freq"] = g.n / g.expo
    g["cost_per_expo"] = g.cost / g.expo
    print(g.round(3))
    # Does the signal exist only in-period? Compare claims in the SAME row vs claims in the PRIOR rows of the policy
    print("contacts mean | test vs train by year:",
          train.groupby("year").customer_service_contacts.mean().round(3).to_dict(), "test:", round(test.customer_service_contacts.mean(), 3))
    # contacts vs history-based claims (known before inception): if contacts tracked only in-period claims, hist corr ~ 0
    print("corr(contacts, claim_count same row):", train[["customer_service_contacts", "claim_count"]].corr().iloc[0, 1].round(3),
          "| corr(contacts, hist_pol_n):", train[["customer_service_contacts", "hist_pol_n"]].corr().iloc[0, 1].round(3))
    base_cols = feature_columns(train)
    for drop in [[], ["customer_service_contacts"], SUSPECT]:
        cols = [c for c in base_cols if c not in drop]
        r = fit_eval(train, cols)
        print(f"  drop={drop or 'none'}: D2={r['d2_uncapped']:.4f} A/E={r['A/E']:.3f} iters={r['best_iter']}")
