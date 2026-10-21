"""Explore claim reporting lag / case-reserve development to size the 2024 immaturity."""
import numpy as np, pandas as pd
from prep import load_claims, load_policies

VAL = pd.Timestamp("2024-12-31")
c = load_claims()
c["lag"] = (c["report_date"] - c["loss_date"]).dt.days
c["ly"] = c["loss_date"].dt.year
print("claims by line/loss-year\n", c.groupby(["line", "ly"]).size().unstack())
print("\nmax report_date", c["report_date"].max(), "max loss_date", c["loss_date"].max(), "neg lags", (c["lag"] < 0).sum())

print("\nlag quantiles by line (loss 2021-22, ~complete)")
old = c[c["ly"] <= 2022]
print(old.groupby("line")["lag"].describe(percentiles=[.5, .9, .95, .99, .999]).round(1))
print("\nlag quantiles by peril (loss 2021-22)")
print(old.groupby(["line", "peril"])["lag"].describe(percentiles=[.5, .9, .99]).round(1))

print("\nstatus share by loss-year and line")
print(c.groupby(["line", "ly"])["status"].value_counts(normalize=True).unstack().round(3))

print("\nopen claims: incurred vs paid by loss year (mean incurred, mean paid)")
o = c[c["status"] == "open"]
print(o.groupby(["line", "ly"])[["incurred_to_date", "paid_to_date"]].mean().round(0))

print("\nmean net_incurred per positive claim by line/loss-year")
print(c[c["net_incurred"] > 0].groupby(["line", "ly"])["net_incurred"].mean().unstack().round(0))

for line in ("motor", "buildings", "contents"):
    p = load_policies(line)
    print(line, "2024 rows period_end max", p.loc[p.year == 2024, "period_end"].max(),
          "| exposure cols:", [x for x in p.columns if "expos" in x.lower()])
