import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 100)
D = "data/public/"
t = {n: pd.read_parquet(D + n + ".parquet") for n in
     ["customers", "addresses", "weather", "policies_motor", "policies_buildings", "policies_contents",
      "claims", "test_motor", "test_buildings", "test_contents", "sample_submission"]}
for n, d in t.items():
    print(f"\n== {n} {d.shape}")
    print(d.dtypes.astype(str).to_dict() if n in ("claims", "policies_motor") else "")
    na = d.isna().mean()
    print("nulls>0:", na[na > 0].round(3).to_dict())

c = t["claims"]
print("\n== claims head\n", c.head(3).T)
print(c.groupby(["line", "peril"]).agg(n=("claim_id", "size"), inc=("incurred_to_date", "mean"),
                                       p99=("incurred_to_date", lambda s: s.quantile(.99))).round(0))
print(c.status.value_counts())
c["ly"] = pd.to_datetime(c.loss_date).dt.year
print(c.groupby(["line", "ly"]).size().unstack())
c["lag"] = (pd.to_datetime(c.report_date) - pd.to_datetime(c.loss_date)).dt.days
print("report lag days by line:\n", c.groupby("line").lag.describe().round(1))
print("incurred stats:\n", c.incurred_to_date.describe(), "\nrecoveries>0:", (c.recoveries > 0).mean(),
      "\npaid>incurred:", (c.paid_to_date > c.incurred_to_date + 1e-6).mean())

for ln in ["motor", "buildings", "contents"]:
    p, te = t["policies_" + ln], t["test_" + ln]
    print(f"\n== {ln}: train {p.shape} test {te.shape}")
    print("exposure by year:", p.assign(y=pd.to_datetime(p.period_start).dt.year).groupby("y").exposure.agg(["size", "sum"]).round(0).to_dict("index"))
    print("test exposure:", te.exposure.describe().round(3).to_dict())
    print("test cols missing vs train:", sorted(set(p.columns) - set(te.columns)))
    print("renewal_decision:", p.renewal_decision.value_counts().to_dict())
    print("test policies seen in train:", te.policy_id.isin(p.policy_id).mean().round(3),
          "| test customers seen:", te.customer_id.isin(p.customer_id).mean().round(3))
    print("dup row_id:", p.row_id.duplicated().sum(), "| exposure<=0:", (p.exposure <= 0).sum())

sub = t["sample_submission"]
allids = pd.concat([t["test_motor"].row_id, t["test_buildings"].row_id, t["test_contents"].row_id])
print("\nsample_submission", sub.shape, "ids match test:", set(sub.row_id) == set(allids), sub.prediction.describe().to_dict())
print("weather years:", sorted(t["weather"].year.unique()))
