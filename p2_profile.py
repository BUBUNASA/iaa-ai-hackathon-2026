"""Part 2: profile the emerging market against the developed market (data quality, drift, experience).

Run: python p2_profile.py > p2_profile_out.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

import prep
from guardrails import exposure_check

D = Path(__file__).parent / "data" / "part2"
LINES = ("motor", "buildings", "contents")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def use(market: str) -> None:
    prep.DATA = D / market  # prep.read() resolves files from this module-level path


def raw(market: str, t: str) -> pd.DataFrame:
    return pd.read_parquet(D / market / f"{t}.parquet")


def sec(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


# ---------------------------------------------------------------- 1. data quality (emerging, raw files)
sec("1. DATA QUALITY (emerging, raw)")
for ln in LINES:
    for kind in ("policies", "test"):
        df = raw("emerging", f"{kind}_{ln}")
        for c in prep.DATE_COLS:
            df[c] = pd.to_datetime(df[c])
        yrs = df["period_start"].dt.year.value_counts().sort_index().to_dict()
        nulls = df.isna().mean()
        nulls = nulls[nulls > 0].sort_values(ascending=False).round(3).head(8).to_dict()
        neg = {c: int((df[c] < 0).sum()) for c in df.select_dtypes("number").columns if (df[c] < 0).any()}
        print(f"{ln:9s} {kind:8s} rows={len(df):>7,} dup_row_id={df['row_id'].duplicated().sum()} "
              f"end<start={(df['period_end'] < df['period_start']).sum()} years={yrs}")
        print(f"   exposure_check={exposure_check(df)} expo_values>1 top={df.loc[df.exposure > 1.0001, 'exposure'].round(3).value_counts().head(4).to_dict()}")
        print(f"   nulls={nulls}")
        if neg:
            print(f"   negatives={neg}")

c = raw("emerging", "claims")
for col in ("loss_date", "report_date"):
    c[col] = pd.to_datetime(c[col])
print(f"\nclaims rows={len(c):,} dup_claim_id={c['claim_id'].duplicated().sum()} lines={c['line'].value_counts().to_dict()}")
print("   status:", c["status"].value_counts().to_dict())
print("   report<loss:", int((c["report_date"] < c["loss_date"]).sum()), " loss>2024-12-31:", int((c["loss_date"] > "2024-12-31").sum()),
      " report>2024-12-31:", int((c["report_date"] > "2024-12-31").sum()))
print("   neg incurred:", int((c["incurred_to_date"] < 0).sum()), " recoveries>incurred:", int((c["recoveries"] > c["incurred_to_date"]).sum()),
      " paid>incurred:", int((c["paid_to_date"] > c["incurred_to_date"] + 0.01).sum()))
print("   nulls:", c.isna().mean()[c.isna().mean() > 0].round(3).to_dict())

# ---------------------------------------------------------------- 2. categorical drift
sec("2. CATEGORIES: emerging vs developed, test vs train")
for ln in LINES:
    dv, em, te = raw("developed", f"policies_{ln}"), raw("emerging", f"policies_{ln}"), raw("emerging", f"test_{ln}")
    for col in em.select_dtypes(include=["object", "bool", "category"]).columns:
        if col in ("row_id", "policy_id", "customer_id", "address_id") or em[col].nunique() > 60:
            continue
        new_vs_dev = sorted(set(em[col].dropna().astype(str)) - set(dv[col].dropna().astype(str))) if col in dv else "n/a (emerging-only column)"
        new_in_test = sorted(set(te[col].dropna().astype(str)) - set(em[col].dropna().astype(str))) if col in te else []
        if new_vs_dev or new_in_test:
            print(f"{ln:9s} {col:24s} new vs developed: {new_vs_dev} | new in test: {new_in_test}")
    for col in ("driver_age_band", "policyholder_age_band"):
        if col in em:
            print(f"{ln:9s} {col} train:", em[col].value_counts(normalize=True).round(3).to_dict())
            print(f"{ln:9s} {col} test: ", te[col].value_counts(normalize=True).round(3).to_dict())

cu_e, cu_d = raw("emerging", "customers"), raw("developed", "customers")
print("\ncustomers gender:", cu_e["gender"].value_counts(dropna=False).to_dict())
print("occupation: emerging unique", cu_e["occupation"].nunique(), "developed unique", cu_d["occupation"].nunique())
print("   emerging top 25:", cu_e["occupation"].value_counts().head(25).to_dict())
print("   developed values:", cu_d["occupation"].value_counts().head(25).to_dict())
for col in ("marital_status", "previous_insurer", "marketing_segment"):
    print(f"{col}: emerging {cu_e[col].value_counts(normalize=True).round(3).head(8).to_dict()}")
    print(f"{' ' * len(col)}  developed {cu_d[col].value_counts(normalize=True).round(3).head(8).to_dict()}")
ad_e, ad_d = raw("emerging", "addresses"), raw("developed", "addresses")
print("region emerging:", ad_e["region"].value_counts().sort_index().to_dict())
for col in ad_e.select_dtypes(include=["object", "bool"]).columns:
    if col in ("address_id", "region"):
        continue
    print(f"addr {col}: emerging {ad_e[col].value_counts(normalize=True).round(3).head(8).to_dict()} | developed {ad_d[col].value_counts(normalize=True).round(3).head(8).to_dict()}")

# ---------------------------------------------------------------- 3. numeric drift (means)
sec("3. NUMERIC MEANS: developed 2024 | emerging train | emerging test")
for ln in LINES:
    dv = raw("developed", f"policies_{ln}")
    dv = dv[pd.to_datetime(dv["period_start"]).dt.year == 2024]
    em, te = raw("emerging", f"policies_{ln}"), raw("emerging", f"test_{ln}")
    em, te = em[em["exposure"] <= 1.0001], te[te["exposure"] <= 1.0001]
    num = [c for c in em.select_dtypes(include=["number", "bool"]).columns if c in te]
    rows = []
    for col in num:
        rows.append((col, dv[col].astype(float).mean() if col in dv else np.nan, em[col].astype(float).mean(), te[col].astype(float).mean()))
    t = pd.DataFrame(rows, columns=["col", "dev2024", "em_train", "em_test"]).set_index("col")
    t["test/train"] = t["em_test"] / t["em_train"]
    print(f"-- {ln}\n{t.round(3).to_string()}")
    for col in ("channel", "cover_type", "payment_method"):
        if col in em:
            print(f"   {col}: train {em[col].value_counts(normalize=True).round(3).to_dict()} | test {te[col].value_counts(normalize=True).round(3).to_dict()}")
    print(f"   test policy_ids seen in train: {te['policy_id'].isin(em['policy_id']).mean():.3f}; test new_business share {te['new_business'].mean():.3f}")

we, wd = raw("emerging", "weather"), raw("developed", "weather")
print("\nweather means by year, emerging:\n", we.groupby("year").mean(numeric_only=True).round(2).to_string())
print("weather means by year, developed:\n", wd.groupby("year").mean(numeric_only=True).round(2).to_string())

# ---------------------------------------------------------------- 4. experience
sec("4. EXPERIENCE (reported at 31-Dec-2024, exposure fixed)")
exp_rows = []
for market in ("developed", "emerging"):
    use(market)
    claims = prep.load_claims()
    for ln in LINES:
        pol, unmatched = prep.attach_claims(prep.load_policies(ln), claims, ln)
        print(f"{market:9s} {ln:9s} unmatched claims={len(unmatched):,} ({len(unmatched) / max(1, (claims['line'] == ln).sum()):.2%}) "
              f"dup matches={pol.attrs['duplicate_matches']} cap={pol.attrs['cap']:,.0f}")
        g = pol.groupby("year").agg(rows=("row_id", "size"), expo=("exposure", "sum"), n=("claim_count", "sum"),
                                    cost=("net_incurred", "sum"), open=("open_count", "sum"), prem=("written_premium", "sum"))
        g["freq"] = g["n"] / g["expo"]
        g["sev"] = g["cost"] / g["n"]
        g["cpe"] = g["cost"] / g["expo"]
        g["open%"] = g["open"] / g["n"]
        g["LR"] = g["cost"] / g["prem"]
        g["prem/expo"] = g["prem"] / g["expo"]
        for y, r in g.iterrows():
            exp_rows.append({"market": market, "line": ln, "year": y, **r.to_dict()})
        if market == "emerging":
            pr = pol.merge(raw("emerging", "addresses")[["address_id", "region"]], on="address_id", how="left")
            rg = pr.groupby("region").agg(expo=("exposure", "sum"), n=("claim_count", "sum"), cost=("net_incurred", "sum"), prem=("written_premium", "sum"))
            rg["freq"], rg["sev"], rg["cpe"], rg["LR"] = rg["n"] / rg["expo"], rg["cost"] / rg["n"], rg["cost"] / rg["expo"], rg["cost"] / rg["prem"]
            print(f"   emerging {ln} by region:\n{rg.round(3).to_string()}")
    pm = claims.groupby(["line", "peril"]).agg(n=("claim_id", "size"), cost=("net_incurred", "sum"))
    pm["share_n"] = pm["n"] / pm.groupby(level=0)["n"].transform("sum")
    pm["share_cost"] = pm["cost"] / pm.groupby(level=0)["cost"].transform("sum")
    pm["mean"] = pm["cost"] / pm["n"]
    print(f"{market} peril mix:\n{pm.round(3).to_string()}")
    lag = (claims["report_date"] - claims["loss_date"]).dt.days
    print(f"{market} report lag days by line: {lag.groupby(claims['line']).quantile([0.5, 0.9]).unstack().round(0).to_dict('index')}")
    print(f"{market} repudiated share: {(claims['status'] == 'repudiated').mean():.3f}; recoveries/incurred: {claims['recoveries'].sum() / claims['incurred_to_date'].sum():.3f}")
ex = pd.DataFrame(exp_rows)
print("\n", ex.round(3).to_string(index=False))
ex.to_csv(Path(__file__).parent / "p2_experience.csv", index=False)
