"""Part 2: segment experience that matters for the 2025 mix shift (new business, age band, gender, occupation)."""
from pathlib import Path

import pandas as pd

import prep

D = Path(__file__).parent / "data" / "part2"
BANDS = [0, 24, 34, 49, 64, 200]
LABELS = ["17-24", "25-34", "35-49", "50-64", "65+"]
pd.set_option("display.width", 250)


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(expo=("exposure", "sum"), n=("claim_count", "sum"), cost=("net_incurred", "sum"))
    g["freq"], g["sev"], g["cpe"] = g["n"] / g["expo"], g["cost"] / g["n"], g["cost"] / g["expo"]
    g["rel"] = g["cpe"] / (g["cost"].sum() / g["expo"].sum())
    return g.round(3)


for market in ("developed", "emerging"):
    prep.DATA = D / market
    claims = prep.load_claims()
    cust = pd.read_parquet(D / market / "customers.parquet")
    for ln in prep.LINES:
        pol, _ = prep.attach_claims(prep.load_policies(ln), claims, ln)
        pol = pol[pol["year"] >= 2023]  # same years in both markets
        pol = pol.merge(cust, on="customer_id", how="left")
        age_col = {"motor": "driver_age"}.get(ln, "policyholder_age")
        if market == "developed":
            pol["age_band"] = pd.cut(pol[age_col], BANDS, labels=LABELS)
        else:
            pol["age_band"] = pol["driver_age_band" if ln == "motor" else "policyholder_age_band"]
        print(f"\n######## {market} {ln} (2023-24)")
        for by in ("new_business", "age_band", "channel") + (("gender",) if market == "emerging" else ()):
            print(table(pol, by).to_string())
        if ln == "buildings":
            print("construction_type:", pol["construction_type"].value_counts().to_dict())
