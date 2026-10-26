"""Part 2: harmonised modelling frames for the developed and emerging markets (identical columns, cached).

Harmonisation (see p2_profile_out.txt for the evidence):
- age: emerging only has bands -> band the developed exact ages the same way; exact ages dropped.
- developed-only fields (credit_band, date_of_birth, annual_mileage, telematics) dropped: not available where we predict.
- occupation: emerging is free text ('Retired ', 'retired', 'Self-employed nurse') -> trimmed lower-case base occupation
  + self_employed flag.
- construction_type: developed 'Brick'/'brick'/'BRICK'/'W'/..., emerging codes BRK/WOD/MIX/STN -> one vocabulary
  (category shares match 1:1: W+T~WOD 12%, non_standard~MIX 10%, stone~STN 8%).
- gender: emerging only (allowed for pricing there) -> missing in developed.
- market flag added; exposure recorded in days / 360-day basis fixed by prep.load_policies (same defect in both markets).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import prep
from features import _history, _prior_exposure

ROOT = Path(__file__).parent
P2 = ROOT / "data" / "part2"
CACHE = ROOT / "data" / "cache_p2"
MARKETS = ("developed", "emerging")
AGE_BINS = [0, 24, 34, 49, 64, 200]
AGE_LABELS = ["17-24", "25-34", "35-49", "50-64", "65+"]
CONSTRUCTION = {"brick": "brick", "brk": "brick", "w": "timber", "t": "timber", "wod": "timber", "stone": "stone", "stn": "stone",
                "non_standard": "non_standard", "mix": "non_standard"}
DROP = ["driver_age", "policyholder_age", "driver_age_band", "policyholder_age_band", "date_of_birth", "credit_band",
        "annual_mileage", "telematics_opt_in", "telematics_score"]


def use_market(market: str) -> None:
    prep.DATA = P2 / market  # prep.read() resolves files from this module-level path at call time


def _harmonise(df: pd.DataFrame, market: str, line: str) -> pd.DataFrame:
    if market == "developed":
        src = "driver_age" if line == "motor" else "policyholder_age"
        df["age_band"] = pd.cut(df[src], AGE_BINS, labels=AGE_LABELS).astype(object)
    else:
        df["age_band"] = df["driver_age_band" if line == "motor" else "policyholder_age_band"].astype(object)
    occ = df["occupation"].astype("string").str.strip().str.lower()
    df["self_employed"] = occ.str.startswith("self-employed").astype("boolean")
    df["occupation"] = occ.str.replace("self-employed ", "", regex=False).astype(object)
    if "construction_type" in df:
        ct = df["construction_type"].astype("string").str.strip().str.lower()
        df["construction_type"] = ct.map(CONSTRUCTION).fillna(ct).astype(object)
    if "gender" not in df:
        df["gender"] = None
    df["market"] = market
    return df.drop(columns=[c for c in DROP if c in df])


def build(market: str, line: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_market(market)
    claims = prep.load_claims()
    train, test = prep.load_policies(line), prep.load_policies(line, test=True)
    train, _ = prep.attach_claims(train, claims, line)
    pol_cust = pd.concat([prep.load_policies(l)[["policy_id", "customer_id"]] for l in prep.LINES]).drop_duplicates("policy_id")
    cust, addr, weather = prep.read("customers"), prep.read("addresses"), prep.read("weather")

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(cust, on="customer_id", how="left").merge(addr, on="address_id", how="left")
        df = df.merge(weather, on=["address_id", "year"], how="left")
        df["start_doy"] = df["period_start"].dt.dayofyear
        df["end_doy"] = df["period_end"].dt.dayofyear
        df["property_age"] = df["year"] - df["year_built"]
        df["tenure_years"] = df["year"] - df["customer_since_year"]
        return _harmonise(df, market, line)

    train, test = enrich(train), enrich(test)
    train["prior_exposure"], test["prior_exposure"] = (s.values for s in _prior_exposure(train, test))
    train = train.join(_history(train, claims, pol_cust, line))
    test = test.join(_history(test, claims, pol_cust, line))
    for d in (train, test):
        d["hist_pol_freq"] = d["hist_pol_n"] / d["prior_exposure"].clip(lower=0.25)
    return train, test


def get_frames(market: str, line: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, te = CACHE / f"{market}_{line}_train.parquet", CACHE / f"{market}_{line}_test.parquet"
    if tr.exists() and te.exists() and not refresh:
        return pd.read_parquet(tr), pd.read_parquet(te)
    train, test = build(market, line)
    train.to_parquet(tr)
    test.to_parquet(te)
    return train, test


def dev_claims() -> pd.DataFrame:
    """Developed-market claims: reporting-lag curves are estimated here (emerging lags match, see p2_profile)."""
    use_market("developed")
    return prep.load_claims()


if __name__ == "__main__":
    for m in MARKETS:
        for ln in prep.LINES:
            tr, te = get_frames(m, ln, refresh=True)
            assert list(tr.columns.drop(["claim_count", "net_incurred", "net_capped", "paid", "open_count", "renewal_decision"], errors="ignore")) \
                == list(te.columns), f"{m} {ln}: train/test columns differ"
            print(f"{m:9s} {ln:9s} train={tr.shape} test={te.shape} age_band={tr['age_band'].value_counts(dropna=False).to_dict()}", flush=True)
    a, b = get_frames("developed", "buildings")[0], get_frames("emerging", "buildings")[0]
    print("construction:", a["construction_type"].value_counts(normalize=True).round(3).to_dict(), b["construction_type"].value_counts(normalize=True).round(3).to_dict())
    print("occupation dev:", sorted(a["occupation"].dropna().unique()), "\nocc em:", sorted(b["occupation"].dropna().unique()))
    print("cols only dev:", set(a.columns) - set(b.columns), "only em:", set(b.columns) - set(a.columns))
