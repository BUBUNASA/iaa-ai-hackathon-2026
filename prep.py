"""Data loading and target construction for the IAA 2026 hackathon (Part 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "public"
LINES = ("motor", "buildings", "contents")
DATE_COLS = ("period_start", "period_end")


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def load_policies(line: str, test: bool = False) -> pd.DataFrame:
    df = read(("test_" if test else "policies_") + line)
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c])
    df["year"] = df["period_start"].dt.year
    # Data-quality fix: ~0.4% of training rows record exposure in days (e.g. 365) or on a 360-day basis (1.014).
    # Recompute those from the policy dates (days/365, the convention every other row follows). Test has none.
    bad = df["exposure"] > 1.0001
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    df.loc[bad, "exposure"] = (days[bad] / 365).clip(upper=1.0)
    return df


def load_claims() -> pd.DataFrame:
    c = read("claims")
    for col in ("loss_date", "report_date"):
        c[col] = pd.to_datetime(c[col])
    # Ultimate-cost proxy at the valuation date: incurred net of recoveries, repudiated claims excluded.
    c["net_incurred"] = np.where(c["status"] == "repudiated", 0.0, c["incurred_to_date"] - c["recoveries"])
    return c


def attach_claims(pol: pd.DataFrame, claims: pd.DataFrame, line: str, cap_q: float = 0.999) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each claim to the policy row whose [period_start, period_end] contains the loss date.

    Adds claim_count, net_incurred (uncapped), net_capped (each claim capped at the cap_q quantile of the
    line's claim distribution), paid, open_count. Returns (policy rows, unmatched claims).
    """
    cl = claims[claims["line"] == line].copy()
    cap = cl.loc[cl["net_incurred"] > 0, "net_incurred"].quantile(cap_q)
    cl["net_capped"] = cl["net_incurred"].clip(upper=cap)
    key = pol[["row_id", "policy_id", "period_start", "period_end"]]
    m = cl.merge(key, on="policy_id", how="left")
    ok = (m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])
    matched = m[ok]
    # a claim could in principle hit two overlapping rows: keep the first, flag duplicates
    dups = matched["claim_id"].duplicated(keep="first")
    matched = matched[~dups]
    unmatched = cl[~cl["claim_id"].isin(matched["claim_id"])]
    agg = matched.groupby("row_id").agg(
        claim_count=("claim_id", "size"),
        net_incurred=("net_incurred", "sum"),
        net_capped=("net_capped", "sum"),
        paid=("paid_to_date", "sum"),
        open_count=("status", lambda s: (s == "open").sum()),
    )
    out = pol.merge(agg, left_on="row_id", right_index=True, how="left")
    for c in ("claim_count", "net_incurred", "net_capped", "paid", "open_count"):
        out[c] = out[c].fillna(0)
    out.attrs["duplicate_matches"] = int(dups.sum())
    out.attrs["cap"] = float(cap)
    return out, unmatched
