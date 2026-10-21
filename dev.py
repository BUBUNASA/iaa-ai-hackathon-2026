"""Claim-development (IBNR / reporting-lag) adjustment for immature recent exposure.

Claims are valued at 2024-12-31, so recent loss dates are only partly reported (median lag 9 days but bodily-injury
claims take 181 days on average and up to ~1,400 days). We estimate a per-peril reporting-lag CDF with a chain ladder
on the loss-age x reporting-lag count triangle (weekly blocks), cost-weight the perils per line, and convert it to a
per-row "reported fraction" f_row = average over the policy period of P(claim already reported at valuation).

Training then uses developed exposure (exposure * f_row) so the model learns *ultimate* rates; the test target
(expected ultimate cost for 2025) is predicted with f = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prep import load_claims

VAL = pd.Timestamp("2024-12-31")
BLOCK = 7            # weekly development blocks
MAX_BLOCKS = 160     # ~3 years; development beyond this assumed complete
MIN_DENOM = 30       # skip chain-ladder links built on fewer claims than this (factor -> 1)
MAX_AGE = 2200       # days


def _peril_cdf(lags: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """P(lag <= age) on a daily age grid 0..MAX_AGE for one peril, via right-truncated chain ladder."""
    nb = MAX_AGE // BLOCK + 2
    j = np.minimum(ages // BLOCK, nb - 1)
    k = np.minimum(lags // BLOCK, nb - 1)
    tri = np.zeros((nb, nb))
    np.add.at(tri, (j, k), 1.0)
    cum = tri.cumsum(axis=1)
    f = np.ones(MAX_BLOCKS)
    for kk in range(MAX_BLOCKS):
        rows = np.arange(kk + 2, nb)              # loss-age blocks where dev blocks kk and kk+1 are complete
        den = cum[rows, kk].sum()
        if den >= MIN_DENOM:
            f[kk] = max(cum[rows, kk + 1].sum() / den, 1.0)
    tail = np.cumprod(f[::-1])[::-1]               # product of links from kk to the end
    F_blocks = 1.0 / tail                          # CDF at end of dev block kk (lag <= BLOCK*(kk+1)-1)
    x = np.concatenate([[0.0], BLOCK * (np.arange(MAX_BLOCKS) + 1) - 1.0, [MAX_AGE + 1.0]])
    y = np.concatenate([[0.0], F_blocks, [1.0]])
    return np.interp(np.arange(MAX_AGE + 1), x, y)


def reported_fraction_curves(claims: pd.DataFrame | None = None, ref_years=(2021, 2022)) -> dict[str, np.ndarray]:
    """Cost-weighted P(reported by age a days) per line: {line: array over a = 0..MAX_AGE}."""
    c = load_claims() if claims is None else claims
    c = c[(c["loss_date"] <= VAL) & (c["status"] != "repudiated")].copy()
    c["age"] = (VAL - c["loss_date"]).dt.days.astype(int)
    c["lag"] = (c["report_date"] - c["loss_date"]).dt.days.astype(int).clip(lower=0)
    out = {}
    for line, g in c.groupby("line"):
        cdfs, wts = {}, {}
        ref = g[g["loss_date"].dt.year.isin(ref_years) & (g["net_incurred"] > 0)]
        for peril, gp in g.groupby("peril"):
            cdfs[peril] = _peril_cdf(gp["lag"].to_numpy(), gp["age"].to_numpy())
            wts[peril] = float(ref.loc[ref["peril"] == peril, "net_incurred"].clip(upper=ref["net_incurred"].quantile(0.999)).sum())
        tot = sum(wts.values())
        out[line] = sum(cdfs[p] * wts[p] / tot for p in cdfs)
    return out


def row_reported_fraction(pol: pd.DataFrame, curve: np.ndarray) -> np.ndarray:
    """Average of curve over the ages of each day of the policy period (period_start..period_end)."""
    G = np.concatenate([[0.0], np.cumsum(curve)])           # G[n] = sum_{a<n} curve[a]
    hi = (VAL - pol["period_start"]).dt.days.to_numpy().clip(0, MAX_AGE)
    lo = (VAL - pol["period_end"]).dt.days.to_numpy().clip(0, MAX_AGE)
    n = (hi - lo + 1).clip(min=1)
    return (G[hi + 1] - G[lo]) / n


def add_development(frame: pd.DataFrame, curve: np.ndarray, test: bool = False) -> pd.DataFrame:
    """Adds `dev_f` (reported fraction; 1.0 for the test year, which is predicted on an ultimate basis)."""
    out = frame.copy()
    out["dev_f"] = 1.0 if test else row_reported_fraction(frame, curve)
    return out
