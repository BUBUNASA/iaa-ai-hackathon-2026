"""Cost-trend factor estimated from developed, exposure-weighted history (train years only).

The model is trained on pooled years, so it predicts roughly the exposure-weighted average train-year level.
factor(Y) = exp(damp * g * (Y - ybar)), ybar = exposure-weighted mean train year,
g = freq slope + sev slope (log-linear, exposure / claim-count weighted). Validated in trend_eval.py (damp = 0.75).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def yearly(train: pd.DataFrame) -> pd.DataFrame:
    g = train.assign(eff=train["exposure"] * train["dev_f"]).groupby("year").agg(
        eff=("eff", "sum"), cnt=("claim_count", "sum"), cost=("net_capped", "sum"))
    g["cpe"], g["freq"] = g["cost"] / g["eff"], g["cnt"] / g["eff"]
    g["sev"] = g["cost"] / g["cnt"]
    return g


def slope(years, vals, w) -> float:
    return float(np.polyfit(years, np.log(vals), 1, w=np.sqrt(w))[0])


def factor(g: pd.DataFrame, vy: int, kind: str, damp: float = 1.0) -> tuple[float, float]:
    ybar = float(np.average(g.index, weights=g["eff"]))
    if kind == "total":
        s = slope(g.index, g["cpe"], g["eff"])
    else:  # frequency and severity trends estimated separately
        s = slope(g.index, g["freq"], g["eff"]) + slope(g.index, g["sev"], g["cnt"])
    return float(np.exp(damp * s * (vy - ybar))), s
