"""Per-peril frequency x severity: one Poisson x Gamma pair per peril, each on its own developed exposure, summed.

Adopted for buildings and contents (beat the line-level model on both the 2023 and 2024 holdouts, improve_eval_out.txt);
not for motor (mixed: +0.0025 on 2023, -0.0022 on 2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dev import VAL, _peril_cdf, row_reported_fraction
from models import fit_predict


def peril_curves(claims: pd.DataFrame, line: str) -> dict[str, np.ndarray]:
    c = claims[(claims["line"] == line) & (claims["loss_date"] <= VAL) & (claims["status"] != "repudiated")]
    age = (VAL - c["loss_date"]).dt.days.astype(int).to_numpy()
    lag = (c["report_date"] - c["loss_date"]).dt.days.astype(int).clip(lower=0).to_numpy()
    return {p: _peril_cdf(lag[(c["peril"] == p).to_numpy()], age[(c["peril"] == p).to_numpy()]) for p in sorted(c["peril"].unique())}


def peril_targets(train: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    """Per-row, per-peril claim count and capped net cost (same matching rule and line cap as prep.attach_claims)."""
    cl = claims[claims["line"] == line].copy()
    cap = cl.loc[cl["net_incurred"] > 0, "net_incurred"].quantile(0.999)
    cl["net_capped"] = cl["net_incurred"].clip(upper=cap)
    m = cl.merge(train[["row_id", "policy_id", "period_start", "period_end"]], on="policy_id")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])].drop_duplicates("claim_id")
    return m.groupby(["row_id", "peril"]).agg(n=("claim_id", "size"), c=("net_capped", "sum")).reset_index()


def peril_frame(df: pd.DataFrame, t: pd.DataFrame, curve: np.ndarray, test: bool = False) -> pd.DataFrame:
    """Copy of `df` with targets and dev_f replaced by one peril's (t indexed by row_id)."""
    out = df.copy()
    out["claim_count"] = out["row_id"].map(t["n"]).fillna(0).to_numpy()
    out["net_capped"] = out["row_id"].map(t["c"]).fillna(0).to_numpy()
    out["dev_f"] = 1.0 if test else row_reported_fraction(out, curve)
    return out


def peril_fs_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], claims: pd.DataFrame, line: str,
                     seeds, val_year: int = 2024, refit_mult: float = 1.15) -> tuple[np.ndarray, dict]:
    """Final fit: per peril, early-stop freq and sev on `val_year` (seed 0), refit on all years with seeds; sum rates."""
    curves, ptg = peril_curves(claims, line), peril_targets(train, claims, line)
    rate, iters = np.zeros(len(test)), {}
    for p, curve in curves.items():
        t = ptg[ptg["peril"] == p].set_index("row_id")
        tr = peril_frame(train, t, curve)
        te = peril_frame(test, t.iloc[0:0], curve, test=True)
        fit, val = tr[tr["year"] < val_year], tr[tr["year"] == val_year]
        _, it_f = fit_predict(fit, val, cols, "freq", seeds=(0,), val=val)
        _, it_s = fit_predict(fit, val, cols, "sev", seeds=(0,), val=val)
        fr, _ = fit_predict(tr, te, cols, "freq", seeds=seeds, rounds=int(it_f * refit_mult))
        sv, _ = fit_predict(tr, te, cols, "sev", seeds=seeds, rounds=int(it_s * refit_mult))
        rate += fr * sv
        iters[p] = {"freq": it_f, "sev": it_s}
        print(f"  {line} peril {p}: freq={it_f} sev={it_s}", flush=True)
    return rate, iters
