"""Part 2: test two candidate adjustments on top of the chosen blend (DEV_CAL + POOL + OFFSET), on both holdouts.

1. New-business factor. Estimated ONLY on the fit rows (in-sample blend predictions saved by p2_eval.py):
       k_raw = (A/E on new-business fit rows) / (A/E on renewal fit rows),  on capped claims (robust to one large loss)
       k     = 1 + Z (k_raw - 1),  Z = min(1, sqrt(n_NB_claims / 1082))  (limited-fluctuation credibility, 90% / +-5%)
   then applied to the held-out new-business rows. 'nb_full' uses Z = 1 for reference.
2. Cost trend (time holdout only - the group holdout has no time gap): developed-market freq+sev log-linear slope
   (years <= 2023), applied for the 1-year gap 2023 -> 2024.
Adopt an adjustment only if it improves D2 on both holdouts (trend: on the one holdout where it is testable, and only
if it also does not hurt D2 on capped claims).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from p2_features import dev_claims, get_frames
from train import d2
from trend import factor, yearly

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache_p2"
BLEND = ["DEV_CAL", "POOL", "OFFSET"]
FULL_CRED = 1082


def nb_factor(fit: pd.DataFrame) -> tuple[float, float, int]:
    p = fit[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy()
    nb = fit["new_business"].to_numpy().astype(bool)
    y = fit["net_capped"].to_numpy()
    ae = lambda m: y[m].sum() / p[m].sum()  # noqa: E731
    k_raw = ae(nb) / ae(~nb)
    n = int(fit.loc[nb, "claim_count"].sum())
    return float(k_raw), float(min(1.0, np.sqrt(n / FULL_CRED))), n


def score(h: pd.DataFrame, p: np.ndarray) -> dict:
    y, yc = h["net_incurred"].to_numpy(), h["net_capped"].to_numpy()
    nb = h["new_business"].to_numpy().astype(bool)
    return {"D2": d2(y, p), "D2_cap": d2(yc, p * yc.sum() / y.sum()), "AE": y.sum() / p.sum(),
            "NB_AE": y[nb].sum() / p[nb].sum(), "REN_AE": y[~nb].sum() / p[~nb].sum()}


def main() -> None:
    curves = reported_fraction_curves(dev_claims())
    rows = []
    for ln in ("contents", "buildings", "motor"):
        dev = add_development(get_frames("developed", ln)[0], curves[ln])
        s = factor(yearly(dev[dev["year"] <= 2023]), 2024, "freqsev")[1]
        for scheme in ("time", "group"):
            h = pd.read_parquet(CACHE / f"hold_{scheme}_{ln}.parquet")
            f = pd.read_parquet(CACHE / f"fit_{scheme}_{ln}.parquet")
            base = h[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy()
            nbm = h["new_business"].to_numpy().astype(bool)
            k_raw, z, n = nb_factor(f)
            k = 1 + z * (k_raw - 1)
            cands = {"base": base, "nb_cred": np.where(nbm, base * k, base), "nb_full": np.where(nbm, base * k_raw, base)}
            if scheme == "time":
                tf = float(np.exp(s))
                cands["trend"] = base * tf
                cands["nb_cred+trend"] = cands["nb_cred"] * tf
            for name, p in cands.items():
                rows.append({"line": ln, "scheme": scheme, "cand": name, **score(h, p), "k_raw": k_raw, "Z": z, "n_nb": n,
                             "k": k, "trend_slope": s})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    df.round(4).to_csv(ROOT / "p2_adjust_results.csv", index=False)


if __name__ == "__main__":
    main()
