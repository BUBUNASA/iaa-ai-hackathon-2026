"""Validate a cost-trend factor on the 2023 / 2024 holdouts (trend estimated ONLY from years before the holdout).

The model is trained on pooled years, so it predicts roughly the exposure-weighted average train-year level.
Trend factor for holdout year Y = exp(g * (Y - ybar)), ybar = exposure-weighted mean train year,
g = exposure-weighted slope of log(developed cost per exposure) on year, fitted on train years only.
Variants: total cost trend, or frequency and severity trends estimated separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from prep import LINES
from train import d2
from trend import factor, slope, yearly
from val_preds import CACHE


curves = reported_fraction_curves()
rows = []
for ln in LINES:
    train, _ = get_frames(ln)
    train = add_development(train, curves[ln])
    all_g = yearly(train)
    print(f"\n== {ln} yearly developed (capped) cost/exposure, freq, sev")
    print(all_g[["cpe", "freq", "sev"]].round(4).to_string())
    for vy in (2023, 2024):
        v = pd.read_parquet(CACHE / f"valpred_{ln}_{vy}.parquet")
        y, f = v["net_incurred"].to_numpy(), v["dev_f"].to_numpy()
        base = v["fs1"].to_numpy() * f
        g = yearly(train[train["year"] < vy])
        rows.append(dict(line=ln, vy=vy, variant="none", trend_pa=0.0, factor=1.0, D2=d2(y, base), AE=y.sum() / base.sum()))
        for kind in ("total", "freqsev"):
            for damp in (1.0, 0.75, 0.5):
                fac, s = factor(g, vy, kind, damp)
                p = base * fac
                rows.append(dict(line=ln, vy=vy, variant=f"{kind}_x{damp}", trend_pa=round(damp * s, 4), factor=round(fac, 4),
                                 D2=d2(y, p), AE=y.sum() / p.sum()))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for ln in LINES:
    print(f"\n===== {ln}")
    print(res[res.line == ln].round(4).to_string(index=False))
res.to_csv("trend_eval_results.csv", index=False)

# Factor that would apply to the 2025 test year from all four historical years
print("\n== implied 2025 factors (fit on 2021-24)")
for ln in LINES:
    train, _ = get_frames(ln)
    train = add_development(train, curves[ln])
    g = yearly(train)
    for kind in ("total", "freqsev"):
        for damp in (1.0, 0.75, 0.5):
            fac, s = factor(g, 2025, kind, damp)
            print(f"{ln:10s} {kind:8s} damp={damp}: trend/yr={damp*s:+.4f} factor={fac:.4f}")
