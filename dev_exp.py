"""Does a reporting-lag (IBNR) adjustment improve validation? Compare baseline vs developed-exposure training."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from features import to_model_frame
from final import REG, select_cols
from prep import LINES
from train import PARAMS, d2


def fit_eval_dev(train, cols, val_year, dev: bool):
    X, _ = to_model_frame(train, train.head(1), cols)
    f = train["dev_f"].to_numpy() if dev else np.ones(len(train))
    w = train["exposure"].to_numpy() * f
    y = train["net_capped"].to_numpy() / w
    fit, val = (train["year"] < val_year).to_numpy(), (train["year"] == val_year).to_numpy()
    dtr = lgb.Dataset(X[fit], y[fit], weight=w[fit])
    dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr)
    m = lgb.train({**PARAMS, **REG}, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    rate = m.predict(X[val], num_iteration=m.best_iteration)
    exp_reported = rate * train.loc[val, "exposure"].to_numpy() * f[val]    # what we expect to *see* at valuation
    act = train.loc[val, "net_incurred"].to_numpy()
    return dict(d2=d2(act, exp_reported), AE=act.sum() / exp_reported.sum(), iters=m.best_iteration,
                ultimate_rate=float((rate * train.loc[val, "exposure"].to_numpy()).sum() / train.loc[val, "exposure"].sum()))


if __name__ == "__main__":
    curves = reported_fraction_curves()
    for ln in LINES:
        c = curves[ln]
        print(f"\n===== {ln}: reported fraction at age 30d/90d/180d/365d/730d = "
              f"{c[30]:.3f}/{c[90]:.3f}/{c[180]:.3f}/{c[365]:.3f}/{c[730]:.3f}", flush=True)
        train, _ = get_frames(ln)
        train = add_development(train, c)
        g = train.groupby("year")
        tbl = pd.DataFrame({
            "mean_f": g.apply(lambda d: np.average(d["dev_f"], weights=d["exposure"]), include_groups=False),
            "reported_cost/expo": g.apply(lambda d: d["net_incurred"].sum() / d["exposure"].sum(), include_groups=False),
            "developed_cost/expo": g.apply(lambda d: d["net_incurred"].sum() / (d["exposure"] * d["dev_f"]).sum(), include_groups=False),
        })
        print(tbl.round(3).to_string(), flush=True)
        cols = select_cols(train)
        for vy in (2024, 2023):
            for dev in (False, True):
                r = fit_eval_dev(train, cols, vy, dev)
                print(f"{ln:10s} val={vy} dev={dev!s:5s} D2(reported)={r['d2']:.4f} A/E={r['AE']:.3f} "
                      f"iters={r['iters']} ultimate_rate_pred={r['ultimate_rate']:.1f}", flush=True)
