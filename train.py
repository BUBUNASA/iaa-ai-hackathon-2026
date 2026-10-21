"""Train/validate per-line LightGBM Tweedie models with a time-based split (fit 2021-23, validate 2024)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score

from features import build, feature_columns, to_model_frame

POWER = 1.5
PARAMS = dict(
    objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, max_cat_to_onehot=8, cat_smooth=50, verbose=-1, seed=0, num_threads=8,
)


def d2(y: np.ndarray, p: np.ndarray) -> float:
    return float(d2_tweedie_score(y, np.clip(p, 1e-6, None), power=POWER))


def fit_eval(train: pd.DataFrame, cols: list[str], target: str = "net_capped", rounds: int = 3000,
             params: dict | None = None, val_year: int = 2024) -> dict:
    X, _ = to_model_frame(train, train.head(1), cols)
    fit = train["year"] < val_year
    val = train["year"] == val_year
    prm = {**PARAMS, **(params or {})}
    dtr = lgb.Dataset(X[fit], train.loc[fit, target] / train.loc[fit, "exposure"], weight=train.loc[fit, "exposure"])
    dva = lgb.Dataset(X[val], train.loc[val, target] / train.loc[val, "exposure"], weight=train.loc[val, "exposure"], reference=dtr)
    m = lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = m.predict(X[val], num_iteration=m.best_iteration) * train.loc[val, "exposure"].to_numpy()
    y_unc = train.loc[val, "net_incurred"].to_numpy()
    y_cap = train.loc[val, "net_capped"].to_numpy()
    return {"model": m, "best_iter": m.best_iteration, "d2_uncapped": d2(y_unc, pred), "d2_capped": d2(y_cap, pred),
            "A/E": float(y_unc.sum() / pred.sum()), "n_features": len(cols)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*", default=["motor", "buildings", "contents"])
    a = ap.parse_args()
    for ln in a.lines:
        t0 = time.time()
        train, test, meta = build(ln)
        print(f"[{ln}] built in {time.time()-t0:.0f}s train={train.shape} test={test.shape} cap={meta['cap']:.0f}")
        cols = feature_columns(train)
        r = fit_eval(train, cols)
        print(f"[{ln}] D2 uncapped={r['d2_uncapped']:.4f} capped={r['d2_capped']:.4f} A/E={r['A/E']:.3f} "
              f"iters={r['best_iter']} feats={r['n_features']} ({time.time()-t0:.0f}s)")
        imp = pd.Series(r["model"].feature_importance("gain"), index=cols).sort_values(ascending=False)
        print((imp / imp.sum()).head(12).round(3).to_string())
