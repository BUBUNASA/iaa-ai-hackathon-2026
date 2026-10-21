"""Model zoo used for validation AND the final fit: Tweedie pure-premium, Poisson frequency, Gamma severity, seed bagging.

All models are trained on a developed basis (exposure x reported fraction `dev_f`, see dev.py) so predictions are
ultimate-basis rates per unit exposure, matching the 2025 target.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import to_model_frame
from final import REG, select_cols  # noqa: F401  (re-exported)
from train import PARAMS

SEV_OVERRIDE = dict(min_data_in_leaf=300)
SEEDS = (0, 1, 2, 3, 4)


def _fit(params: dict, X, y, w, rounds: int | None, Xv=None, yv=None, wv=None):
    ds = lgb.Dataset(X, y, weight=w)
    if rounds is None:  # early stopping on the validation set
        dv = lgb.Dataset(Xv, yv, weight=wv, reference=ds)
        m = lgb.train(params, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration
    return lgb.train(params, ds, rounds), rounds


def fit_predict(train: pd.DataFrame, apply_to: pd.DataFrame, cols: list[str], kind: str, seeds=(0,),
                val: pd.DataFrame | None = None, rounds: int | None = None):
    """Fit `kind` in {tweedie, freq, sev} on `train` and predict ultimate rate per unit exposure (freq: claims per
    exposure; sev: cost per claim) for `apply_to`. If `val` is given, the first seed is early-stopped on it and its
    iteration count is reused for the other seeds. Returns (mean prediction over seeds, iterations used)."""
    Xtr, Xap = to_model_frame(train, apply_to, cols)
    w_dev = (train["exposure"] * train["dev_f"]).to_numpy()
    if kind == "tweedie":
        params, y, w, mask = {**PARAMS, **REG}, train["net_capped"].to_numpy() / w_dev, w_dev, None
    elif kind == "freq":
        params = {**PARAMS, **REG, "objective": "poisson"}
        params.pop("tweedie_variance_power", None)
        y, w, mask = train["claim_count"].to_numpy() / w_dev, w_dev, None
    elif kind == "sev":
        params = {**PARAMS, **REG, **SEV_OVERRIDE, "objective": "gamma"}
        params.pop("tweedie_variance_power", None)
        mask = (train["claim_count"] > 0).to_numpy() & (train["net_capped"] > 0).to_numpy()
        cc = train["claim_count"].to_numpy()
        y = np.where(mask, train["net_capped"].to_numpy() / np.maximum(cc, 1), 1.0)
        w = cc.astype(float)
    else:
        raise ValueError(kind)
    if mask is None:
        mask = np.ones(len(train), dtype=bool)
    Xm, ym, wm = Xtr[mask], y[mask], w[mask]
    Xv = yv = wv = None
    if val is not None:
        Xt2, Xv_all = to_model_frame(train, val, cols)
        Xv = Xv_all
        if kind == "sev":
            vm = ((val["claim_count"] > 0) & (val["net_capped"] > 0)).to_numpy()
            Xv, yv, wv = Xv_all[vm], (val["net_capped"] / val["claim_count"].clip(lower=1)).to_numpy()[vm], val["claim_count"].to_numpy()[vm].astype(float)
        else:
            wd = (val["exposure"] * val["dev_f"]).to_numpy()
            tgt = val["net_capped"] if kind == "tweedie" else val["claim_count"]
            yv, wv = tgt.to_numpy() / wd, wd
    preds, its = [], None
    for i, s in enumerate(seeds):
        prm = {**params, "seed": int(s), "bagging_seed": int(s), "feature_fraction_seed": int(s)}
        if rounds is not None:
            m, _ = _fit(prm, Xm, ym, wm, rounds)
            its = rounds
        elif val is not None and i == 0:
            m, its = _fit(prm, Xm, ym, wm, None, Xv, yv, wv)
        else:
            m, _ = _fit(prm, Xm, ym, wm, its)
        preds.append(m.predict(Xap))
    return np.mean(preds, axis=0), its
