"""Part 2 model variants for the emerging market, shared by p2_eval.py (validation) and p2_final.py (submission).

All variants are frequency (Poisson) x severity (Gamma) LightGBM, trained on a developed basis (exposure x reported
fraction), early-stopped on a grouped (policy_id) inner split of the *emerging* training rows - the target domain - then
refit on all training rows with 1.15x rounds (Part 1 protocol).

  EM      emerging data only.
  DEV     developed-market model applied to emerging rows (region/gender/market excluded: unseen or absent there),
          rescaled by the emerging actual/expected (separately for claim count and cost per claim) -> 'DEV_CAL'.
  POOL    one model on developed + emerging rows with a market flag (emerging borrows strength from developed).
  OFFSET  DEV_CAL as a fixed prior (LightGBM init_score), boosted on emerging data only: the emerging data corrects
          the developed structure only where it has enough evidence (a boosting analogue of credibility).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from final import REG, select_cols
from models import SEV_OVERRIDE
from train import PARAMS

DEV_EXCLUDE = ["region", "gender", "market"]
VARIANTS = ("EM", "DEV_CAL", "POOL", "OFFSET")


def inner_split(df: pd.DataFrame, frac: float = 0.25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = df["policy_id"].unique()
    val_ids = np.random.default_rng(seed).choice(ids, int(len(ids) * frac), replace=False)
    m = df["policy_id"].isin(val_ids)
    return df[~m], df[m]


def _params(kind: str) -> dict:
    p = {**PARAMS, **REG, "objective": "poisson" if kind == "freq" else "gamma"}
    if kind == "sev":
        p.update(SEV_OVERRIDE)
    p.pop("tweedie_variance_power", None)
    return p


def _xy(df: pd.DataFrame, kind: str):
    cc = df["claim_count"].to_numpy().astype(float)
    if kind == "freq":
        w = (df["exposure"] * df["dev_f"]).to_numpy()
        return np.ones(len(df), bool), cc / w, w
    mask = (cc > 0) & (df["net_capped"].to_numpy() > 0)
    return mask, np.where(mask, df["net_capped"].to_numpy() / np.maximum(cc, 1), 1.0), cc


def _x(frames: list[pd.DataFrame], cols: list[str]) -> list[pd.DataFrame]:
    """Model matrices with category levels shared across all frames."""
    out = [f[cols].copy() for f in frames]
    for c in cols:
        s0 = out[0][c]
        if s0.dtype == object or str(s0.dtype) in ("bool", "boolean", "string", "category"):
            cats = pd.Index(pd.concat([o[c].astype(str) for o in out]).unique())
            for o in out:
                o[c] = pd.Categorical(o[c].astype(str), categories=cats)
    return out


def fit_rate(kind: str, train: pd.DataFrame, cols: list[str], apply: list[pd.DataFrame] | None = None,
             val: pd.DataFrame | None = None, rounds: int | None = None, seeds=(0,), base: dict | None = None):
    """rounds=None: early-stop on `val`, return (None, best_iter). Else fit `rounds` per seed and return
    (list of rate predictions for `apply`, rounds). `base` = prior rates {'train','val','apply':[..]} used as offset."""
    apply = apply or []
    frames = [train] + ([val] if rounds is None else []) + apply
    X = _x(frames, cols)
    m, y, w = _xy(train, kind)
    ds = lgb.Dataset(X[0][m], y[m], weight=w[m], init_score=np.log(base["train"][m]) if base else None, free_raw_data=False)
    prm = _params(kind)
    if rounds is None:
        vm, yv, wv = _xy(val, kind)
        if vm.sum() == 0 or wv[vm].sum() == 0:  # e.g. a rare peril with no claims in the validation split
            return None, 1
        dv = lgb.Dataset(X[1][vm], yv[vm], weight=wv[vm], init_score=np.log(base["val"][vm]) if base else None, reference=ds)
        mdl = lgb.train({**prm, "seed": seeds[0]}, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        return None, max(int(mdl.best_iteration), 1)
    preds = [np.zeros(len(a)) for a in apply]
    for s in seeds:
        mdl = lgb.train({**prm, "seed": int(s), "bagging_seed": int(s), "feature_fraction_seed": int(s)}, ds, rounds)
        for i, Xa in enumerate(X[1:]):
            r = mdl.predict(Xa, raw_score=True) + (np.log(base["apply"][i]) if base else 0.0)
            preds[i] += np.exp(r) / len(seeds)
    return preds, rounds


def two_step(kind, fit, val, full, cols, apply, seeds, base_fit=None, base_val=None, base_full=None, base_apply=None):
    """Early-stop on (fit, val), refit on `full` with 1.15x rounds; returns (predictions for apply, rounds)."""
    es_base = {"train": base_fit, "val": base_val} if base_fit is not None else None
    _, it = fit_rate(kind, fit, cols, val=val, seeds=seeds, base=es_base)
    rb = {"train": base_full, "apply": base_apply} if base_full is not None else None
    preds, r = fit_rate(kind, full, cols, apply=apply, rounds=max(int(it * 1.15), 1), seeds=seeds, base=rb)
    return preds, it


def _cols(df: pd.DataFrame, exclude=(), extra=()) -> list[str]:
    return [c for c in select_cols(df) if c not in exclude and c not in extra] + list(extra)


def predict_variants(dev: pd.DataFrame, em: pd.DataFrame, targets: list[pd.DataFrame], cutoff: int,
                     variants=VARIANTS, seeds=(0,), log=print, extra=()) -> tuple[dict, dict]:
    """Fit every variant on developed rows (years <= cutoff) + emerging rows `em`, predict rates for `targets`.
    `extra`: additional feature columns (e.g. pricing features) appended to every variant's feature set.
    Returns ({variant: [rate per target]}, info)."""
    dev = dev[dev["year"] <= cutoff]
    em_a, em_b = inner_split(em)
    out, info = {}, {}

    if {"DEV_CAL", "OFFSET"} & set(variants):
        cols = _cols(dev, DEV_EXCLUDE, extra)
        d_fit, d_val = dev[dev["year"] < cutoff], dev[dev["year"] == cutoff]
        apply = [em_a, em_b, em] + targets
        dv = {}
        for kind in ("freq", "sev"):
            dv[kind], it = two_step(kind, d_fit, d_val, dev, cols, apply, seeds)
            info[f"DEV_{kind}_iters"] = it
        # emerging actual / expected on all emerging training rows (claim count, then cost per claim; uncapped)
        e_w = (em["exposure"] * em["dev_f"]).to_numpy()
        f_freq = em["claim_count"].sum() / (dv["freq"][2] * e_w).sum()
        den_sev = (dv["sev"][2] * em["claim_count"].to_numpy()).sum()
        f_sev = em["net_incurred"].sum() / den_sev if den_sev > 0 and em["net_incurred"].sum() > 0 else 1.0
        info.update(DEV_cal_freq=round(float(f_freq), 4), DEV_cal_sev=round(float(f_sev), 4))
        cal = {"freq": [p * f_freq for p in dv["freq"]], "sev": [p * f_sev for p in dv["sev"]]}
        out["DEV_CAL"] = [cal["freq"][3 + i] * cal["sev"][3 + i] for i in range(len(targets))]
        log(f"   DEV done iters={info['DEV_freq_iters']}/{info['DEV_sev_iters']} cal freq={f_freq:.3f} sev={f_sev:.3f}")

    if "EM" in variants:
        cols = _cols(em, extra=extra)
        r = {}
        for kind in ("freq", "sev"):
            r[kind], info[f"EM_{kind}_iters"] = two_step(kind, em_a, em_b, em, cols, targets, seeds)
        out["EM"] = [r["freq"][i] * r["sev"][i] for i in range(len(targets))]
        log(f"   EM done iters={info['EM_freq_iters']}/{info['EM_sev_iters']}")

    if "POOL" in variants:
        pool = pd.concat([dev, em], ignore_index=True)
        cols = _cols(pool, extra=extra)
        r = {}
        for kind in ("freq", "sev"):
            r[kind], info[f"POOL_{kind}_iters"] = two_step(kind, pd.concat([dev, em_a], ignore_index=True), em_b, pool,
                                                           cols, targets, seeds)
        out["POOL"] = [r["freq"][i] * r["sev"][i] for i in range(len(targets))]
        log(f"   POOL done iters={info['POOL_freq_iters']}/{info['POOL_sev_iters']}")

    if "OFFSET" in variants:
        cols = _cols(em, extra=extra)
        r = {}
        for kind in ("freq", "sev"):
            b = cal[kind]
            r[kind], info[f"OFFSET_{kind}_iters"] = two_step(kind, em_a, em_b, em, cols, targets, seeds,
                                                             base_fit=b[0], base_val=b[1], base_full=b[2], base_apply=b[3:])
        out["OFFSET"] = [r["freq"][i] * r["sev"][i] for i in range(len(targets))]
        log(f"   OFFSET done iters={info['OFFSET_freq_iters']}/{info['OFFSET_sev_iters']}")
    return out, info
