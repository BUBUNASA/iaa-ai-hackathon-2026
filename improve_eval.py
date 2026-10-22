"""Candidate model improvements, judged on the 2023 AND 2024 holdouts (same basis as the final model: freq x sev,
developed exposure, holdout-safe trend factor, scored on the reported basis).

    python improve_eval.py motor buildings contents

Variants
  base       current final model (single seed)
  no_wx      weather columns dropped
  loose      less regularised trees (num_leaves 31, min_data_in_leaf 500)
  make_cred  motor only: credibility-weighted claim frequency / severity relativity per make_model (out-of-fold by year)
  peril      separate freq x sev model per peril, summed (each peril trained on its own developed exposure)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import models
from dev import MAX_AGE, VAL, _peril_cdf, add_development, reported_fraction_curves, row_reported_fraction
from exp import get_frames
from features import WEATHER_COLS
from models import fit_predict, select_cols
from perils import peril_curves, peril_targets
from prep import load_claims
from train import d2
from trend import factor, yearly

K_FREQ, K_SEV = 500.0, 200.0   # credibility constants: exposure-years, claims


def make_cred(fit: pd.DataFrame, apply_to: pd.DataFrame, oof: bool) -> pd.DataFrame:
    """Credibility-weighted frequency / severity relativity for make_model, from `fit` only."""
    def enc(src: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
        eff = src["exposure"] * src["dev_f"]
        g = src.assign(eff=eff).groupby("make_model").agg(eff=("eff", "sum"), n=("claim_count", "sum"), c=("net_capped", "sum"))
        f0, s0 = g["n"].sum() / g["eff"].sum(), g["c"].sum() / g["n"].sum()
        fr = ((g["n"] + K_FREQ * f0) / (g["eff"] + K_FREQ)) / f0
        sv = ((g["c"] + K_SEV * s0) / (g["n"] + K_SEV)) / s0
        return pd.DataFrame({"mm_freq_rel": tgt["make_model"].map(fr).fillna(1.0).to_numpy(),
                             "mm_sev_rel": tgt["make_model"].map(sv).fillna(1.0).to_numpy()}, index=tgt.index)
    if not oof:
        return enc(fit, apply_to)
    parts = [enc(fit[fit["year"] != y], fit[fit["year"] == y]) for y in sorted(fit["year"].unique())]
    return pd.concat(parts).reindex(fit.index)


def fs(fit, val, cols, seeds=(0,)):
    fr, _ = fit_predict(fit, val, cols, "freq", seeds=seeds, val=val)
    sv, _ = fit_predict(fit, val, cols, "sev", seeds=seeds, val=val)
    return fr * sv


def run_line(line: str) -> list[dict]:
    claims = load_claims()
    train, _ = get_frames(line)
    train = add_development(train, reported_fraction_curves()[line])
    base_cols = select_cols(train)
    pcurves = peril_curves(claims, line)
    ptg = peril_targets(train, claims, line)
    rows = []
    for vy in (2023, 2024):
        fit, val = train[train["year"] < vy].copy(), train[train["year"] == vy].copy()
        y, f, expo = val["net_incurred"].to_numpy(), val["dev_f"].to_numpy(), val["exposure"].to_numpy()
        tf = factor(yearly(fit), vy, "freqsev", 0.75)[0]
        preds = {}
        preds["base"] = fs(fit, val, base_cols)
        preds["no_wx"] = fs(fit, val, [c for c in base_cols if c not in WEATHER_COLS])
        reg0 = dict(models.REG)
        models.REG.update(num_leaves=31, min_data_in_leaf=500)
        try:
            preds["loose"] = fs(fit, val, base_cols)
        finally:
            models.REG.clear(); models.REG.update(reg0)
        if line == "motor":
            fit2, val2 = fit.join(make_cred(fit, fit, oof=True)), val.join(make_cred(fit, val, oof=False))
            preds["make_cred"] = fs(fit2, val2, base_cols + ["mm_freq_rel", "mm_sev_rel"])
        tot = np.zeros(len(val))
        for p, curve in pcurves.items():
            t = ptg[ptg["peril"] == p].set_index("row_id")
            fp, vp = fit.copy(), val.copy()
            for d in (fp, vp):
                d["claim_count"] = d["row_id"].map(t["n"]).fillna(0).to_numpy()
                d["net_capped"] = d["row_id"].map(t["c"]).fillna(0).to_numpy()
            fp["dev_f"] = row_reported_fraction(fp, curve)
            vp["dev_f"] = row_reported_fraction(vp, curve)
            tot += fs(fp, vp, base_cols)
            print(f"  {line} {vy} peril {p} done", flush=True)
        preds["peril"] = tot
        for name, rate in preds.items():
            p = rate * expo * tf * f
            rows.append(dict(line=line, vy=vy, variant=name, D2=d2(y, p), AE=y.sum() / p.sum()))
            print(f"{line} {vy} {name:10s} D2={rows[-1]['D2']:.4f} A/E={rows[-1]['AE']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    out = []
    for ln in sys.argv[1:] or ["motor", "buildings", "contents"]:
        out += run_line(ln)
        pd.DataFrame(out).to_csv("improve_eval_results.csv", index=False)
    res = pd.DataFrame(out)
    print(res.pivot_table(index=["line", "variant"], columns="vy", values="D2").round(4).to_string())
