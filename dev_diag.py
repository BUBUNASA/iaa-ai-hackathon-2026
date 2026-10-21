"""Diagnose why dev-weighting lowers reported-basis D2: level vs within-year shape. Motor + buildings, val 2024."""
import numpy as np, pandas as pd, lightgbm as lgb
from dev import add_development, reported_fraction_curves
from exp import get_frames
from features import to_model_frame
from final import REG, select_cols
from train import PARAMS, d2, fit_eval

curves = reported_fraction_curves()
for ln in ("motor", "buildings"):
    train, _ = get_frames(ln)
    train = add_development(train, curves[ln])
    cols = select_cols(train)
    r = fit_eval(train, cols, params=REG, val_year=2024)
    X, _ = to_model_frame(train, train.head(1), cols)
    v = (train["year"] == 2024).to_numpy()
    tv = train[v].copy()
    tv["rate"] = r["model"].predict(X[v], num_iteration=r["best_iter"])
    tv["pred"] = tv["rate"] * tv["exposure"]
    y = tv["net_incurred"].to_numpy()
    print(f"\n=== {ln} (baseline model, 2023- fit)  base D2={d2(y, tv['pred']):.4f} A/E={y.sum()/tv['pred'].sum():.3f}")
    for a in (0, 0.5, 1.0):
        p = tv["pred"] * tv["dev_f"] ** a
        print(f"  pred*f^{a}: D2={d2(y, p):.4f}")
    for k in (0.9, 1.0, 1.1, 1.2, 1.3):
        print(f"  pred*{k}: D2={d2(y, tv['pred']*k):.4f}")
    tv["m"] = tv["period_start"].dt.month
    g = tv.groupby("m").apply(lambda d: pd.Series({
        "expo": d["exposure"].sum(), "f": np.average(d["dev_f"], weights=d["exposure"]),
        "act/expo": d["net_incurred"].sum() / d["exposure"].sum(), "pred/expo": d["pred"].sum() / d["exposure"].sum(),
        "cnt/expo": d["claim_count"].sum() / d["exposure"].sum()}), include_groups=False)
    print(g.round(3).to_string())
