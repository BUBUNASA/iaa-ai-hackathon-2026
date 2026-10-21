"""End-to-end: train per-line Tweedie GBMs on 2021-24 and write the Kaggle submission with sanity checks.

    python final.py            # uses cached feature frames in data/cache (built by exp.py / get_frames)
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from exp import get_frames
from features import feature_columns, to_model_frame
from prep import LINES, read
from train import PARAMS, d2, fit_eval

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

REG = dict(num_leaves=15, min_data_in_leaf=1000, lambda_l2=100.0, min_data_per_group=500, cat_l2=50.0,
           cat_smooth=200, max_cat_threshold=16, learning_rate=0.02, feature_fraction=0.6)
HIGH_CARD = 30  # categoricals with more levels than this are dropped (make_model overfits: see exp2_out.txt)


def select_cols(train: pd.DataFrame) -> list[str]:
    cols = feature_columns(train)
    cat = [c for c in cols if train[c].dtype == object or str(train[c].dtype) in ("bool", "category")]
    hi = [c for c in cat if train[c].nunique() > HIGH_CARD]
    return [c for c in cols if c not in hi]


def main() -> None:
    report, preds = {}, []
    for ln in LINES:
        train, test = get_frames(ln)
        cols = select_cols(train)
        val = fit_eval(train, cols, params=REG, val_year=2024)
        rounds = int(val["best_iter"] * 1.15)  # refit on more data -> slightly more trees
        X, Xt = to_model_frame(train, test, cols)
        prm = {**PARAMS, **REG}
        ds = lgb.Dataset(X, train["net_capped"] / train["exposure"], weight=train["exposure"])
        m = lgb.train(prm, ds, rounds)
        rate = m.predict(Xt)
        pred = rate * test["exposure"].to_numpy()
        preds.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred}))
        hist = train.groupby("year").apply(lambda g: g["net_incurred"].sum() / g["exposure"].sum(), include_groups=False)
        report[ln] = {
            "val2024_D2": round(val["d2_uncapped"], 4), "val2024_AE": round(val["A/E"], 3), "rounds": rounds,
            "n_features": len(cols), "pred_cost_per_expo_2025": round(float(pred.sum() / test["exposure"].sum()), 2),
            "hist_cost_per_expo": {int(k): round(float(v), 2) for k, v in hist.items()},
        }
        print(ln, json.dumps(report[ln]), flush=True)

    sub = pd.concat(preds, ignore_index=True)
    ss = read("sample_submission")
    assert set(sub["row_id"]) == set(ss["row_id"]) and len(sub) == len(ss), "row_id mismatch vs sample_submission"
    assert sub["row_id"].is_unique and sub["prediction"].notna().all() and (sub["prediction"] >= 0).all()
    sub = ss[["row_id"]].merge(sub, on="row_id", how="left")
    sub.to_csv(OUT / "submission.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT / 'submission.csv'} rows={len(sub)} mean={sub['prediction'].mean():.2f} max={sub['prediction'].max():.0f}")


if __name__ == "__main__":
    main()
