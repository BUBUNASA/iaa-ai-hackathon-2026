"""final.py variant with the claim-development (reporting-lag) adjustment: models are trained on developed exposure
(exposure x reported fraction), so predicted rates are on an ultimate basis, matching the 2025 target.

    python final_dev.py [--tag dev]    # writes outputs/submission_<tag>.csv and report_<tag>.json
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from dev_exp import fit_eval_dev
from exp import get_frames
from features import to_model_frame
from final import OUT, REG, select_cols
from prep import LINES, read
from train import PARAMS


def main(tag: str) -> None:
    curves = reported_fraction_curves()
    report, preds = {}, []
    for ln in LINES:
        train, test = get_frames(ln)
        train = add_development(train, curves[ln])
        cols = select_cols(train)
        val = fit_eval_dev(train, cols, 2024, dev=True)
        rounds = int(val["iters"] * 1.15)
        X, Xt = to_model_frame(train, test, cols)
        w = (train["exposure"] * train["dev_f"]).to_numpy()
        ds = lgb.Dataset(X, train["net_capped"].to_numpy() / w, weight=w)
        m = lgb.train({**PARAMS, **REG}, ds, rounds)
        pred = m.predict(Xt) * test["exposure"].to_numpy()
        preds.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred}))
        report[ln] = {"rounds": rounds, "val2024_D2_reported_basis": round(val["d2"], 4), "val2024_AE_reported": round(val["AE"], 3),
                      "mean_reported_fraction_2024": round(float(np.average(train.loc[train.year == 2024, "dev_f"],
                                                                            weights=train.loc[train.year == 2024, "exposure"])), 3),
                      "pred_cost_per_expo_2025": round(float(pred.sum() / test["exposure"].sum()), 2)}
        print(ln, json.dumps(report[ln]), flush=True)
    sub = pd.concat(preds, ignore_index=True)
    ss = read("sample_submission")
    assert set(sub["row_id"]) == set(ss["row_id"]) and len(sub) == len(ss)
    assert sub["row_id"].is_unique and sub["prediction"].notna().all() and (sub["prediction"] >= 0).all()
    sub = ss[["row_id"]].merge(sub, on="row_id", how="left")
    sub.to_csv(OUT / f"submission_{tag}.csv", index=False)
    (OUT / f"report_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote submission_{tag}.csv rows={len(sub)} mean={sub['prediction'].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev")
    main(ap.parse_args().tag)
