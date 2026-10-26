"""Part 2 submission: emerging-market 2025 expected claim cost per row.

    python p2_final.py --blend POOL,OFFSET --seeds 5 --tag p2

Per line: developed (2021-24) + emerging (2023-24) frames on a developed basis -> the variants in --blend (p2_models.py),
equal-weight average of their rates -> x optional trend -> x exposure. Guardrails: exposure check, leak screen,
drift screen (emerging 2024 vs 2025 test), submission validity, level sanity vs emerging developed history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from final import OUT
from guardrails import drift_screen, exposure_check, leak_screen, prediction_sanity, validate_submission
from models import SEEDS
from p2_features import P2, dev_claims, get_frames
from p2_models import _cols, predict_variants
from trend import factor as trend_factor, yearly


def main(a) -> None:
    curves = reported_fraction_curves(dev_claims())
    blend = a.blend.split(",")
    variants = tuple(set(blend) | ({"DEV_CAL"} if "OFFSET" in blend else set()))
    report, preds = {}, []
    for ln in ("motor", "buildings", "contents"):
        dev = add_development(get_frames("developed", ln)[0], curves[ln])
        em, test = get_frames("emerging", ln)
        em = add_development(em, curves[ln])
        cols = _cols(em)
        expo = {"train": exposure_check(em), "test": exposure_check(test)}
        assert not expo["train"].get("above_one") and not expo["test"].get("above_one"), f"{ln}: bad exposure {expo}"
        leaks = leak_screen(em, cols)
        assert not leaks, f"{ln}: possible leak(s) {leaks}"
        drift = drift_screen(em[em["year"] == 2024], test, cols)
        rates, info = predict_variants(dev, em, [test], 2024, variants=variants, seeds=SEEDS[: a.seeds],
                                       log=lambda s: print(f"{ln} {s}", flush=True))
        rate = np.mean([rates[v][0] for v in blend], axis=0)
        # developed-market cost trend (freq + sev slope), damped, applied from the emerging experience mid-point
        tf = 1.0
        if a.trend > 0:
            g = yearly(dev)
            s = trend_factor(g, 2025, "freqsev", 1.0)[1]
            ybar = float(np.average(em["year"], weights=em["exposure"] * em["dev_f"]))
            tf = float(np.exp(a.trend * s * (2025 - ybar)))
        pred = rate * tf * test["exposure"].to_numpy()
        preds.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred}))
        hist = em.groupby("year").apply(lambda g: g["net_incurred"].sum() / (g["exposure"] * g["dev_f"]).sum(), include_groups=False)
        nb = test["new_business"].to_numpy().astype(bool)
        report[ln] = {
            "blend": blend, "seeds": a.seeds, "trend_damp": a.trend, "trend_factor": round(tf, 4), **info,
            "pred_cost_per_expo_2025": round(float(pred.sum() / test["exposure"].sum()), 2),
            "pred_cpe_new_business": round(float(pred[nb].sum() / test.loc[nb, "exposure"].sum()), 2),
            "pred_cpe_renewal": round(float(pred[~nb].sum() / test.loc[~nb, "exposure"].sum()), 2),
            "hist_developed_cost_per_expo": {int(k): round(float(v), 2) for k, v in hist.items()},
            "drift_flags": drift, "exposure_issues": expo, "n_features": len(cols),
        }
        report[ln]["sanity_ok"] = prediction_sanity(report[ln]["pred_cost_per_expo_2025"], report[ln]["hist_developed_cost_per_expo"])
        print(ln, json.dumps(report[ln]), flush=True)
        if not report[ln]["sanity_ok"]:
            print(f"WARNING {ln}: predicted level outside the sanity band of emerging history - human review required", flush=True)

    sub = pd.concat(preds, ignore_index=True)
    ss = pd.read_parquet(P2 / "emerging" / "sample_submission.parquet")
    sub = ss[["row_id"]].merge(sub, on="row_id", how="left")
    errs = validate_submission(sub, ss)
    assert not errs, errs
    sub.to_csv(OUT / f"submission_{a.tag}.csv", index=False)
    (OUT / f"report_{a.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote submission_{a.tag}.csv rows={len(sub)} mean={sub['prediction'].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default="POOL,OFFSET", help="comma-separated variants averaged with equal weight")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trend", type=float, default=0.0, help="damping of the developed-market cost trend (0 = off)")
    ap.add_argument("--tag", default="p2")
    main(ap.parse_args())
