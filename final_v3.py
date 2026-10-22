"""Final pipeline (v3): developed-basis Tweedie + frequency x severity blend, seed-bagged, optional large-loss load.

    python final_v3.py --seeds 5 --w-fs 0.5 --load cap --tag final

Writes outputs/submission_<tag>.csv, outputs/report_<tag>.json and runs the guardrail validators before saving.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from final import OUT
from guardrails import drift_screen, leak_screen, prediction_sanity, validate_submission
from models import SEEDS, fit_predict, select_cols
from perils import peril_fs_predict
from prep import LINES, load_claims, read
from trend import factor as trend_factor, yearly


def main(a) -> None:
    curves = reported_fraction_curves()
    w_fs = json.loads(a.w_fs) if a.w_fs.startswith("{") else {ln: float(a.w_fs) for ln in LINES}
    seeds = SEEDS[: a.seeds]
    report, preds = {}, []
    for ln in LINES:
        train, test = get_frames(ln)
        train = add_development(train, curves[ln])
        cols = select_cols(train)
        # --- guardrails on the feature set ---
        leaks = leak_screen(train, cols)
        drift = drift_screen(train[train["year"] == train["year"].max()], test, cols)
        assert not leaks, f"{ln}: possible leak(s) {leaks}"
        # --- iteration counts: early-stop once on the latest year (seed 0), reuse for refit on all years ---
        fit, val = train[train["year"] < 2024], train[train["year"] == 2024]
        it_tw = it_f = it_s = None
        rate = np.zeros(len(test))
        if w_fs[ln] < 1.0:
            _, it_tw = fit_predict(fit, val, cols, "tweedie", seeds=(0,), val=val)
            tw, _ = fit_predict(train, test, cols, "tweedie", seeds=seeds, rounds=int(it_tw * 1.15))
            rate += (1 - w_fs[ln]) * tw
        if ln in a.peril_lines:
            assert w_fs[ln] == 1.0, "per-peril model is freq x sev only"
            rate, it_f = peril_fs_predict(train, test, cols, load_claims(), ln, seeds)
        elif w_fs[ln] > 0:
            _, it_f = fit_predict(fit, val, cols, "freq", seeds=(0,), val=val)
            _, it_s = fit_predict(fit, val, cols, "sev", seeds=(0,), val=val)
            fr, _ = fit_predict(train, test, cols, "freq", seeds=seeds, rounds=int(it_f * 1.15))
            sv, _ = fit_predict(train, test, cols, "sev", seeds=seeds, rounds=int(it_s * 1.15))
            rate += w_fs[ln] * fr * sv
        load = float(train["net_incurred"].sum() / train["net_capped"].sum()) if a.load == "cap" else 1.0
        tf = trend_factor(yearly(train), 2025, "freqsev", a.trend)[0] if a.trend > 0 else 1.0
        pred = rate * load * tf * test["exposure"].to_numpy()
        preds.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred}))
        hist = train.groupby("year").apply(lambda g: g["net_incurred"].sum() / (g["exposure"] * g["dev_f"]).sum(), include_groups=False)
        report[ln] = {
            "seeds": len(seeds), "w_fs": w_fs[ln], "per_peril": ln in a.peril_lines, "large_loss_load": round(load, 4),
            "trend_damp": a.trend, "trend_factor": round(tf, 4),
            "iters": {"tweedie": it_tw, "freq": it_f, "sev": it_s},
            "pred_cost_per_expo_2025": round(float(pred.sum() / test["exposure"].sum()), 2),
            "hist_developed_cost_per_expo": {int(k): round(float(v), 2) for k, v in hist.items()},
            "drift_flags": drift,
        }
        assert prediction_sanity(report[ln]["pred_cost_per_expo_2025"], report[ln]["hist_developed_cost_per_expo"]), f"{ln} level outside sanity band"
        print(ln, json.dumps(report[ln]), flush=True)

    sub = pd.concat(preds, ignore_index=True)
    ss = read("sample_submission")
    sub = ss[["row_id"]].merge(sub, on="row_id", how="left")
    errs = validate_submission(sub, ss)
    assert not errs, errs
    sub.to_csv(OUT / f"submission_{a.tag}.csv", index=False)
    (OUT / f"report_{a.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote submission_{a.tag}.csv rows={len(sub)} mean={sub['prediction'].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--w-fs", default="0.5", help="weight on freq x sev in the blend: a number or a JSON dict per line")
    ap.add_argument("--load", choices=["none", "cap"], default="none")
    ap.add_argument("--trend", type=float, default=0.0, help="damping of the validated freq+sev cost-trend factor (0 = off)")
    ap.add_argument("--peril-lines", nargs="*", default=[], help="lines modelled as a sum of per-peril freq x sev models")
    ap.add_argument("--tag", default="final")
    main(ap.parse_args())
