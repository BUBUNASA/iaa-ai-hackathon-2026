"""Part 2 validation: which way of using the developed market best predicts the emerging market?

  python p2_eval.py --scheme time  [lines]   # train dev<=2023 + emerging 2023, score emerging 2024 (honest in time)
  python p2_eval.py --scheme group [lines]   # hold out 30% of emerging policies (all years), dev<=2024 + rest

Scored like Part 1: prediction = rate x exposure x reported fraction vs reported net_incurred, Tweedie D2 (p=1.5).
Also reported: A/E, D2 at the oracle level (ranking only - diagnostic, not a selectable quantity), the same on the
new-business subset (76% of the 2025 test), and a paired bootstrap of the D2 gain vs EM (P(better)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from p2_features import dev_claims, get_frames
from p2_models import VARIANTS, predict_variants
from train import d2

ROOT = Path(__file__).parent
N_BOOT = 300


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"D2": d2(y, p), "AE": float(y.sum() / p.sum()), "D2_oracle": d2(y, p * y.sum() / p.sum())}


def boot_gain(y, p, p_ref, rng) -> float:
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    return float(np.mean([d2(y[i], p[i]) > d2(y[i], p_ref[i]) for i in idx]))


def run(scheme: str, lines: list[str]) -> None:
    curves = reported_fraction_curves(dev_claims())
    rows, rng = [], np.random.default_rng(1)
    for ln in lines:
        dev = add_development(get_frames("developed", ln)[0], curves[ln])
        em = add_development(get_frames("emerging", ln)[0], curves[ln])
        if scheme == "time":
            fit, hold, cutoff = em[em["year"] == 2023], em[em["year"] == 2024], 2023
        else:
            ids = em["policy_id"].unique()
            hid = np.random.default_rng(7).choice(ids, int(0.3 * len(ids)), replace=False)
            fit, hold, cutoff = em[~em["policy_id"].isin(hid)], em[em["policy_id"].isin(hid)], 2024
        print(f"\n===== {ln} scheme={scheme} fit={len(fit):,} hold={len(hold):,} hold claims={int(hold['claim_count'].sum())}", flush=True)
        rates, info = predict_variants(dev, fit, [hold, fit], cutoff, log=lambda s: print(s, flush=True))
        y = hold["net_incurred"].to_numpy()
        nb = hold["new_business"].to_numpy().astype(bool)
        e = (hold["exposure"] * hold["dev_f"]).to_numpy()
        preds = {v: rates[v][0] * e for v in VARIANTS if v in rates}
        keep = ["row_id", "policy_id", "year", "new_business", "exposure", "dev_f", "claim_count", "net_incurred", "net_capped"]
        hold[keep].assign(**{f"p_{v}": p for v, p in preds.items()}).to_parquet(ROOT / "data" / "cache_p2" / f"hold_{scheme}_{ln}.parquet")
        # in-sample predictions on the fit rows: used (only) to estimate adjustments such as a new-business factor
        e_fit = (fit["exposure"] * fit["dev_f"]).to_numpy()
        fit[keep].assign(**{f"p_{v}": rates[v][1] * e_fit for v in VARIANTS if v in rates}).to_parquet(
            ROOT / "data" / "cache_p2" / f"fit_{scheme}_{ln}.parquet")
        for v, p in preds.items():
            a, b = metrics(y, p), metrics(y[nb], p[nb])
            pb = boot_gain(y, p, preds["EM"], rng) if v != "EM" else np.nan
            rows.append({"scheme": scheme, "line": ln, "variant": v, **{k: round(x, 4) for k, x in a.items()},
                         **{f"NB_{k}": round(x, 4) for k, x in b.items()}, "P_beats_EM": pb})
            print(f"{ln:9s} {v:8s} D2={a['D2']:.4f} A/E={a['AE']:.3f} D2orc={a['D2_oracle']:.4f} | "
                  f"NB D2={b['D2']:.4f} A/E={b['AE']:.3f} D2orc={b['D2_oracle']:.4f} | P(>EM)={pb:.2f}", flush=True)
        print("   info", json.dumps(info), flush=True)
    out = ROOT / f"p2_eval_{scheme}_results.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old["line"].isin(lines)], df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["time", "group"], default="time")
    ap.add_argument("lines", nargs="*", default=["contents", "buildings", "motor"])
    a = ap.parse_args()
    run(a.scheme, a.lines)
