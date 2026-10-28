"""Part 2, idea 3: insurer pricing features (written_premium, discount_applied, premium per unit exposure) in every transfer variant.

Known at inception and leak/drift-screened in Part 1, but circular: they encode the current tariff, and the emerging book is the
one under review (its charged premium is 2-4x below technical, p2_premium.py). Tested on the same two holdouts as p2_eval.py
against the line-level blend; adopt for a line only if it wins on BOTH holdouts, and disclose the circularity.

    python p2_pricing.py --scheme time motor
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from guardrails import leak_screen
from p2_features import dev_claims, get_frames
from p2_models import predict_variants
from p2_peril import BLEND, CACHE, score
from pricing_eval import PRICE, REL, add_price


def priced(df: pd.DataFrame) -> pd.DataFrame:
    """Raw pricing features only (the year-relative versions were worse in Part 1 and are not tested here)."""
    return add_price(df).drop(columns=[c for c in REL if c not in PRICE])


def run(scheme: str, ln: str) -> None:
    curves = reported_fraction_curves(dev_claims())
    dev = priced(add_development(get_frames("developed", ln)[0], curves[ln]))
    em = priced(add_development(get_frames("emerging", ln)[0], curves[ln]))
    if scheme == "time":
        fit, hold, cutoff = em[em["year"] == 2023], em[em["year"] == 2024], 2023
    else:
        ids = em["policy_id"].unique()
        hid = np.random.default_rng(7).choice(ids, int(0.3 * len(ids)), replace=False)
        fit, hold, cutoff = em[~em["policy_id"].isin(hid)], em[em["policy_id"].isin(hid)], 2024
    leaks = leak_screen(fit, PRICE)
    print(f"\n===== {ln} scheme={scheme} leak screen on pricing cols: {leaks or 'clean'}", flush=True)
    r, info = predict_variants(dev, fit, [hold], cutoff, variants=BLEND, extra=PRICE, log=lambda s: print(s, flush=True))
    e = (hold["exposure"] * hold["dev_f"]).to_numpy()
    out = hold[["row_id"]].assign(**{f"p_{v}": r[v][0] * e for v in BLEND})
    out.to_parquet(CACHE / f"hold_{scheme}_{ln}_pricing.parquet")
    base = pd.read_parquet(CACHE / f"hold_{scheme}_{ln}.parquet").set_index("row_id").loc[hold["row_id"]]
    assert np.allclose(base["net_incurred"].to_numpy(), hold["net_incurred"].to_numpy()), "holdout rows differ from p2_eval"
    rows = {"line blend": base[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy(),
            "pricing blend": out[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy()}
    rows.update({f"pricing {v}": out[f"p_{v}"].to_numpy() for v in BLEND})
    for name, p in rows.items():
        s = score(hold, p)
        print(f"{ln:9s} {scheme:5s} {name:16s} D2={s['D2']:.4f} D2cap={s['D2_cap']:.4f} A/E={s['AE']:.3f} NB A/E={s['NB_AE']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["time", "group"], required=True)
    ap.add_argument("line", choices=["motor", "buildings", "contents"])
    a = ap.parse_args()
    run(a.scheme, a.line)
