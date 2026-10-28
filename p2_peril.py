"""Part 2, idea 1: per-peril frequency x severity (Part 1's best private version, v5) inside the same transfer variants.

For each peril, the targets (claim_count, net_capped, net_incurred) and the reported fraction dev_f of both markets' training
frames are replaced by that peril's; DEV_CAL / POOL / OFFSET are fitted exactly as at line level (p2_models.predict_variants,
so each peril is recalibrated to the emerging market separately) and the peril rates are summed.
Scored on the SAME holdout rows as p2_eval.py (line-level reported basis) and compared with the line-level blend saved in
data/cache_p2/hold_{scheme}_{line}.parquet. Adopt for a line only if the per-peril blend wins on BOTH holdouts.

    python p2_peril.py --scheme time contents
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import prep
from dev import add_development, reported_fraction_curves, row_reported_fraction
from p2_features import dev_claims, get_frames, use_market
from p2_models import predict_variants
from perils import peril_curves
from train import d2

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache_p2"
BLEND = ("DEV_CAL", "POOL", "OFFSET")
TARGETS = (("claim_count", "n"), ("net_capped", "c"), ("net_incurred", "u"))


def market_claims(market: str) -> pd.DataFrame:
    use_market(market)
    return prep.load_claims()


def peril_targets(frame: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    """Per (row_id, peril): claim count, capped and uncapped net cost - same matching rule and market cap as prep.attach_claims."""
    cl = claims[claims["line"] == line].copy()
    cap = cl.loc[cl["net_incurred"] > 0, "net_incurred"].quantile(0.999)
    cl["net_capped"] = cl["net_incurred"].clip(upper=cap)
    m = cl.merge(frame[["row_id", "policy_id", "period_start", "period_end"]], on="policy_id")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])].drop_duplicates("claim_id")
    return m.groupby(["peril", "row_id"]).agg(n=("claim_id", "size"), c=("net_capped", "sum"), u=("net_incurred", "sum"))


def set_peril(df: pd.DataFrame, t: pd.DataFrame, peril: str, curve: np.ndarray) -> None:
    """In place: replace targets and dev_f by the peril's (avoids copying the ~700k-row developed frame)."""
    tp = t.loc[peril] if peril in t.index.get_level_values(0) else t.iloc[0:0].droplevel(0)
    for col, src in TARGETS:
        df[col] = df["row_id"].map(tp[src]).fillna(0.0).to_numpy()
    df["dev_f"] = row_reported_fraction(df, curve)


def score(h: pd.DataFrame, p: np.ndarray) -> dict:
    y, yc = h["net_incurred"].to_numpy(), h["net_capped"].to_numpy()
    nb = h["new_business"].to_numpy().astype(bool)
    return {"D2": d2(y, p), "D2_cap": d2(yc, p * yc.sum() / y.sum()), "AE": y.sum() / p.sum(), "NB_AE": y[nb].sum() / p[nb].sum()}


def run(scheme: str, ln: str) -> None:
    dclaims = dev_claims()
    line_curves, curves = reported_fraction_curves(dclaims), peril_curves(dclaims, ln)
    dev = get_frames("developed", ln)[0]
    em = add_development(get_frames("emerging", ln)[0], line_curves[ln])
    if scheme == "time":
        fit, hold, cutoff = em[em["year"] == 2023].copy(), em[em["year"] == 2024], 2023
    else:
        ids = em["policy_id"].unique()
        hid = np.random.default_rng(7).choice(ids, int(0.3 * len(ids)), replace=False)
        fit, hold, cutoff = em[~em["policy_id"].isin(hid)].copy(), em[em["policy_id"].isin(hid)], 2024
    t_dev, t_em = peril_targets(dev, dclaims, ln), peril_targets(fit, market_claims("emerging"), ln)
    print(f"\n===== {ln} scheme={scheme} perils={list(curves)} hold={len(hold):,}", flush=True)
    rate = {v: np.zeros(len(hold)) for v in BLEND}
    for peril, curve in curves.items():
        set_peril(dev, t_dev, peril, curve)
        set_peril(fit, t_em, peril, curve)
        r, info = predict_variants(dev, fit, [hold], cutoff, variants=BLEND, log=lambda s: None)
        for v in BLEND:
            rate[v] += r[v][0]
        print(f"   {peril:18s} em claims={int(fit['claim_count'].sum()):5d} cal freq={info['DEV_cal_freq']:.3f} "
              f"sev={info['DEV_cal_sev']:.3f} iters POOL={info['POOL_freq_iters']}/{info['POOL_sev_iters']} "
              f"OFFSET={info['OFFSET_freq_iters']}/{info['OFFSET_sev_iters']}", flush=True)
    e = (hold["exposure"] * hold["dev_f"]).to_numpy()
    out = hold[["row_id"]].assign(**{f"p_{v}": rate[v] * e for v in BLEND})
    out.to_parquet(CACHE / f"hold_{scheme}_{ln}_peril.parquet")
    base = pd.read_parquet(CACHE / f"hold_{scheme}_{ln}.parquet").set_index("row_id").loc[hold["row_id"]]
    assert np.allclose(base["net_incurred"].to_numpy(), hold["net_incurred"].to_numpy()), "holdout rows differ from p2_eval"
    rows = {"line blend": base[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy(),
            "peril blend": out[[f"p_{v}" for v in BLEND]].mean(axis=1).to_numpy()}
    rows.update({f"peril {v}": out[f"p_{v}"].to_numpy() for v in BLEND})
    for name, p in rows.items():
        s = score(hold, p)
        print(f"{ln:9s} {scheme:5s} {name:14s} D2={s['D2']:.4f} D2cap={s['D2_cap']:.4f} A/E={s['AE']:.3f} NB A/E={s['NB_AE']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["time", "group"], required=True)
    ap.add_argument("line", choices=["buildings", "contents"])
    a = ap.parse_args()
    run(a.scheme, a.line)
