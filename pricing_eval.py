"""Does adding the insurer's own pricing (written_premium, discount_applied, premium per exposure) improve the model?

Judged on the 2023 AND 2024 holdouts, same basis as the final model (developed exposure, holdout-safe trend x0.75,
scored on the reported basis). Also runs the leak and drift screens on the new columns.

    python pricing_eval.py motor buildings contents
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from guardrails import drift_screen, leak_screen
from improve_eval import fs
from models import select_cols
from perils import peril_curves, peril_frame, peril_targets
from prep import load_claims
from train import d2
from trend import factor, yearly

PRICE = ["written_premium", "discount_applied", "prem_rate"]


REL = ["prem_rel", "prem_rate_rel", "discount_applied"]


def add_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(prem_rate=df["written_premium"] / df["exposure"].clip(lower=1e-6))
    # on-level: premium relative to its own policy year's mean, so rate increases don't leak a level shift into the trees
    yr = df.groupby("year") if "year" in df else None
    for c, r in (("written_premium", "prem_rel"), ("prem_rate", "prem_rate_rel")):
        df[r] = df[c] / (yr[c].transform("mean") if yr is not None else df[c].mean())
    return df


def run_rel(line: str) -> list[dict]:
    """Line-level: raw pricing, and raw pricing re-levelled to the base model's total (relativities from the
    pricing model, level from the validated base + trend; uses no holdout targets)."""
    train, _ = get_frames(line)
    train = add_price(add_development(train, reported_fraction_curves()[line]))
    base = [c for c in select_cols(train) if c not in PRICE + REL]
    rows = []
    for vy in (2023, 2024):
        fit, val = train[train["year"] < vy], train[train["year"] == vy]
        y, f, expo = val["net_incurred"].to_numpy(), val["dev_f"].to_numpy(), val["exposure"].to_numpy()
        tf = factor(yearly(fit), vy, "freqsev", 0.75)[0]
        pb, pp = fs(fit, val, base) * expo * tf * f, fs(fit, val, base + PRICE) * expo * tf * f
        for name, p in (("base", pb), ("pricing", pp), ("pricing_anchor", pp * pb.sum() / pp.sum())):
            rows.append(dict(line=line, vy=vy, variant=name, D2=d2(y, p), AE=y.sum() / p.sum()))
            print(f"{line} {vy} {name:14s} D2={rows[-1]['D2']:.4f} A/E={rows[-1]['AE']:.3f}", flush=True)
    return rows


def run_line(line: str, claims: pd.DataFrame) -> list[dict]:
    train, test = get_frames(line)
    train, test = add_price(add_development(train, reported_fraction_curves()[line])), add_price(test)
    base = [c for c in select_cols(train) if c not in PRICE + REL]
    priced = base + PRICE
    latest = train[train["year"] == train["year"].max()]
    print(f"== {line} leak screen (corr with same-row claim count): {leak_screen(train, PRICE)}")
    print(f"== {line} drift 2024 -> 2025: {drift_screen(latest, test, PRICE)}")
    for c in PRICE:
        print(f"   {c}: corr with claim_count={np.corrcoef(train[c], train['claim_count'])[0,1]:.3f}; "
              f"mean by year={train.groupby('year')[c].mean().round(2).to_dict()} test={test[c].mean():.2f}")
    per_peril = line in ("buildings", "contents")
    curves, ptg = (peril_curves(claims, line), peril_targets(train, claims, line)) if per_peril else ({}, None)
    rows = []
    for vy in (2023, 2024):
        fit, val = train[train["year"] < vy], train[train["year"] == vy]
        y, f, expo = val["net_incurred"].to_numpy(), val["dev_f"].to_numpy(), val["exposure"].to_numpy()
        tf = factor(yearly(fit), vy, "freqsev", 0.75)[0]
        preds = {"base": fs(fit, val, base), "pricing": fs(fit, val, priced)}
        if per_peril:
            for name, cols in (("peril", base), ("peril_pricing", priced)):
                tot = np.zeros(len(val))
                for p, curve in curves.items():
                    t = ptg[ptg["peril"] == p].set_index("row_id")
                    tot += fs(peril_frame(fit, t, curve), peril_frame(val, t, curve), cols)
                preds[name] = tot
        for name, rate in preds.items():
            p = rate * expo * tf * f
            rows.append(dict(line=line, vy=vy, variant=name, D2=d2(y, p), AE=y.sum() / p.sum()))
            print(f"{line} {vy} {name:14s} D2={rows[-1]['D2']:.4f} A/E={rows[-1]['AE']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rel"]:
        out = sum((run_rel(ln) for ln in sys.argv[2:] or ["motor", "buildings", "contents"]), [])
        pd.DataFrame(out).to_csv("pricing_rel_results.csv", index=False)
        sys.exit(0)
    claims = load_claims()
    out = []
    for ln in sys.argv[1:] or ["motor", "buildings", "contents"]:
        out += run_line(ln, claims)
        pd.DataFrame(out).to_csv("pricing_eval_results.csv", index=False)
    print(pd.DataFrame(out).pivot_table(index=["line", "variant"], columns="vy", values="D2").round(4).to_string())
