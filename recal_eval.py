"""Re-validation after the exposure fix (prep.load_policies). Every earlier holdout number was computed on polluted
exposure, so re-decide: (a) pricing features yes/no, (b) trend damping -- using the FINAL procedure on both holdouts:
for holdout Y, early-stop on Y-1 (fit < Y-1), refit on all years < Y with 1.15x rounds, predict Y, seed 0.
Trend factor is estimated only from years < Y.

    python recal_eval.py motor buildings contents
"""
from __future__ import annotations

import sys

import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from models import select_cols
from pricing_diag import final_style
from pricing_eval import PRICE, REL, add_price
from train import d2
from trend import factor, yearly

DAMPS = (0.0, 0.5, 0.75, 1.0)


def run(line: str) -> list[dict]:
    train, _ = get_frames(line)
    train = add_price(add_development(train, reported_fraction_curves()[line]))
    base = [c for c in select_cols(train) if c not in PRICE + REL]
    rows = []
    for vy in (2023, 2024):
        tr, ho = train[train["year"] < vy], train[train["year"] == vy]
        y, f, e = ho["net_incurred"].to_numpy(), ho["dev_f"].to_numpy(), ho["exposure"].to_numpy()
        g = yearly(tr)
        for name, cols in (("base", base), ("pricing", base + PRICE)):
            rate, its = final_style(tr, ho, cols, vy - 1)
            for damp in DAMPS:
                tf = factor(g, vy, "freqsev", damp)[0] if damp > 0 else 1.0
                p = rate * e * f * tf
                rows.append(dict(line=line, vy=vy, variant=name, damp=damp, factor=round(tf, 4), D2=d2(y, p), AE=y.sum() / p.sum()))
                print(f"{line} {vy} {name:8s} damp={damp:<4} f={tf:.3f} D2={rows[-1]['D2']:.4f} A/E={rows[-1]['AE']:.3f}", flush=True)
    g = yearly(train)
    print(f"== {line} 2025 factors: " + ", ".join(f"{d}: {factor(g, 2025, 'freqsev', d)[0]:.4f}" for d in DAMPS[1:]))
    print(g[["cpe", "freq", "sev"]].round(4).to_string())
    return rows


if __name__ == "__main__":
    out = []
    for ln in sys.argv[1:] or ["motor", "buildings", "contents"]:
        out += run(ln)
        pd.DataFrame(out).to_csv("recal_eval_results.csv", index=False)
    res = pd.DataFrame(out)
    print(res.pivot_table(index=["line", "variant", "damp"], columns="vy", values=["D2", "AE"]).round(4).to_string())
