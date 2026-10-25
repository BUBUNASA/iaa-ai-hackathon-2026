"""Why did pricing features gain ~+0.02 D2 on holdouts but only +0.0017 on the LB?

1. Final-style backtest: replicate final_v3 exactly with 2024 as the "test" year -- early-stop on 2023 (fit < 2023),
   refit on all years < 2024 with 1.15x rounds, predict 2024. (pricing_eval early-stopped ON the holdout year itself,
   which is optimistic and is not what the final pipeline does.)
2. Same procedure for the real 2025 test (seed 0): where is the pricing model's level uplift vs base concentrated?

    python pricing_diag.py motor buildings contents
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from models import fit_predict, select_cols
from pricing_eval import PRICE, REL, add_price
from train import d2
from trend import factor, yearly


def final_style(train: pd.DataFrame, target: pd.DataFrame, cols: list[str], es_year: int) -> tuple[np.ndarray, dict]:
    """Early-stop on es_year (fit on earlier years), refit on all of `train` with 1.15x rounds, seed 0."""
    fit, val = train[train["year"] < es_year], train[train["year"] == es_year]
    out, its = 1.0, {}
    for kind in ("freq", "sev"):
        _, it = fit_predict(fit, val, cols, kind, seeds=(0,), val=val)
        p, _ = fit_predict(train, target, cols, kind, seeds=(0,), rounds=int(it * 1.15))
        out, its[kind] = out * p, it
    return out, its


def bands(df: pd.DataFrame, ratio: np.ndarray, w: np.ndarray) -> None:
    d = df.assign(ratio=ratio, w=w)
    d["expo_band"] = pd.cut(d["exposure"], [0, 0.25, 0.5, 0.99, 1.01])
    d["prem_decile"] = pd.qcut(d["written_premium"].rank(method="first"), 10, labels=False)
    for g in ("expo_band", "prem_decile", "discount_applied"):
        t = d.groupby(g, observed=True).apply(lambda x: np.average(x["ratio"], weights=x["w"]), include_groups=False)
        print(f"   pricing/base by {g}: {t.round(3).to_dict()}")


def run(line: str) -> list[dict]:
    train, test = get_frames(line)
    train = add_price(add_development(train, reported_fraction_curves()[line]))
    test = add_price(test)
    base = [c for c in select_cols(train) if c not in PRICE + REL]
    priced = base + PRICE
    rows = []
    # 1. final-style backtest on 2024
    tr, ho = train[train["year"] < 2024], train[train["year"] == 2024]
    y, f, e = ho["net_incurred"].to_numpy(), ho["dev_f"].to_numpy(), ho["exposure"].to_numpy()
    tf = factor(yearly(tr), 2024, "freqsev", 0.75)[0]
    pb, ib = final_style(tr, ho, base, 2023)
    pp, ip = final_style(tr, ho, priced, 2023)
    pb, pp = pb * e * tf * f, pp * e * tf * f
    print(f"== {line} final-style backtest 2024: iters base={ib} pricing={ip}; pricing/base total={pp.sum() / pb.sum():.3f}")
    for name, p in (("base", pb), ("pricing", pp), ("pricing_anchor", pp * pb.sum() / pp.sum())):
        rows.append(dict(line=line, test="backtest2024", variant=name, D2=d2(y, p), AE=y.sum() / p.sum()))
        print(f"   {name:15s} D2={rows[-1]['D2']:.4f} A/E={rows[-1]['AE']:.3f}", flush=True)
    bands(ho, pp / pb, pb)
    # 2. real 2025 test
    tb, ib = final_style(train, test, base, 2024)
    tp, ip = final_style(train, test, priced, 2024)
    e = test["exposure"].to_numpy()
    print(f"== {line} 2025 test: iters base={ib} pricing={ip}; pricing/base total={(tp * e).sum() / (tb * e).sum():.3f}; "
          f"rank corr={pd.Series(tp).corr(pd.Series(tb), method='spearman'):.3f}")
    bands(test, tp / tb, tb * e)
    rows.append(dict(line=line, test="test2025", variant="ratio", D2=np.nan, AE=(tp * e).sum() / (tb * e).sum()))
    return rows


if __name__ == "__main__":
    out = []
    for ln in sys.argv[1:] or ["motor", "buildings", "contents"]:
        out += run(ln)
        pd.DataFrame(out).to_csv("pricing_diag_results.csv", index=False)
