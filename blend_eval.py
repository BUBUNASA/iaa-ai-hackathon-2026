"""Decide (on the 2023 AND 2024 holdouts) whether to (a) apply a large-loss load, (b) blend freq x sev with Tweedie, (c) bag seeds.

Scoring is on the reported basis: expected reported cost = ultimate prediction x reported fraction dev_f.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp import get_frames
from prep import LINES
from train import d2
from val_preds import CACHE


def load_ratio(line: str, vy: int) -> float:
    train, _ = get_frames(line)
    t = train[train["year"] < vy]
    return float(t["net_incurred"].sum() / t["net_capped"].sum())


rows = []
for ln in LINES:
    for vy in (2023, 2024):
        v = pd.read_parquet(CACHE / f"valpred_{ln}_{vy}.parquet")
        y, yc, f = v["net_incurred"].to_numpy(), v["net_capped"].to_numpy(), v["dev_f"].to_numpy()
        L = load_ratio(ln, vy)
        for name, base in (("tw1", v["tw1"]), ("tw3", v["tw3"]), ("fs1", v["fs1"])):
            p = base.to_numpy() * f
            rows.append(dict(line=ln, vy=vy, model=name, load=1.0, D2=d2(y, p), AE_unc=y.sum() / p.sum(), AE_cap=yc.sum() / p.sum()))
            rows.append(dict(line=ln, vy=vy, model=name, load=L, D2=d2(y, p * L), AE_unc=y.sum() / (p * L).sum(), AE_cap=yc.sum() / (p * L).sum()))
        for w in (0.25, 0.5, 0.75):
            p = (w * v["fs1"] + (1 - w) * v["tw3"]).to_numpy() * f
            for lo in (1.0, L):
                rows.append(dict(line=ln, vy=vy, model=f"blend_fs{w}", load=lo, D2=d2(y, p * lo),
                                 AE_unc=y.sum() / (p * lo).sum(), AE_cap=yc.sum() / (p * lo).sum()))
        print(f"{ln} {vy}: large-loss load (train unc/capped) = {L:.4f}; "
              f"corr(tw3,fs1)={np.corrcoef(v['tw3'], v['fs1'])[0,1]:.3f}", flush=True)
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for ln in LINES:
    print(f"\n===== {ln}")
    print(res[res.line == ln].pivot_table(index=["model", "load"], columns="vy", values=["D2", "AE_unc", "AE_cap"]).round(4).to_string())
res.to_csv("blend_eval_results.csv", index=False)
