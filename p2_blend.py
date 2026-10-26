"""Part 2: compare single variants and simple blends on the saved held-out predictions (both validation schemes).

Reads data/cache_p2/hold_{scheme}_{line}.parquet written by p2_eval.py. Blends are fixed equal weights (no fitting on the
holdout). D2_cap scores against claims capped at the line's 99.9th percentile - a robustness view showing whether a
ranking is driven by one or two very large claims (e.g. the 136,767 new-business motor claim in the group holdout).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train import d2

CACHE = Path(__file__).parent / "data" / "cache_p2"
BLENDS = {
    "EM": ["EM"], "DEV_CAL": ["DEV_CAL"], "POOL": ["POOL"], "OFFSET": ["OFFSET"],
    "POOL+OFFSET": ["POOL", "OFFSET"], "DEV+POOL+OFFSET": ["DEV_CAL", "POOL", "OFFSET"],
}


def main() -> None:
    rows = []
    for scheme in ("time", "group"):
        for ln in ("contents", "buildings", "motor"):
            h = pd.read_parquet(CACHE / f"hold_{scheme}_{ln}.parquet")
            y, yc, nb = h["net_incurred"].to_numpy(), h["net_capped"].to_numpy(), h["new_business"].to_numpy().astype(bool)
            for name, parts in BLENDS.items():
                p = np.mean([h[f"p_{v}"].to_numpy() for v in parts], axis=0)
                rows.append({"scheme": scheme, "line": ln, "model": name, "D2": d2(y, p), "AE": y.sum() / p.sum(),
                             "D2_oracle": d2(y, p * y.sum() / p.sum()), "D2_cap": d2(yc, p * yc.sum() / p.sum()),
                             "NB_D2": d2(y[nb], p[nb]), "NB_AE": y[nb].sum() / p[nb].sum()})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for ln, g in df.groupby("line", sort=False):
        w = g.pivot(index="model", columns="scheme", values=["D2", "D2_oracle", "D2_cap", "AE", "NB_AE"]).round(4)
        w[("D2", "min")] = w["D2"].min(axis=1)
        print(f"\n== {ln}\n{w.sort_values(('D2', 'min'), ascending=False).to_string()}")
    df.round(4).to_csv(Path(__file__).parent / "p2_blend_results.csv", index=False)


if __name__ == "__main__":
    main()
