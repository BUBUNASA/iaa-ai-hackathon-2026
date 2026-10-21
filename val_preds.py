"""Cache validation predictions (ultimate basis) for tweedie / freq x sev, with 1 and 3 seeds, on the 2023 and 2024 holdouts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dev import add_development, reported_fraction_curves
from exp import get_frames
from models import fit_predict, select_cols
from prep import LINES

CACHE = Path(__file__).parent / "data" / "cache"


def main(lines):
    curves = reported_fraction_curves()
    for ln in lines:
        train, _ = get_frames(ln)
        train = add_development(train, curves[ln])
        cols = select_cols(train)
        for vy in (2023, 2024):
            fit, val = train[train["year"] < vy], train[train["year"] == vy]
            out = val[["row_id", "year", "exposure", "dev_f", "claim_count", "net_incurred", "net_capped"]].copy()
            tw1, it_tw = fit_predict(fit, val, cols, "tweedie", seeds=(0,), val=val)
            out["tw1"] = tw1 * out["exposure"]
            tw3, _ = fit_predict(fit, val, cols, "tweedie", seeds=(0, 1, 2), rounds=it_tw)
            out["tw3"] = tw3 * out["exposure"]
            fr, it_f = fit_predict(fit, val, cols, "freq", seeds=(0,), val=val)
            sv, it_s = fit_predict(fit, val, cols, "sev", seeds=(0,), val=val)
            out["fs1"] = fr * sv * out["exposure"]
            out["freq"], out["sev"] = fr, sv
            out.to_parquet(CACHE / f"valpred_{ln}_{vy}.parquet")
            print(f"{ln} {vy} iters tw={it_tw} freq={it_f} sev={it_s}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(LINES))
