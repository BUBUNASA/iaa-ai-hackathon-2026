"""Cached frames + ablation experiments (honest feature set, leak removed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from features import build, feature_columns
from train import fit_eval

CACHE = Path(__file__).parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def get_frames(line: str, refresh: bool = False):
    tr, te = CACHE / f"{line}_train.parquet", CACHE / f"{line}_test.parquet"
    if tr.exists() and te.exists() and not refresh:
        return pd.read_parquet(tr), pd.read_parquet(te)
    train, test, _ = build(line)
    train.to_parquet(tr)
    test.to_parquet(te)
    return train, test


if __name__ == "__main__":
    lines = sys.argv[1:] or ["motor", "buildings", "contents"]
    for ln in lines:
        train, test = get_frames(ln, refresh=True)
        print(f"\n===== {ln} train={train.shape} test={test.shape}", flush=True)
        configs = {
            "all": dict(),
            "no_history": dict(use_history=False),
            "no_weather": dict(use_weather=False),
            "with_pricing": dict(use_pricing=True),
            "with_noise": dict(use_noise=True),
        }
        for name, kw in configs.items():
            cols = feature_columns(train, **kw)
            for vy in (2023, 2024):
                r = fit_eval(train, cols, val_year=vy)
                print(f"{ln:10s} {name:13s} val={vy} D2={r['d2_uncapped']:.4f} A/E={r['A/E']:.3f} "
                      f"iters={r['best_iter']} feats={r['n_features']}", flush=True)
        cols = feature_columns(train)
        r = fit_eval(train, cols, val_year=2024)
        imp = pd.Series(r["model"].feature_importance("gain"), index=cols).sort_values(ascending=False)
        print((imp / imp.sum()).head(15).round(3).to_string(), flush=True)
