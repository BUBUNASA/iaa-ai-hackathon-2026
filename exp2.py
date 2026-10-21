"""Experiment 2: tame high-cardinality categoricals and over-flexible trees."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from exp import get_frames
from features import feature_columns
from train import fit_eval

REG = dict(num_leaves=15, min_data_in_leaf=1000, lambda_l2=100.0, min_data_per_group=500, cat_l2=50.0,
           cat_smooth=200, max_cat_threshold=16, learning_rate=0.02, feature_fraction=0.6)

for ln in (sys.argv[1:] or ["motor", "buildings", "contents"]):
    train, test = get_frames(ln)
    cols = feature_columns(train)
    cat = [c for c in cols if train[c].dtype == object or str(train[c].dtype) in ("bool", "category")]
    card = {c: int(train[c].nunique()) for c in cat}
    print(f"\n===== {ln} categorical cardinality:", {k: v for k, v in sorted(card.items(), key=lambda x: -x[1])[:6]}, flush=True)
    hi = [c for c, v in card.items() if v > 30]

    def freq_enc(df_tr, df_te, cs):
        a, b = df_tr.copy(), df_te.copy()
        for c in cs:
            vc = a[c].astype(str).value_counts(normalize=True)
            a[c + "_freq"] = a[c].astype(str).map(vc).astype(float)
        return a

    variants = {
        "base": (cols, {}),
        "reg": (cols, REG),
        "reg_drop_hi": ([c for c in cols if c not in hi], REG),
    }
    for name, (cs, prm) in variants.items():
        for vy in (2023, 2024):
            r = fit_eval(train, cs, params=prm, val_year=vy)
            print(f"{ln:10s} {name:12s} val={vy} D2={r['d2_uncapped']:.4f} A/E={r['A/E']:.3f} iters={r['best_iter']} feats={r['n_features']}", flush=True)
    print("high-card cols dropped:", hi, flush=True)
