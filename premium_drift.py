"""Did the insurer's rating plan change in 2025? If premium is a proxy for the insurer's risk view, the relationship
premium ~ risk features should be stable. Fit log(written_premium) on the base risk features using 2022-2023,
then compare fit quality on 2024 (in-period) vs 2025 (test). A big drop on 2025 = repricing -> premium's meaning drifted.

    python premium_drift.py
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from exp import get_frames
from features import to_model_frame
from models import select_cols
from pricing_eval import PRICE, REL, add_price

for ln in ("motor", "buildings", "contents"):
    train, test = get_frames(ln)
    train, test = add_price(train), add_price(test)
    cols = [c for c in select_cols(train) if c not in PRICE + REL]
    fit = train[train["year"].isin([2022, 2023])]
    h24 = train[train["year"] == 2024]
    X, X24 = to_model_frame(fit, h24, cols)
    _, X25 = to_model_frame(fit, test, cols)
    y = np.log(fit["written_premium"])
    m = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, verbose=-1),
                  lgb.Dataset(X, y), 600)
    print(f"== {ln}: top features {pd.Series(m.feature_importance('gain'), cols).nlargest(6).round(0).to_dict()}")
    for name, d, Xd in (("fit 22-23", fit, X), ("2024", h24, X24), ("2025", test, X25)):
        r = np.log(d["written_premium"]) - m.predict(Xd)
        r2 = 1 - r.var() / np.log(d["written_premium"]).var()
        print(f"   {name:9s} R2={r2:.3f} mean resid={r.mean():+.3f} sd resid={r.std():.3f} "
              f"(expected mean ~+{0.04 * (d['year'].iloc[0] - 2022.5) if 'year' in d else 0.04 * 2.5:.2f} from inflation)")
    # exposure-band sanity for pricing_diag's odd table: does pd.cut drop rows?
    band = pd.cut(h24["exposure"], [0, 0.25, 0.5, 0.99, 1.01])
    print(f"   2024 rows outside expo bands: {band.isna().sum()} of {len(h24)}; exposure range {h24['exposure'].min():.4f}-{h24['exposure'].max():.4f}")
