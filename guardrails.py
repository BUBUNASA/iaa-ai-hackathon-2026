"""Governance guardrails: automated checks run before any submission is accepted.

1. leak_screen   - features that explain the *same-row* outcome too well (|corr with claim_count| high) are suspect.
2. drift_screen  - features whose 2025 test distribution differs sharply from the latest training year.
3. validate_submission - format/ID/finite/non-negative checks against the official sample_submission.
4. prediction_sanity   - portfolio-level predicted cost per exposure vs the developed historical range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_CORR = 0.30        # |Pearson corr| with claim_count above which a feature is flagged as a possible leak
DRIFT_SMD = 0.50        # standardised mean difference (numeric) flag
DRIFT_TVD = 0.15        # total-variation distance (categorical) flag
SANITY_BAND = 0.25      # predicted cost/exposure must sit within +/-25% of the developed historical range


def leak_screen(df: pd.DataFrame, cols: list[str], target: str = "claim_count", thr: float = LEAK_CORR) -> dict[str, float]:
    out = {}
    y = df[target].astype(float)
    for c in cols:
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("bool", "category", "string"):
            # target-mean encoding strength for categoricals (between-group share of variance), capped to low cardinality
            if s.nunique() > 50:
                continue
            g = y.groupby(s.astype(str))
            r = float(np.sqrt(((g.transform("mean") - y.mean()) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)))
        else:
            x = pd.to_numeric(s, errors="coerce")
            if x.nunique() < 2:
                continue
            r = float(abs(x.corr(y)))
        if np.isfinite(r) and r > thr:
            out[c] = round(r, 3)
    return out


def drift_screen(train_latest: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for c in cols:
        if c not in train_latest or c not in test:
            continue
        a, b = train_latest[c], test[c]
        if a.dtype == object or str(a.dtype) in ("bool", "category", "string"):
            pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
            idx = pa.index.union(pb.index)
            v = float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())
            if v > DRIFT_TVD:
                out[c] = round(v, 3)
        else:
            xa, xb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
            sd = xa.std()
            if sd and np.isfinite(sd) and sd > 0:
                v = float(abs(xa.mean() - xb.mean()) / sd)
                if v > DRIFT_SMD:
                    out[c] = round(v, 3)
    return out


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    errs = []
    if list(sub.columns) != ["row_id", "prediction"]:
        errs.append(f"columns must be ['row_id','prediction'], got {list(sub.columns)}")
        return errs
    if len(sub) != len(sample):
        errs.append(f"row count {len(sub)} != sample {len(sample)}")
    if not sub["row_id"].is_unique:
        errs.append("duplicate row_id")
    if set(sub["row_id"]) != set(sample["row_id"]):
        errs.append("row_id set differs from sample_submission")
    p = pd.to_numeric(sub["prediction"], errors="coerce")
    if p.isna().any() or not np.isfinite(p).all():
        errs.append("non-finite predictions")
    if (p < 0).any():
        errs.append("negative predictions")
    return errs


def prediction_sanity(pred_cost_per_expo: float, hist: dict[int, float], band: float = SANITY_BAND) -> bool:
    lo, hi = min(hist.values()) * (1 - band), max(hist.values()) * (1 + band)
    return lo <= pred_cost_per_expo <= hi
