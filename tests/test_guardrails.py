"""Automated guardrail tests. Run:  C:\\workspace\\python.exe -m pytest -q

Tests that need the cached feature frames (data/cache) or a built submission are skipped if those are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev import MAX_AGE, reported_fraction_curves, row_reported_fraction  # noqa: E402
from features import ID_COLS, LEAK_COLS, NOISE_COLS, PRICING_COLS, feature_columns  # noqa: E402
from guardrails import drift_screen, exposure_check, leak_screen, prediction_sanity, validate_submission  # noqa: E402
from prep import LINES  # noqa: E402

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs"
HAVE_CACHE = all((CACHE / f"{ln}_{k}.parquet").exists() for ln in LINES for k in ("train", "test"))
needs_cache = pytest.mark.skipif(not HAVE_CACHE, reason="feature cache not built (run exp.py)")


@pytest.fixture(scope="module")
def frames():
    return {ln: (pd.read_parquet(CACHE / f"{ln}_train.parquet"), pd.read_parquet(CACHE / f"{ln}_test.parquet")) for ln in LINES}


# ---------- feature governance -------------------------------------------------------------------------------
@needs_cache
@pytest.mark.parametrize("ln", LINES)
def test_forbidden_columns_never_used(frames, ln):
    train, _ = frames[ln]
    train = train.assign(dev_f=1.0)
    cols = set(feature_columns(train))
    forbidden = set(LEAK_COLS) | set(ID_COLS) | set(NOISE_COLS) | set(PRICING_COLS) | {"dev_f", "claim_count", "net_incurred", "net_capped", "paid", "open_count", "year", "exposure"}
    assert not cols & forbidden, f"forbidden columns in feature set: {cols & forbidden}"


@needs_cache
@pytest.mark.parametrize("ln", LINES)
def test_no_feature_leaks_same_row_outcome(frames, ln):
    train, _ = frames[ln]
    cols = feature_columns(train.assign(dev_f=1.0))
    assert leak_screen(train, cols) == {}, "a feature correlates too strongly with the same-row claim count"


@needs_cache
def test_leak_screen_catches_known_leak(frames):
    """Self-test: the screen must flag customer_service_contacts (the leak found in EDA) when it IS included."""
    train, _ = frames["motor"]
    assert "customer_service_contacts" in leak_screen(train, ["customer_service_contacts"])


@needs_cache
@pytest.mark.parametrize("ln", LINES)
def test_no_severe_train_test_drift_in_features(frames, ln):
    train, test = frames[ln]
    cols = feature_columns(train.assign(dev_f=1.0))
    flagged = drift_screen(train[train["year"] == train["year"].max()], test, cols)
    assert len(flagged) <= 3, f"features with large 2024->2025 drift: {flagged}"


@needs_cache
@pytest.mark.parametrize("ln", LINES)
def test_time_split_no_future_info(frames, ln):
    train, test = frames[ln]
    assert train["year"].max() == 2024 and (pd.to_datetime(test["period_start"]).dt.year == 2025).all()
    assert (train["exposure"] > 0).all() and (test["exposure"] > 0).all()
    assert test["row_id"].is_unique and train["row_id"].is_unique
    assert not set(train["row_id"]) & set(test["row_id"])


# ---------- claim-development curve -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def curves():
    return reported_fraction_curves()


@pytest.mark.parametrize("ln", LINES)
def test_reporting_curve_is_a_valid_cdf(curves, ln):
    c = curves[ln]
    assert len(c) == MAX_AGE + 1
    assert c[0] >= 0 and np.all(np.diff(c) >= -1e-9), "curve must be non-decreasing"
    assert 0.99 <= c[-1] <= 1.0 + 1e-9, "fully developed at the tail"
    assert c[365] > 0.9


def test_row_reported_fraction_bounds_and_order(curves):
    pol = pd.DataFrame({"period_start": pd.to_datetime(["2020-01-01", "2024-01-01", "2024-07-01"]),
                        "period_end": pd.to_datetime(["2020-12-31", "2024-12-30", "2024-12-30"])})
    f = row_reported_fraction(pol, curves["motor"])
    assert np.all((f > 0) & (f <= 1.0 + 1e-9))
    assert f[0] > f[1] > f[2], "more recent exposure must be less developed"


# ---------- submission --------------------------------------------------------------------------------------
def test_validate_submission_detects_problems():
    sample = pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": 0.0})
    good = pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": [1.0, 2.0, 0.0]})
    assert validate_submission(good, sample) == []
    assert validate_submission(good.assign(prediction=[1.0, -1.0, 0.0]), sample)
    assert validate_submission(good.assign(prediction=[1.0, np.nan, 0.0]), sample)
    assert validate_submission(good.iloc[:2], sample)
    assert validate_submission(pd.DataFrame({"row_id": ["a", "a", "c"], "prediction": [1, 1, 1.0]}), sample)


def test_prediction_sanity_band():
    hist = {2021: 400.0, 2022: 440.0, 2023: 430.0}
    assert prediction_sanity(420.0, hist) and not prediction_sanity(250.0, hist) and not prediction_sanity(700.0, hist)


# ---------- exposure ----------------------------------------------------------------------------------------
def test_exposure_check_flags_day_units_and_360_basis():
    d = pd.DataFrame({"period_start": pd.to_datetime(["2024-01-01"] * 3), "period_end": pd.to_datetime(["2024-12-31"] * 3),
                      "exposure": [1.0, 365.0, 365 / 360]})
    # 365/360 is within EXPO_TOL of the dates (normal rows differ by up to 0.0137), so only above_one catches it
    assert exposure_check(d) == {"above_one": 2, "date_mismatch": 1}
    assert exposure_check(d.iloc[:1]) == {}


@needs_cache
def test_cached_exposure_is_clean(frames):
    for ln, (tr, te) in frames.items():
        assert "above_one" not in exposure_check(tr), ln
        assert "above_one" not in exposure_check(te), ln


@pytest.mark.skipif(not (OUT / "submission_final.csv").exists(), reason="final submission not built")
def test_final_submission_is_valid_and_sane():
    sub = pd.read_csv(OUT / "submission_final.csv")
    sample = pd.read_parquet(ROOT / "data" / "public" / "sample_submission.parquet")
    assert validate_submission(sub, sample) == []
    import json
    rep = json.loads((OUT / "report_final.json").read_text())
    for ln in LINES:
        assert prediction_sanity(rep[ln]["pred_cost_per_expo_2025"], {int(k): v for k, v in rep[ln]["hist_developed_cost_per_expo"].items()}), ln
