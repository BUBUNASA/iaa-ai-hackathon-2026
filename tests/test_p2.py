"""Part 2 guardrail tests: market harmonisation, feature governance on the emerging frames, office-premium arithmetic.

Skipped when the Part 2 caches / submission are missing (run p2_features.py / p2_final.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import ID_COLS, LEAK_COLS  # noqa: E402
from guardrails import exposure_check, validate_submission  # noqa: E402
from p2_features import CACHE, P2, _harmonise  # noqa: E402
from p2_models import DEV_EXCLUDE, _cols  # noqa: E402
from prep import LINES  # noqa: E402

TARGETS = ["claim_count", "net_incurred", "net_capped", "paid", "open_count", "renewal_decision"]
HAVE = all((CACHE / f"{m}_{ln}_{k}.parquet").exists() for m in ("developed", "emerging") for ln in LINES for k in ("train", "test"))
needs_p2 = pytest.mark.skipif(not HAVE, reason="Part 2 cache not built (run p2_features.py)")


@pytest.fixture(scope="module")
def p2():
    return {(m, ln, k): pd.read_parquet(CACHE / f"{m}_{ln}_{k}.parquet")
            for m in ("developed", "emerging") for ln in LINES for k in ("train", "test")}


def test_harmonise_cleans_occupation_age_and_construction():
    raw = pd.DataFrame({"occupation": ["Retired ", "self-employed Nurse", None], "policyholder_age_band": ["17-24", "65+", "35-49"],
                        "construction_type": ["BRK", "WOD", "MIX"]})
    out = _harmonise(raw.copy(), "emerging", "buildings")
    assert out["occupation"].tolist()[:2] == ["retired", "nurse"]
    assert out["self_employed"].tolist()[:2] == [False, True]
    assert out["construction_type"].tolist() == ["brick", "timber", "non_standard"]
    assert out["age_band"].tolist() == ["17-24", "65+", "35-49"] and (out["market"] == "emerging").all()
    dev = _harmonise(pd.DataFrame({"occupation": ["Teacher"], "policyholder_age": [24], "construction_type": ["T"]}), "developed", "buildings")
    assert dev["age_band"].iloc[0] == "17-24" and dev["construction_type"].iloc[0] == "timber" and "policyholder_age" not in dev


@needs_p2
@pytest.mark.parametrize("ln", LINES)
def test_markets_share_one_schema_and_vocabulary(p2, ln):
    d, e = p2[("developed", ln, "train")], p2[("emerging", ln, "train")]
    # column ORDER differs (gender is appended for developed); all model code selects columns by name, so compare sets
    assert set(d.columns) == set(e.columns)
    assert set(p2[("emerging", ln, "test")].columns) == set(e.columns) - set(TARGETS)
    for col in ("occupation", "age_band") + (("construction_type",) if ln == "buildings" else ()):
        assert set(e[col].dropna()) <= set(d[col].dropna()), f"{col}: emerging levels unseen in developed"


@needs_p2
@pytest.mark.parametrize("ln", LINES)
def test_emerging_features_governed_and_exposure_clean(p2, ln):
    e, t = p2[("emerging", ln, "train")], p2[("emerging", ln, "test")]
    cols = set(_cols(e.assign(dev_f=1.0)))
    assert not cols & (set(LEAK_COLS) | set(ID_COLS) | set(TARGETS) | {"dev_f", "year", "exposure"})
    assert "gender" in cols and "market" in cols and "age_band" in cols
    assert not set(DEV_EXCLUDE) & set(_cols(p2[("developed", ln, "train")].assign(dev_f=1.0), DEV_EXCLUDE))
    assert "above_one" not in exposure_check(e) and "above_one" not in exposure_check(t)


def test_office_premium_build_up_arithmetic(tmp_path, monkeypatch):
    import p2_premium
    em = tmp_path / "emerging"
    em.mkdir()
    pd.DataFrame({"item": ["fixed_expense_per_policy", "variable_expense", "commission_direct", "commission_comparison_site",
                           "commission_broker", "premium_tax", "cat_reinsurance", "target_profit_margin"],
                  "developed": [45, .12, 0, .10, .15, .12, .06, .05], "emerging": [85, .15, 0, np.nan, .20, .08, .12, .08]}
                 ).to_parquet(em / "expenses.parquet")
    pd.DataFrame({"row_id": ["B1", "M1"], "line": ["buildings", "motor"], "channel": ["broker", "comparison_site"],
                  "exposure": [1.0, 1.0], "written_premium": [500.0, 500.0]}).to_parquet(em / "reference_policies.parquet")
    monkeypatch.setattr(p2_premium, "EM", em)
    out, notes = p2_premium.build_up(pd.DataFrame({"row_id": ["B1", "M1"], "prediction": [100.0, 100.0]}))
    b, m = out.set_index("row_id").loc["B1"], out.set_index("row_id").loc["M1"]
    assert b["office_premium"] == pytest.approx((100 + 12 + 85) / (1 - .15 - .20 - .08 - .08), abs=0.01)
    assert m["cat_reinsurance"] == 0 and m["commission_rate"] == pytest.approx(0.10) and notes  # fallback flagged
    parts = b[["variable_expense", "commission", "premium_tax", "target_profit_margin"]].sum() + 100 + 12 + 85
    assert parts == pytest.approx(b["office_premium"], abs=0.05)  # components add back to the premium


@pytest.mark.skipif(not (ROOT / "outputs" / "submission_p2.csv").exists(), reason="Part 2 submission not built")
def test_part2_submission_valid():
    sub = pd.read_csv(ROOT / "outputs" / "submission_p2.csv", dtype={"row_id": str})
    assert validate_submission(sub, pd.read_parquet(P2 / "emerging" / "sample_submission.parquet")) == []
