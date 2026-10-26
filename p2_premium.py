"""Part 2 report item: office-premium build-up for the 20 reference policies (emerging market).

    python p2_premium.py outputs/submission_p2.csv

Risk premium R = our predicted expected claim cost for the row (from the submission). Then, from the `expenses` table:
  cat reinsurance   = cat_reinsurance x R               (property lines only: buildings, contents)
  fixed expense     = fixed_expense_per_policy           (per policy row, not pro-rated - stated assumption)
  office premium P  = (R + cat + fixed) / (1 - variable_expense - commission[channel] - premium_tax - profit_margin)
Commission for the comparison-site channel is not given for the emerging market (NaN): flagged, and the developed
rate is used as a stated assumption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
EM = ROOT / "data" / "part2" / "emerging"
PROPERTY = ("buildings", "contents")


def build_up(sub: pd.DataFrame, market: str = "emerging") -> tuple[pd.DataFrame, list[str]]:
    ex = pd.read_parquet(EM / "expenses.parquet").set_index("item")
    notes = []
    rate = ex[market].copy()
    if pd.isna(rate["commission_comparison_site"]):
        rate["commission_comparison_site"] = ex.loc["commission_comparison_site", "developed"]
        notes.append(f"commission_comparison_site missing for {market}: used developed rate {rate['commission_comparison_site']:.2f}")
    ref = pd.read_parquet(EM / "reference_policies.parquet").merge(sub, on="row_id", how="left", validate="1:1")
    missing = int(ref["prediction"].isna().sum())
    assert missing == 0, f"{missing} reference policies have no prediction"
    out = ref[["row_id", "line", "channel", "exposure", "written_premium"]].copy()
    out["risk_premium"] = ref["prediction"]
    out["cat_reinsurance"] = ref["line"].isin(PROPERTY) * rate["cat_reinsurance"] * out["risk_premium"]
    out["fixed_expense"] = rate["fixed_expense_per_policy"]
    out["commission_rate"] = ref["channel"].map(lambda c: rate[f"commission_{c}"])
    loading = rate["variable_expense"] + out["commission_rate"] + rate["premium_tax"] + rate["target_profit_margin"]
    out["office_premium"] = (out["risk_premium"] + out["cat_reinsurance"] + out["fixed_expense"]) / (1 - loading)
    for k in ("variable_expense", "premium_tax", "target_profit_margin"):
        out[k] = rate[k] * out["office_premium"]
    out["commission"] = out["commission_rate"] * out["office_premium"]
    out["office/charged"] = out["office_premium"] / out["written_premium"]
    return out.round(2), notes


if __name__ == "__main__":
    sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "submission_p2.csv", dtype={"row_id": str})
    table, notes = build_up(sub)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print("\nnotes:", notes)
    print("mean office/charged by line:", table.groupby("line")["office/charged"].mean().round(3).to_dict())
    table.to_csv(ROOT / "outputs" / "p2_reference_premiums.csv", index=False)
