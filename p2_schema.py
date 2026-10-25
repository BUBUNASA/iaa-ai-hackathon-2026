"""Part 2: compare emerging vs developed schemas and show the expenses / reference-policy tables."""
from pathlib import Path

import pandas as pd

D = Path(__file__).parent / "data" / "part2"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 60)


def rd(market: str, t: str) -> pd.DataFrame:
    return pd.read_parquet(D / market / f"{t}.parquet")


for t in ["customers", "addresses", "weather", "claims", "policies_motor", "policies_buildings", "policies_contents"]:
    e, v = rd("emerging", t), rd("developed", t)
    print(f"== {t} | emerging only: {sorted(set(e.columns) - set(v.columns))} | developed only: {sorted(set(v.columns) - set(e.columns))}")
    print("   cols:", list(e.columns))
for ln in ("motor", "buildings", "contents"):
    te, tr = rd("emerging", f"test_{ln}"), rd("emerging", f"policies_{ln}")
    print(f"== test_{ln} | test only: {sorted(set(te.columns) - set(tr.columns))} | train only: {sorted(set(tr.columns) - set(te.columns))}")
print("\n== expenses\n", rd("emerging", "expenses").to_string())
r = rd("emerging", "reference_policies")
print("\n== reference_policies cols:", r.columns.tolist())
print(r.head(3).T.to_string())
print("\n== sample_submission\n", rd("emerging", "sample_submission").head(3))
