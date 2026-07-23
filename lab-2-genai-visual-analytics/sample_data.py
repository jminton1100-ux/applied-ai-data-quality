"""
Synthetic Washington State water-monitoring sample dataset.

Generates a small, realistic-looking environmental monitoring dataset with
deliberate data quality issues so the data-quality flagging step of the lab
has something concrete to surface.

This file exists so the lab is self-contained: a user can run the whole
notebook without uploading their own CSV.

Data-quality issues seeded intentionally:
- Missing values (NaN) in temperature and dissolved oxygen columns
- Outliers: a pH reading of 25.4 (impossible — pH scale is 0-14)
- Outliers: a temperature reading of 142.0 (impossible for natural water in F)
- Inconsistent unit: one row records turbidity in different scale
- Duplicate rows: same site, same date, same readings (data-entry error)
- Type inconsistency: most pH values are floats, a few are strings ("7.2")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Return a realistic synthetic WA water-monitoring dataset with seeded DQ issues."""
    rng = np.random.default_rng(seed)

    # Five monitoring sites across Washington State, loosely modeled on real basins.
    sites = [
        {"site_id": "WAEC-001", "site_name": "Spokane River at Riverside",      "basin": "Spokane"},
        {"site_id": "WAEC-002", "site_name": "Lake Washington at Madison Park", "basin": "Puget Sound"},
        {"site_id": "WAEC-003", "site_name": "Yakima River at Cle Elum",        "basin": "Yakima"},
        {"site_id": "WAEC-004", "site_name": "Columbia River at Wenatchee",     "basin": "Columbia"},
        {"site_id": "WAEC-005", "site_name": "Chehalis River at Centralia",     "basin": "Chehalis"},
    ]

    # Generate 120 days of readings, roughly weekly per site.
    start = datetime(2025, 1, 1)
    records = []
    for site in sites:
        for week in range(24):
            date = start + timedelta(weeks=week, days=int(rng.integers(0, 3)))
            # Seasonal-ish temperature pattern in degrees Fahrenheit.
            base_temp = 45 + 20 * np.sin(week / 24 * np.pi)
            temperature_f = base_temp + rng.normal(0, 3)

            records.append({
                "site_id":           site["site_id"],
                "site_name":         site["site_name"],
                "basin":             site["basin"],
                "sample_date":       date.strftime("%Y-%m-%d"),
                "temperature_f":     round(float(temperature_f), 1),
                "ph":                round(float(7.2 + rng.normal(0, 0.3)), 2),
                "dissolved_oxygen_mgL": round(float(9.5 + rng.normal(0, 1.2)), 2),
                "turbidity_ntu":     round(float(abs(rng.normal(3.0, 1.5))), 2),
                "conductivity_uScm": round(float(180 + rng.normal(0, 40)), 1),
            })

    df = pd.DataFrame(records)

    # ---- Seed deliberate data-quality issues ----

    # Missing values (completeness issues)
    df.loc[7,  "temperature_f"]        = np.nan
    df.loc[23, "temperature_f"]        = np.nan
    df.loc[44, "dissolved_oxygen_mgL"] = np.nan
    df.loc[88, "dissolved_oxygen_mgL"] = np.nan
    df.loc[101, "ph"]                  = np.nan

    # Impossible values (validity/accuracy issues)
    df.loc[15, "ph"]            = 25.4   # pH scale max is 14 — clearly wrong
    df.loc[62, "temperature_f"] = 142.0  # impossible for surface water in WA

    # Inconsistent scale (one row appears to be in different units)
    df.loc[33, "turbidity_ntu"] = 4500.0   # likely entered in wrong scale

    # Type inconsistency: most pH are floats, but a few got entered as strings.
    # We do this by converting the column to object dtype and replacing some cells.
    df["ph"] = df["ph"].astype(object)
    df.loc[50, "ph"] = "7.2"
    df.loc[78, "ph"] = "7.5"

    # Duplicate row (same site, date, all readings the same — a re-entry error)
    duplicate = df.iloc[10].copy()
    df = pd.concat([df, duplicate.to_frame().T], ignore_index=True)

    return df


if __name__ == "__main__":
    # Quick smoke test when this file is run directly.
    df = generate_sample_dataset()
    print(f"Generated dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing values per column:")
    print(df.isna().sum())
