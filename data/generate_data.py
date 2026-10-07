"""
generate_data.py
-----------------
Generates a synthetic but realistically-patterned historical dataset
(2010-2025) for groundwater risk forecasting, covering rainfall,
borewell usage and land-use for a set of Tamil Nadu localities.

Real government sources that a production version of this system would
pull from instead of synthetic data:
  - Rainfall:          India Meteorological Department (IMD) gridded rainfall
  - Borewell/GW level: Central Ground Water Board (CGWB) / State Ground
                        Water & Surface Water Resources Data Centre (WRD)
  - Land use:          Bhuvan / NRSC LULC (National Remote Sensing Centre)

The generator builds each locality with a persistent "character"
(baseline rainfall, land-use mix, borewell growth rate) and then rolls a
year-by-year physical-ish simulation forward, so the resulting labels
have a genuine learnable relationship to the features instead of being
random noise. This is what makes the downstream ML model meaningful.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# 25 Tamil Nadu localities with approximate lat/lon and a land-use archetype
LOCALITIES = [
    ("Chennai_Anna_Nagar",      13.0850, 80.2101, "urban"),
    ("Chennai_Tambaram",        12.9249, 80.1000, "urban"),
    ("Chengalpattu",            12.6819, 79.9888, "mixed"),
    ("Tiruvallur",              13.1231, 79.9120, "agriculture"),
    ("Kanchipuram",             12.8342, 79.7036, "agriculture"),
    ("Vellore",                 12.9165, 79.1325, "mixed"),
    ("Krishnagiri",             12.5266, 78.2150, "agriculture"),
    ("Dharmapuri",              12.1357, 78.1580, "agriculture"),
    ("Salem",                   11.6643, 78.1460, "mixed"),
    ("Namakkal",                11.2189, 78.1677, "agriculture"),
    ("Erode",                   11.3410, 77.7172, "agriculture"),
    ("Coimbatore",              11.0168, 76.9558, "urban"),
    ("Tiruppur",                11.1085, 77.3411, "urban"),
    ("Nilgiris",                11.4064, 76.6932, "forest"),
    ("Karur",                   10.9601, 78.0766, "agriculture"),
    ("Tiruchirappalli",         10.7905, 78.7047, "mixed"),
    ("Perambalur",              11.2342, 78.8807, "agriculture"),
    ("Ariyalur",                11.1401, 79.0782, "agriculture"),
    ("Thanjavur",               10.7870, 79.1378, "agriculture"),
    ("Pudukkottai",             10.3833, 78.8200, "agriculture"),
    ("Sivaganga",               9.8437,  78.4809, "agriculture"),
    ("Madurai",                 9.9252,  78.1198, "urban"),
    ("Theni",                   10.0104, 77.4768, "forest"),
    ("Virudhunagar",            9.5851,  77.9581, "agriculture"),
    ("Tirunelveli",             8.7139,  77.7567, "mixed"),
]

YEARS = list(range(2010, 2026))  # 16 years of history (2010-2025)

LAND_USE_PARAMS = {
    # recharge_factor: fraction of rainfall that infiltrates to groundwater
    # demand_factor: relative water demand pressure from this land use
    "urban":       dict(recharge_factor=0.10, demand_factor=1.3, borewell_growth=0.06),
    "agriculture": dict(recharge_factor=0.22, demand_factor=1.8, borewell_growth=0.05),
    "forest":      dict(recharge_factor=0.42, demand_factor=0.4, borewell_growth=0.01),
    "mixed":       dict(recharge_factor=0.28, demand_factor=1.1, borewell_growth=0.035),
}

rows = []

for name, lat, lon, land_use in LOCALITIES:
    params = LAND_USE_PARAMS[land_use]

    # Persistent locality "character"
    baseline_rainfall = np.random.uniform(650, 1100)      # mm/year average
    rainfall_trend = np.random.uniform(-8, 2)              # mm/year drift (many areas trending drier)
    num_borewells = np.random.randint(150, 900)            # starting borewell count
    groundwater_level = np.random.uniform(6, 14)           # meters below ground (shallower=better)
    urban_creep = np.random.uniform(0.0, 0.015) if land_use != "forest" else 0.0

    for i, year in enumerate(YEARS):
        # --- Rainfall: trend + monsoon variability + occasional drought/flood years
        rainfall = baseline_rainfall + rainfall_trend * i
        rainfall += np.random.normal(0, 90)
        drought_shock = np.random.choice([0, 1], p=[0.82, 0.18])
        if drought_shock:
            rainfall *= np.random.uniform(0.55, 0.75)
        rainfall = max(rainfall, 150)

        # --- Land use slowly urbanizes for non-forest zones (reduces recharge over time)
        effective_recharge_factor = max(params["recharge_factor"] - urban_creep * i, 0.05)
        impervious_pct = min(10 + urban_creep * i * 300, 85) if land_use == "urban" else \
                          min(5 + urban_creep * i * 150, 60)

        # --- Borewells grow each year (more extraction pressure)
        num_borewells = int(num_borewells * (1 + params["borewell_growth"] + np.random.normal(0, 0.01)))
        avg_pump_hours = np.clip(np.random.normal(6 + params["demand_factor"], 1.2), 2, 14)

        # --- Water balance: recharge (rain soaking in) vs extraction (pumping)
        recharge_mm = rainfall * effective_recharge_factor
        extraction_index = num_borewells * avg_pump_hours * params["demand_factor"] / 10000.0

        # Groundwater level change: recharge lowers the depth (water rises),
        # extraction increases the depth (water table drops further underground)
        delta = -(recharge_mm / 120.0) + extraction_index
        delta += np.random.normal(0, 0.3)
        groundwater_level = max(groundwater_level + delta, 1.5)
        groundwater_level = min(groundwater_level, 45)

        rows.append(dict(
            locality=name, lat=lat, lon=lon, land_use=land_use, year=year,
            rainfall_mm=round(rainfall, 1),
            num_borewells=num_borewells,
            avg_pump_hours_per_day=round(avg_pump_hours, 2),
            impervious_pct=round(impervious_pct, 1),
            groundwater_depth_m=round(groundwater_level, 2),
        ))

df = pd.DataFrame(rows)

# --- Build the *future* label: how much did depth worsen over the NEXT 2 years?
# This is the ground-truth target a real system would derive from monitoring
# records once they become available. We compute it here as (future - present).
df = df.sort_values(["locality", "year"]).reset_index(drop=True)
df["future_depth_2y"] = df.groupby("locality")["groundwater_depth_m"].shift(-2)
df["depth_increase_2y"] = df["future_depth_2y"] - df["groundwater_depth_m"]

valid = df["depth_increase_2y"].notna()
q_high = df.loc[valid, "depth_increase_2y"].quantile(0.70)
q_med = df.loc[valid, "depth_increase_2y"].quantile(0.35)

def label_risk(x):
    if pd.isna(x):
        return np.nan
    if x >= q_high:
        return "high"
    elif x >= q_med:
        return "medium"
    else:
        return "low"

df["risk_label"] = df["depth_increase_2y"].apply(label_risk)
print(f"Thresholds -> medium >= {q_med:.2f} m, high >= {q_high:.2f} m (worsening in depth over 2 yrs)")

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groundwater_history.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows for {df['locality'].nunique()} localities -> {out_path}")
print(df["risk_label"].value_counts(dropna=False))
