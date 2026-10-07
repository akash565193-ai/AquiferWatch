"""
analyze_data.py
----------------
Exploratory data analysis on the historical dataset: summary statistics
and a handful of charts showing the relationships the model later learns
from (rainfall vs. groundwater depth, borewell growth vs. depletion,
land-use vs. risk). Saves PNGs into model/eda_charts/.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "groundwater_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "eda_charts")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=== Dataset shape ===")
print(df.shape)
print("\n=== Years covered ===", df.year.min(), "-", df.year.max())
print("\n=== Localities ===", df.locality.nunique())
print("\n=== Land-use breakdown (locality-years) ===")
print(df.land_use.value_counts())
print("\n=== Risk label breakdown ===")
print(df.risk_label.value_counts(dropna=False))
print("\n=== Summary statistics ===")
print(df[["rainfall_mm", "num_borewells", "avg_pump_hours_per_day",
          "impervious_pct", "groundwater_depth_m"]].describe().round(1))

plt.rcParams.update({"figure.facecolor": "#111", "axes.facecolor": "#111",
                      "savefig.facecolor": "#111", "text.color": "#eee",
                      "axes.labelcolor": "#eee", "xtick.color": "#aaa",
                      "ytick.color": "#aaa", "axes.edgecolor": "#555"})

# 1. Rainfall vs groundwater depth trend for a few localities
fig, ax1 = plt.subplots(figsize=(9, 4.5))
sample_localities = df.locality.unique()[:3]
for loc in sample_localities:
    sub = df[df.locality == loc].sort_values("year")
    ax1.plot(sub.year, sub.groundwater_depth_m, marker="o", label=loc)
ax1.set_ylabel("Groundwater depth (m, deeper = worse)")
ax1.set_xlabel("Year")
ax1.set_title("Groundwater depth trend — sample localities")
ax1.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_depth_trend_sample.png", dpi=130)
plt.close(fig)

# 2. Land use vs average groundwater depth
fig, ax = plt.subplots(figsize=(6, 4.5))
means = df.groupby("land_use")["groundwater_depth_m"].mean().sort_values()
ax.barh(means.index, means.values, color="#2E86AB")
ax.set_xlabel("Avg groundwater depth (m)")
ax.set_title("Land use vs. average groundwater depth")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_landuse_vs_depth.png", dpi=130)
plt.close(fig)

# 3. Borewell count vs depth-worsening (scatter, colored by risk)
fig, ax = plt.subplots(figsize=(6.5, 5))
colors = {"low": "#3FA34D", "medium": "#F2B705", "high": "#E4572E"}
plot_df = df.dropna(subset=["risk_label"])
for label, c in colors.items():
    sub = plot_df[plot_df.risk_label == label]
    ax.scatter(sub.num_borewells, sub.depth_increase_2y, s=18, color=c, label=label, alpha=0.7)
ax.set_xlabel("Number of borewells")
ax.set_ylabel("Depth increase over next 2 yrs (m)")
ax.set_title("Borewell count vs. future depletion, colored by risk label")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_borewells_vs_future_depletion.png", dpi=130)
plt.close(fig)

# 4. Rainfall distribution by risk label
fig, ax = plt.subplots(figsize=(6.5, 4.5))
plot_df.boxplot(column="rainfall_mm", by="risk_label", ax=ax,
                 patch_artist=True,
                 boxprops=dict(facecolor="#1C282B", color="#5FB4D6"),
                 medianprops=dict(color="#E4572E"))
ax.set_title("Rainfall distribution by future risk label")
ax.set_xlabel("Risk label (next 1-2 yrs)")
ax.set_ylabel("Rainfall (mm)")
plt.suptitle("")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_rainfall_by_risk.png", dpi=130)
plt.close(fig)

print(f"\nSaved 4 charts to {OUT_DIR}/")
