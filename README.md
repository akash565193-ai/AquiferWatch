# AquiferWatch — Groundwater Scarcity Early-Warning System

A complete, runnable prototype for the project brief: forecast groundwater
scarcity risk 1–2 years in advance from rainfall, borewell usage, and
land-use history, and present it as a color-coded dashboard.

This is an end-to-end pipeline — data → analysis → ML model → API → dashboard
— built so you can run it locally in a few minutes and use it as the basis
for a report, demo, or further development with real data.

## What's inside

```
groundwater-project/
├── data/
│   ├── generate_data.py        # builds the historical dataset (see "About the data" below)
│   └── groundwater_history.csv # 25 localities x 16 years = 400 rows
├── model/
│   ├── analyze_data.py         # exploratory data analysis -> model/eda_charts/*.png
│   ├── train_model.py          # feature engineering + RandomForest training + export
│   ├── risk_model.joblib       # trained model (after running train_model.py)
│   ├── metrics.json            # accuracy / precision / recall / confusion matrix
│   ├── feature_importance.csv
│   └── eda_charts/             # 4 exploratory charts
├── backend/
│   ├── app.py                  # Flask API + serves the dashboard
│   ├── static/
│   │   ├── style.css
│   │   ├── app.js
│   │   ├── current_predictions.json   # latest-year risk prediction per locality
│   │   └── history.json               # full 16-year history for charts
│   └── templates/
│       └── index.html          # the dashboard UI
├── requirements.txt
└── README.md
```

## How to run it

```bash
cd groundwater-project
pip install -r requirements.txt

# 1. Generate the historical dataset
python3 data/generate_data.py

# 2. Explore it (optional, produces charts in model/eda_charts/)
python3 model/analyze_data.py

# 3. Engineer features and train the risk model
python3 model/train_model.py

# 4. Launch the API + dashboard
cd backend
python3 app.py
```

Then open **http://localhost:5050** in your browser. The dashboard needs
internet access in the browser for its map tiles and fonts (Leaflet,
CartoDB, Chart.js, Google Fonts CDNs) — the API, data and model all run
100% locally.

## About the data

Real groundwater/rainfall/land-use records for Indian districts exist but
sit across several disconnected government portals, are inconsistently
digitized, and typically require registration to bulk-download:

| Ingredient | Real source you'd wire this up to |
|---|---|
| Rainfall | India Meteorological Department (IMD) gridded rainfall data |
| Borewell count / groundwater level | Central Ground Water Board (CGWB) & state Water Resources Departments (WRD) monitoring wells |
| Land use | Bhuvan / NRSC LULC (National Remote Sensing Centre) |

To make this project runnable end-to-end without needing those data-access
approvals, `generate_data.py` **simulates** 16 years (2010–2025) of
plausible data for 25 real Tamil Nadu localities. It isn't random noise —
each locality gets a persistent "character" (baseline rainfall, land-use
mix, borewell growth rate), and a simple water-balance rule
(`depth change = extraction pressure − recharge from rain`) is rolled
forward year by year. This means the resulting patterns are genuinely
learnable, the same way real hydrogeological patterns are, so the trained
model is meaningfully validating a real methodology rather than
memorizing randomness.

**To use real data**, swap `groundwater_history.csv` for actual
CGWB/IMD/Bhuvan records with the same columns (`locality, lat, lon,
land_use, year, rainfall_mm, num_borewells, avg_pump_hours_per_day,
impervious_pct, groundwater_depth_m`) and re-run `train_model.py` — no
other code needs to change.

## How the model works

1. **Label construction**: for every locality-year, we look 2 years ahead
   and compute how much the groundwater depth worsened. The worst 30% of
   locality-years become `high` risk, the next 35% `medium`, the rest `low`
   (quantile-based, so classes are balanced for training).
2. **Features**: current rainfall, 3-year rainfall average and trend,
   borewell count and growth rate, pumping hours, impervious (concrete)
   cover, current groundwater depth, and 1-year depth change, plus a
   one-hot encoding of land use.
3. **Model**: `RandomForestClassifier` (300 trees, balanced class weights).
   On held-out data it reaches ~80% accuracy with sensible feature
   importances — current depth, borewell count, and recent depth change
   dominate, rainfall trend and land use contribute secondarily, which
   matches real hydrogeological intuition.
4. **Serving**: the trained model runs the same feature pipeline on each
   locality's most recent year to produce today's live risk map, and is
   also exposed via `/api/predict` for the "what-if" simulator in the
   dashboard.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/localities` | GET | Latest risk prediction + stats for every locality |
| `/api/history/<locality>` | GET | Full 16-year history for one locality (for charts) |
| `/api/metrics` | GET | Model accuracy, classification report, feature importance |
| `/api/predict` | POST | Run the model on custom rainfall/borewell/land-use numbers |

Example:
```bash
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall_mm": 400, "num_borewells": 900, "avg_pump_hours_per_day": 12,
       "impervious_pct": 60, "groundwater_depth_m": 20, "land_use": "urban"}'
```

## Extending toward production

- Replace synthetic data with real CGWB/IMD/Bhuvan feeds (see table above);
  the pipeline's column schema is designed to make this a drop-in swap.
- Add a scheduled retraining job (e.g. yearly, once a new monsoon's data
  lands) rather than a one-off `train_model.py` run.
- Swap `RandomForestClassifier` for a model that natively handles
  irregular time series (e.g. gradient-boosted trees on rolling windows,
  or an LSTM) once more years of real history are available.
- Add authentication and role-based views (farmer vs. district officer)
  if this becomes a public-facing government tool.
- Move from SQLite/CSV to a proper time-series database if scaling past a
  few hundred localities.
