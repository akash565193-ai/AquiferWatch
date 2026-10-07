"""
app.py - Groundwater Risk Forecast backend

Endpoints:
  GET  /                          -> dashboard (index.html)
  GET  /api/localities            -> latest risk prediction per locality (for map/cards)
  GET  /api/history/<locality>    -> full year-by-year history for one locality (for charts)
  GET  /api/metrics               -> model evaluation metrics (accuracy, report, feature importance)
  POST /api/predict                -> run the trained model on a custom hypothetical input
"""

import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "risk_model.joblib")
METRICS_PATH = os.path.join(PROJECT_DIR, "model", "metrics.json")
IMPORTANCE_PATH = os.path.join(PROJECT_DIR, "model", "feature_importance.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CURRENT_PRED_PATH = os.path.join(STATIC_DIR, "current_predictions.json")
HISTORY_PATH = os.path.join(STATIC_DIR, "history.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---- Load model + data once at startup ----
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
LABEL_ENCODER = bundle["label_encoder"]
FEATURE_COLS = bundle["feature_cols"]

with open(CURRENT_PRED_PATH) as f:
    CURRENT_PREDICTIONS = json.load(f)

with open(HISTORY_PATH) as f:
    HISTORY = json.load(f)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

IMPORTANCE_DF = pd.read_csv(IMPORTANCE_PATH, index_col=0)


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/localities")
def get_localities():
    return jsonify(CURRENT_PREDICTIONS)


@app.route("/api/history/<locality>")
def get_history(locality):
    rows = [r for r in HISTORY if r["locality"] == locality]
    if not rows:
        return jsonify({"error": "locality not found"}), 404
    rows.sort(key=lambda r: r["year"])
    return jsonify(rows)


@app.route("/api/metrics")
def get_metrics():
    return jsonify({
        "test_accuracy": METRICS["test_accuracy"],
        "classification_report": METRICS["classification_report"],
        "confusion_matrix": METRICS["confusion_matrix"],
        "classes": METRICS["classes"],
        "n_train": METRICS["n_train"],
        "n_test": METRICS["n_test"],
        "feature_importance": IMPORTANCE_DF["importance"].to_dict(),
    })


def _build_feature_row(payload):
    """Build a single-row feature dataframe from a simplified user input,
    filling in engineered features with reasonable derived/default values."""
    rainfall = float(payload.get("rainfall_mm", 900))
    rainfall_prev = float(payload.get("rainfall_mm_last_year", rainfall))
    num_borewells = float(payload.get("num_borewells", 400))
    num_borewells_prev = float(payload.get("num_borewells_last_year", num_borewells))
    pump_hours = float(payload.get("avg_pump_hours_per_day", 6))
    impervious_pct = float(payload.get("impervious_pct", 20))
    depth = float(payload.get("groundwater_depth_m", 10))
    depth_prev = float(payload.get("groundwater_depth_m_last_year", depth))
    land_use = payload.get("land_use", "mixed")

    row = {
        "rainfall_mm": rainfall,
        "rainfall_3yr_avg": (rainfall + rainfall_prev) / 2,
        "rainfall_trend_3yr": rainfall - rainfall_prev,
        "num_borewells": num_borewells,
        "borewell_growth_pct": ((num_borewells - num_borewells_prev) / max(num_borewells_prev, 1)) * 100,
        "avg_pump_hours_per_day": pump_hours,
        "impervious_pct": impervious_pct,
        "groundwater_depth_m": depth,
        "depth_change_1yr": depth - depth_prev,
        "land_use_agriculture": 1 if land_use == "agriculture" else 0,
        "land_use_forest": 1 if land_use == "forest" else 0,
        "land_use_mixed": 1 if land_use == "mixed" else 0,
        "land_use_urban": 1 if land_use == "urban" else 0,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        X = _build_feature_row(payload)
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    probs = MODEL.predict_proba(X)[0]
    classes = LABEL_ENCODER.classes_
    pred_idx = probs.argmax()
    result = {
        "predicted_risk": classes[pred_idx],
        "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
