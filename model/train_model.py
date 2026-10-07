"""
train_model.py
---------------
Loads the historical dataset, engineers features (trailing rainfall
trends, borewell density, pumping pressure, land-use encoding),
trains a RandomForestClassifier to predict 1-2 year-ahead groundwater
risk (low / medium / high), evaluates it, and exports:

  - model/risk_model.joblib        (trained classifier + label encoder)
  - model/feature_importance.csv
  - model/metrics.json
  - backend/static/current_predictions.json  (latest-year prediction per
    locality, consumed directly by the frontend dashboard)
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "groundwater_history.csv")
MODEL_DIR = SCRIPT_DIR
STATIC_DIR = os.path.join(PROJECT_DIR, "backend", "static")

FEATURE_COLS = [
    "rainfall_mm", "rainfall_3yr_avg", "rainfall_trend_3yr",
    "num_borewells", "borewell_growth_pct",
    "avg_pump_hours_per_day", "impervious_pct",
    "groundwater_depth_m", "depth_change_1yr",
    "land_use_agriculture", "land_use_forest", "land_use_mixed", "land_use_urban",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["locality", "year"]).copy()
    g = df.groupby("locality")

    df["rainfall_3yr_avg"] = g["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["rainfall_trend_3yr"] = g["rainfall_mm"].transform(lambda s: s.diff(3))
    df["borewell_growth_pct"] = g["num_borewells"].transform(lambda s: s.pct_change()) * 100
    df["depth_change_1yr"] = g["groundwater_depth_m"].transform(lambda s: s.diff(1))

    df[["rainfall_trend_3yr", "borewell_growth_pct", "depth_change_1yr"]] = \
        df[["rainfall_trend_3yr", "borewell_growth_pct", "depth_change_1yr"]].fillna(0)

    land_dummies = pd.get_dummies(df["land_use"], prefix="land_use")
    for col in ["land_use_agriculture", "land_use_forest", "land_use_mixed", "land_use_urban"]:
        if col not in land_dummies.columns:
            land_dummies[col] = 0
    df = pd.concat([df, land_dummies], axis=1)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    train_df = df.dropna(subset=["risk_label"]).copy()

    X = train_df[FEATURE_COLS]
    y = train_df["risk_label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # alphabetical: high=0, low=1, medium=2 (order captured by le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    importances.to_csv(f"{MODEL_DIR}/feature_importance.csv", header=["importance"])
    print("\nTop features:\n", importances.head(6))

    metrics = {
        "test_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": le.classes_.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"model": clf, "label_encoder": le, "feature_cols": FEATURE_COLS}, f"{MODEL_DIR}/risk_model.joblib")
    print(f"\nSaved model -> {MODEL_DIR}/risk_model.joblib")

    # ---- Build "current" predictions: predict the NEXT 1-2yr risk for every
    # locality using its most recent year of data (this is what the live
    # dashboard shows, since real future outcomes aren't known yet for it).
    latest = df.sort_values("year").groupby("locality").tail(1).copy()
    X_latest = latest[FEATURE_COLS]
    probs = clf.predict_proba(X_latest)
    pred_idx = probs.argmax(axis=1)
    latest["predicted_risk"] = le.inverse_transform(pred_idx)
    for i, cls in enumerate(le.classes_):
        latest[f"prob_{cls}"] = probs[:, i]

    out_cols = [
        "locality", "lat", "lon", "land_use", "year",
        "rainfall_mm", "num_borewells", "avg_pump_hours_per_day",
        "impervious_pct", "groundwater_depth_m",
        "predicted_risk", "prob_low", "prob_medium", "prob_high",
    ]
    latest_out = latest[out_cols].sort_values("predicted_risk")
    latest_out.to_json(f"{STATIC_DIR}/current_predictions.json", orient="records", indent=2)
    print(f"Saved current predictions -> {STATIC_DIR}/current_predictions.json")

    # Also dump full history (with engineered cols trimmed) for frontend charts
    hist_cols = ["locality", "lat", "lon", "land_use", "year", "rainfall_mm",
                 "num_borewells", "avg_pump_hours_per_day", "impervious_pct",
                 "groundwater_depth_m"]
    df[hist_cols].to_json(f"{STATIC_DIR}/history.json", orient="records", indent=2)
    print(f"Saved history -> {STATIC_DIR}/history.json")


if __name__ == "__main__":
    main()
