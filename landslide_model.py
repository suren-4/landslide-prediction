"""
Landslide Risk Prediction Model Trainer
Trains a Random Forest Classifier on geotechnical sensor telemetry
(Rainfall, Soil Moisture, Slope Angle, Vibration).
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "synthetic_landslide_data.csv")
    model_path = os.path.join(current_dir, "landslide_prediction_model.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run New.py first.")

    print(f"[1/5] Loading geotechnical dataset from: {data_path}")
    data = pd.read_csv(data_path)

    # Validate missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Features and Target
    feature_names = ["rainfall", "soil_moisture", "slope_angle", "vibration"]
    X = data[feature_names]
    y = data["landslide_occurred"]

    print(f"[2/5] Splitting dataset ({len(data)} records, {len(feature_names)} features)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[3/5] Training Random Forest Classifier (100 estimators)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("[4/5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n================ MODEL PERFORMANCE ================")
    print(f"Accuracy Score : {acc * 100:.2f}%")
    print(f"ROC-AUC Score  : {auc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Landslide"]))

    print("Feature Importances:")
    importances = model.feature_importances_
    for feat, imp in zip(feature_names, importances):
        print(f"  * {feat:15}: {imp * 100:.2f}%")
    print("===================================================\n")

    print(f"[5/5] Saving model weights to: {model_path}")
    joblib.dump(model, model_path)
    print("Model successfully trained and persisted.")
    return model

if __name__ == "__main__":
    train_model()
