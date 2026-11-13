"""
GEO-SENTINEL AI: Demo Dataset & Test Vector Generator
Generates curated and randomized geotechnical sensor inputs for showcasing the model,
saving them to structured JSON and CSV files ready for batch testing or frontend demos.
"""
import os
import json
import random
import datetime
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landslide_prediction_model.pkl")

# Load trained model to pre-compute predictions
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

def generate_curated_showcase():
    """Generates a structured list of diverse showcase scenarios."""
    scenarios = [
        {
            "scenario_id": "SCEN-01",
            "name": "Arid Granite Plateau (Safe Baseline)",
            "description": "Clear weather conditions, hard bedrock, minimal seismic activity.",
            "rainfall": 6.5,
            "soil_moisture": 0.14,
            "slope_angle": 10.5,
            "vibration": 0.015,
            "expected_hazard": "LOW"
        },
        {
            "scenario_id": "SCEN-02",
            "name": "Gentle Valley Rain Shower",
            "description": "Persistent rain on low-gradient slope with moderate drainage.",
            "rainfall": 62.0,
            "soil_moisture": 0.42,
            "slope_angle": 14.0,
            "vibration": 0.03,
            "expected_hazard": "LOW"
        },
        {
            "scenario_id": "SCEN-03",
            "name": "Pre-Monsoon Soil Saturation",
            "description": "Continuous soaking rains raising water pore pressure on agricultural terraced hills.",
            "rainfall": 88.0,
            "soil_moisture": 0.65,
            "slope_angle": 26.0,
            "vibration": 0.06,
            "expected_hazard": "GUARD"
        },
        {
            "scenario_id": "SCEN-04",
            "name": "Torrential Mountain Cloudburst",
            "description": "Extreme precipitation rate exceeding 150 mm on a steep 36-degree mountain face.",
            "rainfall": 155.0,
            "soil_moisture": 0.81,
            "slope_angle": 36.0,
            "vibration": 0.11,
            "expected_hazard": "WARNING"
        },
        {
            "scenario_id": "SCEN-05",
            "name": "Tectonic Tremor on Waterlogged Incline",
            "description": "Seismic shockwave shaking fully saturated 42-degree slope. Critical slope failure imminent.",
            "rainfall": 182.0,
            "soil_moisture": 0.93,
            "slope_angle": 43.5,
            "vibration": 0.44,
            "expected_hazard": "CRITICAL"
        },
        {
            "scenario_id": "SCEN-06",
            "name": "Post-Wildfire Flash Mudflow",
            "description": "Vegetation-depleted slope undergoing sudden convective storm.",
            "rainfall": 135.0,
            "soil_moisture": 0.76,
            "slope_angle": 31.5,
            "vibration": 0.08,
            "expected_hazard": "WARNING"
        },
        {
            "scenario_id": "SCEN-07",
            "name": "Road Cutting Blasting Vibration",
            "description": "Construction blasting vibrations triggering shear stress on wet hillside.",
            "rainfall": 45.0,
            "soil_moisture": 0.55,
            "slope_angle": 38.0,
            "vibration": 0.35,
            "expected_hazard": "GUARD"
        },
        {
            "scenario_id": "SCEN-08",
            "name": "Steep Dry Escarpment (Slope stress only)",
            "description": "Steep 48-degree cliff face but completely dry with zero rain.",
            "rainfall": 0.0,
            "soil_moisture": 0.08,
            "slope_angle": 48.0,
            "vibration": 0.02,
            "expected_hazard": "LOW"
        }
    ]

    for item in scenarios:
        if model is not None:
            df = pd.DataFrame([[
                item["rainfall"],
                item["soil_moisture"],
                item["slope_angle"],
                item["vibration"]
            ]], columns=["rainfall", "soil_moisture", "slope_angle", "vibration"])
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            item["model_prediction"] = pred
            item["landslide_probability"] = round(prob, 4)
            item["probability_percent"] = round(prob * 100, 1)

    return scenarios

def generate_random_test_suite(count=50, seed=101):
    """Generates randomized test vectors covering the full operational range."""
    np.random.seed(seed)
    records = []

    for i in range(1, count + 1):
        rain = round(float(np.random.uniform(0.0, 220.0)), 1)
        soil = round(float(np.random.uniform(0.05, 0.98)), 3)
        slope = round(float(np.random.uniform(2.0, 50.0)), 1)
        vib = round(float(np.random.uniform(0.005, 0.55)), 4)

        rec = {
            "test_id": f"TEST-{i:03d}",
            "rainfall": rain,
            "soil_moisture": soil,
            "slope_angle": slope,
            "vibration": vib
        }

        if model is not None:
            df = pd.DataFrame([[rain, soil, slope, vib]], columns=["rainfall", "soil_moisture", "slope_angle", "vibration"])
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            rec["prediction"] = pred
            rec["probability_percent"] = round(prob * 100, 1)
            rec["risk_level"] = "LOW" if prob < 0.25 else ("GUARD" if prob < 0.55 else ("WARNING" if prob < 0.80 else "CRITICAL"))

        records.append(rec)

    return records

def main():
    print("=" * 65)
    print("  GEO-SENTINEL AI: DEMO TEST DATA GENERATOR")
    print("=" * 65)

    curated = generate_curated_showcase()
    curated_json_path = os.path.join(BASE_DIR, "demo_showcase_scenarios.json")
    with open(curated_json_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)
    print(f"✔ Saved {len(curated)} curated showcase scenarios to: {curated_json_path}")

    # Generate 50 random test vectors
    random_vectors = generate_random_test_suite(count=50)
    rand_json_path = os.path.join(BASE_DIR, "demo_random_inputs.json")
    rand_csv_path = os.path.join(BASE_DIR, "demo_random_inputs.csv")

    with open(rand_json_path, "w", encoding="utf-8") as f:
        json.dump(random_vectors, f, indent=2)

    df_rand = pd.DataFrame(random_vectors)
    df_rand.to_csv(rand_csv_path, index=False)

    print(f"✔ Saved {len(random_vectors)} random test vectors to: {rand_json_path}")
    print(f"✔ Saved CSV version to: {rand_csv_path}")
    print("=" * 65)
    print("All demo script data files generated successfully!")

if __name__ == "__main__":
    main()
