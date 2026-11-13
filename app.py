"""
GEO-SENTINEL AI: Landslide Prediction & Geological Telemetry Backend
Flask REST API providing real-time geotechnical inference, continuous telemetry streaming,
scenario generation, and serving the futuristic holographic dashboard.
"""
import os
import time
import datetime
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder=".")
CORS(app)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landslide_prediction_model.pkl")

# Load Machine Learning Model
try:
    model = joblib.load(MODEL_PATH)
    print(f"[GEO-SENTINEL] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[GEO-SENTINEL] WARNING: Failed to load model ({e}). Will attempt to train...")
    from landslide_model import train_model
    model = train_model()

# Rolling historical telemetry buffer (stores last 50 data points)
telemetry_history = []
MAX_HISTORY = 50
total_predictions_served = 0
start_time = time.time()

# Predefined Geological Scenarios
PRESET_SCENARIOS = [
    {
        "id": "stable_bedrock",
        "name": "Stable Bedrock Plateau",
        "description": "Dry sunny conditions on mild granite slope. Minimal seismic vibration.",
        "rainfall": 8.5,
        "soil_moisture": 0.18,
        "slope_angle": 12.0,
        "vibration": 0.02
    },
    {
        "id": "monsoon_early",
        "name": "Early Monsoon Infiltration",
        "description": "Steady continuous rainfall, moderate saturation on medium terrace slope.",
        "rainfall": 75.0,
        "soil_moisture": 0.58,
        "slope_angle": 24.5,
        "vibration": 0.05
    },
    {
        "id": "steep_torrential",
        "name": "Steep Escarpment Cloudburst",
        "description": "Intense torrential downpour on a 36-degree mountain face. High soil liquefaction.",
        "rainfall": 158.0,
        "soil_moisture": 0.82,
        "slope_angle": 36.5,
        "vibration": 0.12
    },
    {
        "id": "seismic_catastrophe",
        "name": "Tectonic Tremor on Waterlogged Incline",
        "description": "Magnitude 5.2 local tremor shaking saturated slope after 3 days of torrential rain.",
        "rainfall": 185.0,
        "soil_moisture": 0.92,
        "slope_angle": 42.0,
        "vibration": 0.42
    },
    {
        "id": "post_wildfire",
        "name": "Post-Wildfire Debris Incline",
        "description": "Vegetation-depleted slope undergoing heavy sudden storm surge.",
        "rainfall": 125.0,
        "soil_moisture": 0.74,
        "slope_angle": 32.0,
        "vibration": 0.08
    }
]

def analyze_risk(rainfall, soil_moisture, slope_angle, vibration, probability):
    """
    Computes geotechnical threat factors, factor of safety, and tactical emergency protocol.
    """
    prob_pct = round(probability * 100, 1)

    # Determine risk category & hazard badge
    if prob_pct < 25.0:
        risk_level = "LOW"
        hazard_code = "SECURE"
        hazard_color = "#00ff88"
        alert_tone = "nominal"
    elif prob_pct < 55.0:
        risk_level = "GUARD"
        hazard_code = "ELEVATED"
        hazard_color = "#ffb700"
        alert_tone = "advisory"
    elif prob_pct < 80.0:
        risk_level = "WARNING"
        hazard_code = "SEVERE"
        hazard_color = "#ff7b00"
        alert_tone = "warning"
    else:
        risk_level = "CRITICAL"
        hazard_code = "EVACUATE"
        hazard_color = "#ff0055"
        alert_tone = "alarm"

    # Geotechnical Factor of Safety (FS) approximation
    # FS > 1.5 is safe; FS = 1.0 is limit equilibrium; FS < 1.0 is slope failure
    factor_of_safety = max(0.5, round(2.5 - (probability * 1.85), 2))

    # Threat contribution analysis
    threat_weights = {
        "Precipitation Surge": min(100, (rainfall / 200.0) * 100),
        "Pore Water Saturation": min(100, (soil_moisture / 0.90) * 100),
        "Slope Shear Stress": min(100, (slope_angle / 45.0) * 100),
        "Seismic Dynamic Load": min(100, (vibration / 0.45) * 100)
    }
    dominant_threat = max(threat_weights.items(), key=lambda x: x[1])[0]

    # Actionable Recommendation Matrix
    if risk_level == "LOW":
        recommendation = "Status Nominal. Terrain stable under current hydrological loading. Routine orbital telemetry active."
        action_step = "STANDBY // LOG TELEMETRY"
    elif risk_level == "GUARD":
        recommendation = "Pore water pressure increasing. Inspect drainage culverts and dispatch automated drone survey to slope perimeter."
        action_step = "ADVISORY // ACTIVATE DRONE SURVEILLANCE"
    elif risk_level == "WARNING":
        recommendation = "High landslide probability! Restrict downhill access roads, prime drainage pump manifolds, issue regional alert."
        action_step = "STAGE 2 WARNING // RESTRICT HIGHWAY ACCESS"
    else:
        recommendation = "CRITICAL HAZARD DETECTED! Immediate downslope evacuation required. Sound audible civil defense sirens and trigger emergency bypass."
        action_step = "STAGE 3 EVACUATE // SOUND ACOUSTIC SIRENS"

    return {
        "risk_level": risk_level,
        "hazard_code": hazard_code,
        "hazard_color": hazard_color,
        "alert_tone": alert_tone,
        "probability_percent": prob_pct,
        "factor_of_safety": factor_of_safety,
        "threat_factors": {k: round(v, 1) for k, v in threat_weights.items()},
        "dominant_threat": dominant_threat,
        "recommendation": recommendation,
        "action_step": action_step
    }

def process_telemetry(rainfall, soil_moisture, slope_angle, vibration):
    """
    Standardizes input, queries model, and formats full response payload.
    """
    global total_predictions_served

    # Auto-normalize soil moisture if passed as percentage (e.g., 65 -> 0.65)
    if soil_moisture > 1.0:
        soil_moisture = soil_moisture / 100.0

    soil_moisture = max(0.0, min(1.0, float(soil_moisture)))
    rainfall = max(0.0, float(rainfall))
    slope_angle = max(0.0, min(90.0, float(slope_angle)))
    vibration = max(0.0, float(vibration))

    # Prepare DataFrame with feature names to match model training
    input_df = pd.DataFrame(
        [[rainfall, soil_moisture, slope_angle, vibration]],
        columns=["rainfall", "soil_moisture", "slope_angle", "vibration"]
    )

    t0 = time.perf_counter()
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    total_predictions_served += 1
    analysis = analyze_risk(rainfall, soil_moisture, slope_angle, vibration, probability)

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    iso_time = datetime.datetime.now().isoformat()

    record = {
        "packet_id": total_predictions_served,
        "timestamp": timestamp,
        "iso_time": iso_time,
        "rainfall": round(rainfall, 1),
        "soil_moisture": round(soil_moisture * 100, 1),  # in % for UI display
        "soil_moisture_ratio": round(soil_moisture, 3),
        "slope_angle": round(slope_angle, 1),
        "vibration": round(vibration, 3),
        "prediction": prediction,
        "probability": probability,
        "probability_percent": analysis["probability_percent"],
        "risk_level": analysis["risk_level"],
        "hazard_code": analysis["hazard_code"],
        "hazard_color": analysis["hazard_color"],
        "factor_of_safety": analysis["factor_of_safety"],
        "dominant_threat": analysis["dominant_threat"],
        "threat_factors": analysis["threat_factors"],
        "recommendation": analysis["recommendation"],
        "action_step": analysis["action_step"],
        "inference_ms": max(0.1, inference_ms)
    }

    # Append to rolling history
    telemetry_history.append(record)
    if len(telemetry_history) > MAX_HISTORY:
        telemetry_history.pop(0)

    return record

# Seed initial baseline reading so dashboard is never empty
if not telemetry_history:
    process_telemetry(rainfall=18.0, soil_moisture=0.25, slope_angle=15.0, vibration=0.03)

# ==================== ROUTE HANDLERS ====================

@app.route("/")
def index():
    """Serves the main futuristic dashboard."""
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts landslide hazard from incoming sensor telemetry."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    try:
        rainfall = float(data.get("rainfall", 0.0))
        soil_moisture = float(data.get("soil_moisture", 0.0))
        slope_angle = float(data.get("slope_angle", 0.0))
        vibration = float(data.get("vibration", 0.0))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid numeric parameter: {str(e)}"}), 400

    result = process_telemetry(rainfall, soil_moisture, slope_angle, vibration)
    return jsonify(result)

@app.route("/latest", methods=["GET"])
def latest():
    """Returns the most recent telemetry packet and assessment."""
    if telemetry_history:
        return jsonify(telemetry_history[-1])
    # Fallback default
    return jsonify(process_telemetry(rainfall=15.0, soil_moisture=0.20, slope_angle=14.0, vibration=0.02))

@app.route("/update_telemetry", methods=["POST"])
def update_telemetry():
    """Endpoint for background demo scripts / external sensor streams."""
    data = request.get_json(silent=True) or {}
    rainfall = float(data.get("rainfall", random.uniform(5, 160)))
    soil_moisture = float(data.get("soil_moisture", random.uniform(0.1, 0.9)))
    slope_angle = float(data.get("slope_angle", random.uniform(10, 42)))
    vibration = float(data.get("vibration", random.uniform(0.01, 0.35)))

    result = process_telemetry(rainfall, soil_moisture, slope_angle, vibration)
    return jsonify({
        "status": "success",
        "message": "Telemetry ingested and processed by GEO-SENTINEL AI",
        "telemetry": result
    })

@app.route("/history", methods=["GET"])
def history():
    """Returns rolling historical telemetry points for chart streaming."""
    return jsonify({
        "count": len(telemetry_history),
        "history": telemetry_history
    })

@app.route("/scenarios", methods=["GET"])
def scenarios():
    """Returns preset mission scenarios."""
    return jsonify(PRESET_SCENARIOS)

@app.route("/random_scenario", methods=["GET"])
def random_scenario():
    """Generates a dynamic randomized realistic geological scenario."""
    category = random.choice(["safe", "moderate", "high", "critical"])
    
    if category == "safe":
        rain = round(random.uniform(2.0, 45.0), 1)
        soil = round(random.uniform(0.10, 0.40), 2)
        slope = round(random.uniform(5.0, 20.0), 1)
        vib = round(random.uniform(0.01, 0.06), 3)
    elif category == "moderate":
        rain = round(random.uniform(55.0, 110.0), 1)
        soil = round(random.uniform(0.45, 0.68), 2)
        slope = round(random.uniform(20.0, 30.0), 1)
        vib = round(random.uniform(0.04, 0.12), 3)
    elif category == "high":
        rain = round(random.uniform(115.0, 165.0), 1)
        soil = round(random.uniform(0.70, 0.85), 2)
        slope = round(random.uniform(30.0, 39.0), 1)
        vib = round(random.uniform(0.10, 0.25), 3)
    else:  # critical
        rain = round(random.uniform(160.0, 215.0), 1)
        soil = round(random.uniform(0.82, 0.96), 2)
        slope = round(random.uniform(36.0, 48.0), 1)
        vib = round(random.uniform(0.24, 0.55), 3)

    result = process_telemetry(rain, soil, slope, vib)
    return jsonify(result)

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Processes an array of telemetry sensor readings in one batch."""
    payload = request.get_json(silent=True) or {}
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"error": "Field 'samples' must be a non-empty array of objects"}), 400

    results = []
    for item in samples:
        r = float(item.get("rainfall", 0))
        m = float(item.get("soil_moisture", 0))
        s = float(item.get("slope_angle", 0))
        v = float(item.get("vibration", 0))
        results.append(process_telemetry(r, m, s, v))

    return jsonify({"processed_count": len(results), "results": results})

@app.route("/status", methods=["GET"])
def status():
    """System health check and diagnostic information."""
    uptime_sec = round(time.time() - start_time, 1)
    return jsonify({
        "system": "GEO-SENTINEL Planetary Landslide Warning System",
        "version": "2.4.0-CYBER",
        "status": "OPERATIONAL",
        "model_type": type(model).__name__,
        "uptime_seconds": uptime_sec,
        "total_inferences": total_predictions_served,
        "buffer_size": len(telemetry_history),
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"================================================================")
    print(f"  GEO-SENTINEL AI TELEMETRY ENGINE ONLINE")
    print(f"  Access Futuristic Dashboard at: http://127.0.0.1:{port}")
    print(f"  Inference API: POST http://127.0.0.1:{port}/predict")
    print(f"================================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
