"""
GEO-SENTINEL AI - Autonomous Telemetry Demo Runner
Generates randomized geological sensor telemetry, queries the trained model (or live API),
and feeds live real-time simulation streams directly into the futuristic frontend dashboard.

Usage:
    python demo_runner.py --random          # Run single random input test
    python demo_runner.py --scenarios       # Run all preset disaster scenarios
    python demo_runner.py --stream          # Continuous live stream to frontend (1 packet/sec)
    python demo_runner.py --batch 10        # Run 10 random batch simulations
    python demo_runner.py --export          # Export random test payloads to JSON & CSV
"""
import sys
import os
import time
import json
import random
import argparse
import datetime
import urllib.request
import urllib.error

# Ensure base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landslide_prediction_model.pkl")

# Try to import joblib & pandas for local standalone inference if API is not running
try:
    import joblib
    import pandas as pd
    LOCAL_MODEL = joblib.load(MODEL_PATH)
except Exception:
    LOCAL_MODEL = None

API_URL = "http://127.0.0.1:5000"

def generate_random_telemetry(profile="mixed"):
    """
    Generates realistic random geological telemetry based on chosen risk profile.
    Profiles: 'safe', 'elevated', 'severe', 'catastrophic', 'mixed'
    """
    if profile == "mixed":
        profile = random.choice(["safe", "elevated", "severe", "catastrophic"])

    if profile == "safe":
        rain = round(random.uniform(5.0, 50.0), 1)
        soil = round(random.uniform(0.12, 0.40), 2)
        slope = round(random.uniform(6.0, 20.0), 1)
        vib = round(random.uniform(0.01, 0.05), 3)
        label = "Safe Baseline / Valley Terrain"
    elif profile == "elevated":
        rain = round(random.uniform(60.0, 110.0), 1)
        soil = round(random.uniform(0.48, 0.68), 2)
        slope = round(random.uniform(22.0, 31.0), 1)
        vib = round(random.uniform(0.04, 0.12), 3)
        label = "Elevated Moisture / Terrace Incline"
    elif profile == "severe":
        rain = round(random.uniform(120.0, 165.0), 1)
        soil = round(random.uniform(0.72, 0.86), 2)
        slope = round(random.uniform(32.0, 39.0), 1)
        vib = round(random.uniform(0.12, 0.28), 3)
        label = "Severe Cloudburst / Saturated Escarpment"
    else:  # catastrophic
        rain = round(random.uniform(160.0, 215.0), 1)
        soil = round(random.uniform(0.85, 0.98), 2)
        slope = round(random.uniform(38.0, 48.0), 1)
        vib = round(random.uniform(0.28, 0.58), 3)
        label = "Catastrophic Storm Surge + Seismic Tremor"

    return {
        "profile": profile,
        "label": label,
        "rainfall": rain,
        "soil_moisture": soil,
        "slope_angle": slope,
        "vibration": vib,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }

def send_to_api(payload, endpoint="/update_telemetry"):
    """Sends payload to local Flask backend via HTTP."""
    url = f"{API_URL}{endpoint}"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None

def predict_local(payload):
    """Fallback local inference directly from model pickle if Flask is offline."""
    if LOCAL_MODEL is None:
        return {"error": "Local model not available and API unreachable"}

    rain = payload["rainfall"]
    soil = payload["soil_moisture"]
    slope = payload["slope_angle"]
    vib = payload["vibration"]

    df = pd.DataFrame([[rain, soil, slope, vib]], columns=["rainfall", "soil_moisture", "slope_angle", "vibration"])
    pred = int(LOCAL_MODEL.predict(df)[0])
    prob = float(LOCAL_MODEL.predict_proba(df)[0][1])
    prob_pct = round(prob * 100, 1)

    if prob_pct < 25.0:
        level = "LOW"
    elif prob_pct < 55.0:
        level = "GUARD"
    elif prob_pct < 80.0:
        level = "WARNING"
    else:
        level = "CRITICAL"

    return {
        "rainfall": rain,
        "soil_moisture": round(soil * 100, 1),
        "slope_angle": slope,
        "vibration": vib,
        "prediction": pred,
        "probability_percent": prob_pct,
        "risk_level": level,
        "timestamp": payload.get("timestamp", datetime.datetime.now().strftime("%H:%M:%S"))
    }

def print_banner():
    banner = r"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║    ██████╗ ███████╗ ██████╗       ███████╗███████╗███╗   ██╗████████╗    ║
  ║   ██╔════╝ ██╔════╝██╔═══██╗      ██╔════╝██╔════╝████╗  ██║╚══██╔══╝    ║
  ║   ██║  ███╗█████╗  ██║   ██║█████╗███████╗█████╗  ██╔██╗ ██║   ██║       ║
  ║   ██║   ██║██╔══╝  ██║   ██║╚════╝╚════██║██╔══╝  ██║╚██╗██║   ██║       ║
  ║   ╚██████╔╝███████╗╚██████╔╝      ███████║███████╗██║ ╚████║   ██║       ║
  ║    ╚═════╝ ╚══════╝ ╚═════╝       ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ║
  ║           PLANETARY GEOLOGICAL DEFENSE & LANDSLIDE TELEMETRY              ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def render_ascii_result(sample, result, source="API"):
    """Prints a beautiful futuristic telemetry inspection card in console."""
    prob = result.get("probability_percent", 0.0)
    level = result.get("risk_level", "UNKNOWN")
    hazard = result.get("hazard_code", level)
    fs = result.get("factor_of_safety", 1.5)
    rec = result.get("recommendation", "Monitoring active.")

    # Visual risk bar
    bar_len = 30
    filled = int((prob / 100.0) * bar_len)
    risk_bar = "█" * filled + "░" * (bar_len - filled)

    color_tag = "[OK]" if level == "LOW" else ("[WARN]" if level in ("GUARD", "WARNING") else "[ALERT]")

    print(f"\n┌─── TELEMETRY INGESTION PACKET [{sample.get('timestamp')}] (Source: {source}) ─────────┐")
    print(f"│ Scenario Profile  : {sample.get('label', 'Dynamic Random Vector')}")
    print(f"│ ├── Rainfall      : {sample['rainfall']:>6.1f} mm/hr")
    print(f"│ ├── Soil Moisture : {sample['soil_moisture'] * 100:>6.1f} %")
    print(f"│ ├── Slope Angle   : {sample['slope_angle']:>6.1f} °")
    print(f"│ └── Vibration     : {sample['vibration']:>6.3f} g (RMS)")
    print(f"├────────────────────────────────────────────────────────────────────────┤")
    print(f"│ PREDICTED RISK    : {color_tag} {level:<8} [{hazard}]")
    print(f"│ Hazard Probability: {prob:>5.1f}%  [{risk_bar}]")
    print(f"│ Factor of Safety  : {fs:>5.2f}  (Failure Limit = 1.0)")
    print(f"│ Action Protocol   : {rec}")
    print(f"└────────────────────────────────────────────────────────────────────────┘")

def run_single_random():
    """Generates 1 random telemetry point and prints analysis."""
    sample = generate_random_telemetry()
    api_res = send_to_api(sample)
    if api_res and "telemetry" in api_res:
        render_ascii_result(sample, api_res["telemetry"], source="LIVE FLASK BACKEND")
        print("\n✔ Telemetry successfully streamed to frontend at http://127.0.0.1:5000!")
    else:
        local_res = predict_local(sample)
        render_ascii_result(sample, local_res, source="STANDALONE LOCAL MODEL")
        print("\nℹ [Flask backend offline - evaluated using local model file]")

def run_scenarios():
    """Steps through all 5 preset scenarios."""
    scenarios = [
        {"rainfall": 8.0, "soil_moisture": 0.15, "slope_angle": 10.0, "vibration": 0.02, "label": "1. Stable Granite Ridge"},
        {"rainfall": 68.0, "soil_moisture": 0.52, "slope_angle": 22.0, "vibration": 0.06, "label": "2. Pre-Monsoon Ground Infiltration"},
        {"rainfall": 130.0, "soil_moisture": 0.78, "slope_angle": 33.0, "vibration": 0.14, "label": "3. Saturated Mountain Escarpment"},
        {"rainfall": 178.0, "soil_moisture": 0.91, "slope_angle": 42.0, "vibration": 0.38, "label": "4. Flash Tectonic Tremor in Cloudburst"},
        {"rainfall": 115.0, "soil_moisture": 0.65, "slope_angle": 28.0, "vibration": 0.08, "label": "5. Wildfire Silt Runoff"}
    ]
    print(f"\n[DEMO RUNNER] Testing {len(scenarios)} Geological Disaster Scenarios...\n")
    for s in scenarios:
        s["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
        res = send_to_api(s)
        if res and "telemetry" in res:
            render_ascii_result(s, res["telemetry"], source="LIVE FLASK BACKEND")
        else:
            local_res = predict_local(s)
            render_ascii_result(s, local_res, source="STANDALONE MODEL")
        time.sleep(0.8)
    print("\n✔ All scenarios completed and streamed to dashboard!")

def run_stream(interval=1.2, count=30):
    """
    Runs a continuous live telemetry feed to the frontend.
    The browser dashboard will show real-time changes!
    """
    print(f"\n[DEMO RUNNER] Starting real-time simulation stream to {API_URL}...")
    print(f"[DEMO RUNNER] Open http://127.0.0.1:5000 in your browser to view the live dashboard.")
    print(f"[DEMO RUNNER] Press Ctrl+C at any time to halt stream.\n")

    # Start with a gentle drift simulation
    rain = 25.0
    soil = 0.30
    slope = 18.0
    vib = 0.03

    packet_id = 0
    try:
        while True:
            packet_id += 1
            # Random walk drift with occasional storm surges
            if random.random() < 0.15:
                # Surge spike
                rain = min(210.0, max(5.0, rain + random.uniform(20.0, 45.0)))
                soil = min(0.96, max(0.15, soil + random.uniform(0.08, 0.18)))
                vib = min(0.55, max(0.01, vib + random.uniform(0.05, 0.20)))
            else:
                rain = min(210.0, max(5.0, rain + random.uniform(-15.0, 15.0)))
                soil = min(0.96, max(0.15, soil + random.uniform(-0.04, 0.04)))
                vib = min(0.55, max(0.01, vib + random.uniform(-0.03, 0.03)))

            sample = {
                "rainfall": round(rain, 1),
                "soil_moisture": round(soil, 3),
                "slope_angle": round(slope, 1),
                "vibration": round(vib, 3),
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

            res = send_to_api(sample)
            if res and "telemetry" in res:
                t = res["telemetry"]
                prob = t["probability_percent"]
                lvl = t["risk_level"]
                bar = "█" * int(prob / 5)
                print(f"[Packet #{packet_id:03d} | {sample['timestamp']}] Rain: {rain:>5.1f}mm | Soil: {soil*100:>5.1f}% | Slope: {slope:>4.1f}° | Vib: {vib:>5.3f}g -> RISK: {lvl:<8} ({prob:>5.1f}%) {bar}")
            else:
                local_res = predict_local(sample)
                prob = local_res["probability_percent"]
                lvl = local_res["risk_level"]
                print(f"[Packet #{packet_id:03d} | Local] Rain: {rain:>5.1f}mm | Soil: {soil*100:>5.1f}% | Slope: {slope:>4.1f}° | Vib: {vib:>5.3f}g -> RISK: {lvl} ({prob}%)")

            if count and packet_id >= count:
                print(f"\n✔ Completed {count} stream packets.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n[DEMO RUNNER] Simulation stream paused by user.")

def run_batch(count=10):
    """Generates and evaluates N random samples in batch."""
    print(f"\n[DEMO RUNNER] Generating and evaluating batch of {count} random telemetry records...\n")
    samples = [generate_random_telemetry() for _ in range(count)]
    
    # Try batch API
    payload = {"samples": samples}
    res = send_to_api(payload, endpoint="/batch_predict")
    
    table_rows = []
    if res and "results" in res:
        for item in res["results"]:
            table_rows.append((
                item["timestamp"],
                item["rainfall"],
                item["soil_moisture"],
                item["slope_angle"],
                item["vibration"],
                item["probability_percent"],
                item["risk_level"]
            ))
    else:
        for s in samples:
            lr = predict_local(s)
            table_rows.append((
                lr["timestamp"],
                lr["rainfall"],
                lr["soil_moisture"],
                lr["slope_angle"],
                lr["vibration"],
                lr["probability_percent"],
                lr["risk_level"]
            ))

    print(f"{'TIMESTAMP':<10} {'RAIN(mm)':<10} {'SOIL(%)':<10} {'SLOPE(°)':<10} {'VIB(g)':<10} {'PROB(%)':<10} {'HAZARD':<10}")
    print("-" * 75)
    for r in table_rows:
        print(f"{r[0]:<10} {r[1]:<10.1f} {r[2]:<10.1f} {r[3]:<10.1f} {r[4]:<10.3f} {r[5]:<10.1f} {r[6]:<10}")
    print("-" * 75)

def export_samples(count=20):
    """Generates random demo inputs and exports them to JSON & CSV files."""
    samples = []
    for i in range(count):
        s = generate_random_telemetry()
        # compute prediction
        loc = predict_local(s)
        s["predicted_probability"] = loc.get("probability_percent", 0.0)
        s["risk_level"] = loc.get("risk_level", "LOW")
        samples.append(s)

    json_path = os.path.join(BASE_DIR, "demo_inputs.json")
    csv_path = os.path.join(BASE_DIR, "demo_inputs.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    df = pd.DataFrame(samples)
    df.to_csv(csv_path, index=False)

    print(f"\n✔ Exported {count} random demo telemetry samples to:")
    print(f"  * JSON : {json_path}")
    print(f"  * CSV  : {csv_path}\n")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="GEO-SENTINEL AI Telemetry Demo Runner")
    parser.add_argument("--random", action="store_true", help="Generate single random telemetry vector")
    parser.add_argument("--scenarios", action="store_true", help="Cycle through preset disaster scenarios")
    parser.add_argument("--stream", action="store_true", help="Stream live telemetry to frontend (real-time)")
    parser.add_argument("--interval", type=float, default=1.2, help="Streaming interval in seconds (default 1.2s)")
    parser.add_argument("--batch", type=int, default=0, help="Evaluate N random samples in batch table")
    parser.add_argument("--export", action="store_true", help="Export random demo datasets to JSON and CSV")

    args = parser.parse_args()

    if args.stream:
        run_stream(interval=args.interval)
    elif args.scenarios:
        run_scenarios()
    elif args.batch > 0:
        run_batch(args.batch)
    elif args.export:
        export_samples(25)
    elif args.random:
        run_single_random()
    else:
        # Default interactive behavior if no flags: run 1 random test and prompt options
        print("[Mode: Auto-Demonstration]")
        run_single_random()
        print("\nTip: Run with --stream to watch the frontend dashboard update live!")
        print("Example: python demo_runner.py --stream")
        print("Example: python demo_runner.py --scenarios")
        print("Example: python demo_runner.py --export")

if __name__ == "__main__":
    main()
