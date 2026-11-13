"""
========================================================================================
  TERRA-GUARD AI: RANDOM TELEMETRY GENERATOR & MODEL SHOWCASE SCRIPT
========================================================================================
Generates randomized realistic geological sensor inputs (Rainfall, Soil Moisture,
Slope Angle, Vibration), passes them directly to the machine learning model via the
backend API, and displays the results live on the frontend dashboard.

Usage:
    python random_generator.py             # Interactive mode (generate on Enter)
    python random_generator.py --once      # Send single random sample and exit
    python random_generator.py --stream    # Continuous stream (1 sample every 1.5s)
    python random_generator.py --count 5   # Send exactly 5 random samples
========================================================================================
"""
import sys
import os
import time
import json
import random
import datetime
import argparse
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:5000"

def generate_random_telemetry():
    """
    Generates realistic random geological telemetry covering all threat stages:
    - Normal sunny / dry conditions (Low Risk)
    - Moderate soaking rains on medium slopes (Caution / Guard)
    - Heavy cloudbursts on steep mountain faces (High Risk)
    - Flash torrential storms with seismic shaking (Critical Evacuate)
    """
    profile = random.choices(
        ["safe", "elevated", "high_risk", "critical"],
        weights=[0.35, 0.25, 0.25, 0.15],
        k=1
    )[0]

    if profile == "safe":
        rain = round(random.uniform(2.0, 45.0), 1)
        soil = round(random.uniform(0.10, 0.38), 2)
        slope = round(random.uniform(5.0, 19.0), 1)
        vib = round(random.uniform(0.008, 0.045), 3)
        profile_name = "Stable Terrain / Baseline"
    elif profile == "elevated":
        rain = round(random.uniform(50.0, 105.0), 1)
        soil = round(random.uniform(0.45, 0.65), 2)
        slope = round(random.uniform(20.0, 29.0), 1)
        vib = round(random.uniform(0.035, 0.10), 3)
        profile_name = "Monsoon Saturation / Infiltration"
    elif profile == "high_risk":
        rain = round(random.uniform(110.0, 160.0), 1)
        soil = round(random.uniform(0.68, 0.85), 2)
        slope = round(random.uniform(30.0, 39.0), 1)
        vib = round(random.uniform(0.09, 0.24), 3)
        profile_name = "Steep Slope Cloudburst"
    else:  # critical
        rain = round(random.uniform(160.0, 215.0), 1)
        soil = round(random.uniform(0.82, 0.98), 2)
        slope = round(random.uniform(38.0, 48.0), 1)
        vib = round(random.uniform(0.24, 0.55), 3)
        profile_name = "Torrential Storm Surge + Tremor"

    return {
        "profile": profile_name,
        "rainfall": rain,
        "soil_moisture": soil,
        "slope_angle": slope,
        "vibration": vib,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }

def send_to_backend(payload):
    """Sends random telemetry to the Flask backend and retrieves the model's prediction."""
    url = f"{API_URL}/update_telemetry"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return None

def print_result_card(sample, result):
    """Prints a clean, professional summary of input and model output."""
    prob = result.get("probability_percent", 0.0)
    level = result.get("risk_level", "UNKNOWN")
    hazard = result.get("hazard_code", level)
    fs = result.get("factor_of_safety", 1.5)
    packet_id = result.get("packet_id", "#")
    dominant = result.get("dominant_threat", "Precipitation")
    rec = result.get("recommendation", "Monitoring active.")

    # Status tag
    if level == "LOW":
        badge = "🟢 [STABLE - LOW RISK]"
    elif level == "GUARD":
        badge = "🟡 [ELEVATED - CAUTION]"
    elif level == "WARNING":
        badge = "🟠 [HIGH RISK - ALERT]"
    else:
        badge = "🔴 [CRITICAL - EVACUATE]"

    # Progress visualizer bar
    bar_len = 25
    filled = int((prob / 100.0) * bar_len)
    risk_bar = "█" * filled + "░" * (bar_len - filled)

    print("-" * 72)
    print(f"  RANDOM TELEMETRY PACKET #{packet_id} [{sample['timestamp']}]")
    print(f"  Category: {sample['profile']}")
    print("-" * 72)
    print("  INPUTS PASSED TO MODEL:")
    print(f"    • Rainfall (Precipitation)  : {sample['rainfall']:>6.1f} mm/hr")
    print(f"    • Soil Moisture Content     : {sample['soil_moisture'] * 100:>6.1f} %")
    print(f"    • Topographic Slope Angle   : {sample['slope_angle']:>6.1f} °")
    print(f"    • Seismic Ground Vibration  : {sample['vibration']:>6.3f} g (RMS)")
    print()
    print("  MODEL PREDICTION OUTPUT:")
    print(f"    • Hazard Status             : {badge}")
    print(f"    • Landslide Probability     : {prob:>5.1f}%  [{risk_bar}]")
    print(f"    • Factor of Safety (FS)     : {fs:>5.2f}  (Failure Limit = 1.0)")
    print(f"    • Primary Risk Driver       : {dominant}")
    print(f"    • Tactical Recommendation   : {rec}")
    print("-" * 72)
    print("  ✔ Result dispatched live to frontend dashboard at http://127.0.0.1:5000\n")

def execute_once():
    """Generates 1 random input, passes to model, and updates frontend."""
    sample = generate_random_telemetry()
    res = send_to_backend(sample)
    if res and "telemetry" in res:
        print_result_card(sample, res["telemetry"])
        return True
    else:
        print("\n[ERROR] Unable to reach backend at http://127.0.0.1:5000")
        print("Please ensure 'python app.py' is running in another terminal.\n")
        return False

def execute_stream(interval=1.5, count=None):
    """Continuously generates random telemetry and streams to the dashboard."""
    print(f"\n[STREAM MODE] Streaming live random telemetry to http://127.0.0.1:5000 every {interval}s")
    print("[STREAM MODE] Open your browser to view the real-time gauges and slope simulator!")
    print("[STREAM MODE] Press Ctrl+C at any time to halt stream.\n")

    n = 0
    try:
        while True:
            n += 1
            sample = generate_random_telemetry()
            res = send_to_backend(sample)
            if res and "telemetry" in res:
                t = res["telemetry"]
                prob = t["probability_percent"]
                lvl = t["risk_level"]
                bar = "█" * int(prob / 5)
                print(f"[Packet #{t.get('packet_id', n):03d} | {sample['timestamp']}] Rain: {sample['rainfall']:>5.1f}mm | Soil: {sample['soil_moisture']*100:>5.1f}% | Slope: {sample['slope_angle']:>4.1f}° | Vib: {sample['vibration']:>5.3f}g -> {lvl:<8} ({prob:>5.1f}%) {bar}")
            else:
                print(f"[Packet #{n}] Backend offline at {API_URL}")

            if count and n >= count:
                print(f"\n✔ Completed {count} telemetry packets.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[STREAM MODE] Stream halted by user.")

def main():
    parser = argparse.ArgumentParser(description="TERRA-GUARD AI Random Telemetry Generator")
    parser.add_argument("--once", action="store_true", help="Generate single random sample and exit")
    parser.add_argument("--stream", action="store_true", help="Run continuous real-time telemetry stream")
    parser.add_argument("--interval", type=float, default=1.5, help="Stream interval in seconds (default 1.5s)")
    parser.add_argument("--count", type=int, default=0, help="Send exactly N random samples")

    args = parser.parse_args()

    print("=" * 72)
    print("   TERRA-GUARD AI // RANDOM GEOTECHNICAL TELEMETRY GENERATOR")
    print("=" * 72)

    if args.stream:
        execute_stream(interval=args.interval)
    elif args.count > 0:
        execute_stream(interval=args.interval, count=args.count)
    elif args.once:
        execute_once()
    else:
        # Interactive mode: send 1 sample and wait for user to press ENTER for more
        execute_once()
        print("Interactive Demo Ready:")
        print("• Press [ENTER] to generate and send another random input to the dashboard")
        print("• Type 's' and press [ENTER] to start continuous streaming")
        print("• Type 'q' and press [ENTER] to exit\n")

        while True:
            try:
                user_input = input(">> Action (Enter = Next Random / 's' = Stream / 'q' = Quit): ").strip().lower()
                if user_input == 'q':
                    print("Exiting generator. Have a great demo!")
                    break
                elif user_input == 's':
                    execute_stream(interval=1.5)
                    break
                else:
                    execute_once()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

if __name__ == "__main__":
    main()
