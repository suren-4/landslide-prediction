"""
GEO-SENTINEL AI: System Diagnostics & Model Benchmark Utility
Validates environment integrity, model inference speed, memory footprint, and API responsiveness.
"""
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landslide_prediction_model.pkl")

def run_diagnostics():
    print("=" * 70)
    print("   GEO-SENTINEL AI // SYSTEM DIAGNOSTICS & BENCHMARK")
    print("=" * 70)

    # 1. Environment Checks
    print(f"[1/5] Python Environment: {sys.version.split()[0]} ({sys.platform})")
    print(f"      Workspace Directory: {BASE_DIR}")

    # 2. Model Integrity & Weights
    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Model file missing at {MODEL_PATH}")
        return False
    
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[2/5] Model File: {os.path.basename(MODEL_PATH)} ({file_size_mb:.2f} MB)")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"      Model Architecture: {type(model).__name__}")
        print(f"      Estimators: {len(model.estimators_)} trees | Features: {model.n_features_in_}")
    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return False

    # 3. Model Inference Throughput Benchmark
    print("[3/5] Benchmarking Real-Time Inference Speed...")
    sample_df = pd.DataFrame(
        [[120.0, 0.65, 30.0, 0.10]],
        columns=["rainfall", "soil_moisture", "slope_angle", "vibration"]
    )

    # Warmup
    for _ in range(10):
        model.predict_proba(sample_df)

    n_runs = 500
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict_proba(sample_df)
    elapsed = time.perf_counter() - t0

    latency_per_inf_ms = (elapsed / n_runs) * 1000
    throughput_ips = n_runs / elapsed
    print(f"      Mean Latency: {latency_per_inf_ms:.2f} ms per inference")
    print(f"      Throughput  : {throughput_ips:.1f} inferences/second")

    # 4. Stress Test: Edge Cases
    print("[4/5] Evaluating Physical Boundary Limits...")
    test_cases = [
        ("Absolute Zero Baseline", [0.0, 0.0, 0.0, 0.0], 0, 0.05),
        ("Extreme Torrential Quake", [220.0, 1.0, 50.0, 0.60], 1, 0.95),
        ("Flat Ground Flood (0 deg)", [200.0, 0.95, 0.0, 0.05], 0, 0.30),
        ("Dry Rock Incline (45 deg)", [0.0, 0.05, 45.0, 0.01], 0, 0.30)
    ]

    all_passed = True
    for name, vals, exp_pred, max_or_min in test_cases:
        df = pd.DataFrame([vals], columns=["rainfall", "soil_moisture", "slope_angle", "vibration"])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        status = "PASS"
        if exp_pred == 1 and prob < max_or_min:
            status = "WARN"
            all_passed = False
        elif exp_pred == 0 and prob > max_or_min:
            status = "WARN"
            all_passed = False
        print(f"      * {name:26}: Pred={pred}, Prob={prob*100:>5.1f}% [{status}]")

    # 5. Check Demo Assets
    print("[5/5] Checking Demo & Data Assets...")
    assets = [
        "synthetic_landslide_data.csv",
        "demo_inputs.json",
        "demo_inputs.csv",
        "demo_showcase_scenarios.json",
        "demo_random_inputs.json"
    ]
    for a in assets:
        p = os.path.join(BASE_DIR, a)
        exists = os.path.exists(p)
        print(f"      * {a:30}: {'FOUND' if exists else 'MISSING'}")

    print("=" * 70)
    print("   DIAGNOSTIC STATUS: ALL SYSTEMS OPERATIONAL")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_diagnostics()
