"""
GEO-SENTINEL AI: Project Master Launcher
Starts the Flask telemetry server and automatically opens the futuristic dashboard in your browser.
"""
import os
import sys
import time
import webbrowser
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "landslide_prediction_model.pkl")

def ensure_model():
    if not os.path.exists(MODEL_FILE):
        print("[LAUNCHER] Model file not found. Training model now...")
        from landslide_model import train_model
        train_model()

def open_browser():
    time.sleep(1.2)
    print("\n[LAUNCHER] Opening futuristic dashboard in default browser...")
    webbrowser.open("http://127.0.0.1:5000")

def main():
    print("=" * 70)
    print("   GEO-SENTINEL AI // AUTONOMOUS LANDSLIDE WARNING SYSTEM")
    print("   Starting System Core & High-Tech Mission Dashboard...")
    print("=" * 70)

    ensure_model()

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Import and run app
    from app import app
    print("[LAUNCHER] Server active at: http://127.0.0.1:5000")
    print("[LAUNCHER] Press Ctrl+C in this terminal to stop.")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()
