# 🛰️ GEO-SENTINEL AI // Autonomous Landslide Early Warning System

An advanced AI-powered geotechnical early warning and geological telemetry defense platform. **GEO-SENTINEL AI** processes real-time hydrological, soil saturation, slope gradient, and seismic shockwave metrics to predict slope failure and trigger multi-stage civil defense protocols.

---

## ⚡ Quick Start (One-Click Launch)

### Option 1: Double-Click Batch File (Windows)
Double-click `start_demo.bat` in the project folder. It verifies your environment, starts the backend, and automatically opens the futuristic dashboard in your default browser.

### Option 2: Python Launcher
```bash
python run_project.py
```
This boots the Flask telemetry server on `http://127.0.0.1:5000` and launches your browser automatically.

### Option 3: Standard Flask Run
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in any modern web browser.

---

## 🎲 Random Telemetry Generator & Live Showcase

To demonstrate the system, run the dedicated generator script in your terminal. It generates randomized realistic sensor inputs, passes them to the machine learning model, and dispatches the results to the frontend dashboard in real time:

```bash
python random_generator.py
```
- **Interactive Mode**: Press `[ENTER]` to generate and send a new random sample. Watch the browser update immediately! Type `s` for continuous live streaming, or `q` to quit.
- **Single Shot Mode**: `python random_generator.py --once`
- **Continuous Stream Mode**: `python random_generator.py --stream` (sends a new random telemetry vector every 1.5s)
- **Batch Count**: `python random_generator.py --count 5`

---

### 1. `demo_runner.py` (Interactive & Automated Demo Engine)
| Command | Purpose |
|---|---|
| `python demo_runner.py --random` | Generates a single randomized geological telemetry vector, queries the model/API, and prints a futuristic cyber ASCII terminal report with ASCII gauges. |
| `python demo_runner.py --scenarios` | Automatically cycles through 5 distinct preset geological disaster scenarios (Stable Granite Ridge, Monsoon Infiltration, Escarpment Cloudburst, Tectonic Tremor, Wildfire Debris). |
| `python demo_runner.py --stream` | **Continuous Live Telemetry Stream!** Generates fluctuating sensor data every 1.2s and streams it directly into the frontend dashboard in real time! |
| `python demo_runner.py --batch 10` | Evaluates $N$ random telemetry records in batch and prints a formatted hazard audit table. |
| `python demo_runner.py --export` | Generates and exports randomized test vectors to `demo_inputs.json` and `demo_inputs.csv`. |

### 2. `generate_demo_data.py` (Test Suite & Dataset Generator)
Generates structured offline datasets for demo presentations, edge-case testing, and benchmarks:
- `demo_showcase_scenarios.json`: 8 hand-crafted geotechnical scenarios with ground-truth physics.
- `demo_random_inputs.json` & `.csv`: 50 diverse randomized test vectors with pre-computed probabilities.

```bash
python generate_demo_data.py
```

### 3. `New.py` (Geotechnical Dataset Generator)
Simulates 3,500 samples modeling Mohr-Coulomb slope stability mechanics with continuous logistic failure curves and realistic noise.
```bash
python New.py
```

### 4. `landslide_model.py` (Model Training & Evaluation)
Trains the Random Forest classifier (93.0% accuracy, 98.5% ROC-AUC), evaluates confusion matrix and feature importances, and exports `landslide_prediction_model.pkl`.
```bash
python landslide_model.py
```

---

## 🚀 Innovative Futuristic Frontend Highlights

1. **Cyberpunk Aerospace Holographic HUD**:
   - Deep space palette (`#060911`), glowing neon cyan, matrix green, warning amber, and alert crimson accents.
   - Dual-clock telemetry showing synchronized **UTC** and **Local Time**.
   - Ambient scan lines, animated grid, and corner reticles.

2. **4 Real-Time Interactive Sensor Holographic Cards**:
   - **Precipitation Surge ($mm/hr$)**: Dual control with visual surge progress gauge ($0-220\text{ mm}$).
   - **Pore Saturation ($\%$)**: Soil moisture reservoir with liquid liquefaction threshold indicator ($0-100\%$).
   - **Topographic Slope Gradient ($^\circ$)**: Dynamic mountain slope incline visualizer that physically tilts as you adjust the angle ($0-50^\circ$).
   - **Seismic Vibration ($g$)**: **Real-Time Canvas Seismograph Oscilloscope** rendering live vibrating waveforms whose amplitude scales with tremor intensity ($0.00-0.60\text{ g}$).

3. **Central AI Threat Intelligence Core**:
   - Circular holographic SVG gauge displaying **Landslide Probability ($0-100\%$)** with glowing progress ring.
   - Dynamic 4-Tier Threat Status Banner:
     - 🟢 `LEVEL 0 // STABLE & NOMINAL`
     - 🟡 `LEVEL 1 // ELEVATED WATCH`
     - 🟠 `LEVEL 2 // SEVERE WARNING`
     - 🔴 `LEVEL 3 // CRITICAL EVACUATION HAZARD` (with emergency pulsating strobe)
   - Real-time **Factor of Safety ($FS$)** approximation ($FS > 1.5$ safe, $< 1.0$ failure).
   - Automated **Emergency Mitigation Protocol** providing tactical guidance.

4. **3D Topographic Terrain & Radar Scanner**:
   - Canvas-rendered isometric 3D topographic wireframe mesh with rotating radar sweep beam.
   - Interactive sensor probe nodes (Alpha-HYD, Beta-SAT, Gamma-INC, Delta-ACC) plotted on the terrain.
   - Contour wireframe dynamically shifts color and elevation in response to simulated stress.

5. **Audio & Voice Warning Synthesizer**:
   - Built-in **Web Audio API** synthesizer producing futuristic UI clicks, chimes, and warning sirens (no external audio files required).
   - **Web Speech API AI Voice Broadcast**: Vocally announces critical warnings when landslide risk exceeds threshold.

6. **Live Multi-Channel Seismogram (Chart.js)**:
   - Rolling 25-point historical telemetry stream tracking Rainfall, Soil Moisture, Seismic Shock, and Predicted Risk.

7. **Monospace Mission Console Feed**:
   - Terminal-style live log capturing every ingested telemetry packet with microsecond timestamps and status badges.

8. **Zero-Failure Hybrid Inference Engine**:
   - Connects directly to the Flask backend `/predict` endpoint.
   - Features an embedded client-side fallback matching the trained Random Forest model equations, guaranteeing that the dashboard will run smoothly even in standalone mode.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the futuristic GEO-SENTINEL AI dashboard |
| `/predict` | `POST` | Ingests `{ "rainfall": float, "soil_moisture": float, "slope_angle": float, "vibration": float }` and returns complete risk breakdown, probability, factor of safety, and recommendations |
| `/latest` | `GET` | Returns the most recent telemetry packet |
| `/update_telemetry` | `POST` | Ingestion endpoint for continuous external sensor streams or `demo_runner.py --stream` |
| `/history` | `GET` | Returns rolling 50-point telemetry history |
| `/scenarios` | `GET` | Returns list of pre-configured disaster scenarios |
| `/random_scenario` | `GET` | Generates a dynamic random scenario with model prediction |
| `/batch_predict` | `POST` | Evaluates array of sensor packets in a single call |
| `/status` | `GET` | Health check, uptime, and model telemetry status |

---

## 📦 Dependencies
- Python 3.9+
- `flask`, `flask-cors`
- `scikit-learn`, `joblib`, `scipy`
- `pandas`, `numpy`
