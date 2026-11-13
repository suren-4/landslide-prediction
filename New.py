"""
Synthetic Landslide Dataset Generator
Simulates geotechnical sensor measurements (Rainfall, Soil Moisture, Slope Angle, Vibration)
and landslide occurrence based on geotechnical slope stability mechanics (Mohr-Coulomb failure criteria).
"""
import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=3000, seed=42):
    np.random.seed(seed)

    # Core geotechnical variables
    # 1. Rainfall: 0 to 220 mm (hourly / cumulative storm index)
    rainfall = np.random.uniform(0, 220, n_samples)
    
    # 2. Soil Moisture: 0 to 1.0 (volumetric water content ratio, 0% - 100%)
    soil_moisture = np.random.uniform(0, 1.0, n_samples)
    
    # 3. Slope Angle: 0 to 50 degrees (mountain terrain gradient)
    slope_angle = np.random.uniform(0, 50, n_samples)
    
    # 4. Vibration: 0 to 0.60 g (peak ground acceleration / seismic intensity)
    vibration = np.random.uniform(0, 0.60, n_samples)

    # Geotechnical failure physics simulation:
    # Shear stress increases with slope incline and soil weight (increased by water).
    # Shear strength decreases with pore-water pressure (soil moisture) and dynamic shaking (vibration).
    norm_rain = (rainfall / 120.0) ** 1.5
    norm_soil = (soil_moisture / 0.65) ** 2.0
    norm_slope = (slope_angle / 28.0) ** 1.8
    norm_vib = (vibration / 0.18) ** 1.3
    
    # Add realistic environmental noise / rock cohesion variability
    environmental_noise = np.random.normal(0, 0.08, n_samples)
    
    # Composite geotechnical stress index
    risk_index = (
        0.32 * norm_rain +
        0.30 * norm_soil +
        0.26 * norm_slope +
        0.12 * norm_vib +
        environmental_noise
    )

    # Logistic failure probability curve
    # Failure threshold centered around risk_index = 1.15
    probability = 1.0 / (1.0 + np.exp(-4.5 * (risk_index - 1.15)))
    
    # Binary label: 1 = Landslide occurred, 0 = Stable slope
    landslide_occurred = (probability >= 0.50).astype(int)

    # Ensure flat terrain (< 8 deg) never slides regardless of rain
    flat_mask = (slope_angle < 8.0)
    landslide_occurred[flat_mask] = 0

    # Ensure completely dry terrain (< 15% moisture and < 15mm rain) rarely slides unless extreme seismic quake (> 0.45g)
    dry_mask = (soil_moisture < 0.15) & (rainfall < 15.0) & (vibration < 0.45)
    landslide_occurred[dry_mask] = 0

    data = pd.DataFrame({
        "rainfall": np.round(rainfall, 2),
        "soil_moisture": np.round(soil_moisture, 3),
        "slope_angle": np.round(slope_angle, 2),
        "vibration": np.round(vibration, 4),
        "landslide_occurred": landslide_occurred
    })

    return data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "synthetic_landslide_data.csv")
    
    df = generate_synthetic_data(n_samples=3500)
    df.to_csv(output_path, index=False)
    
    pos_rate = df["landslide_occurred"].mean() * 100
    print(f"Synthetic dataset saved to: {output_path}")
    print(f"Total samples: {len(df)} | Landslide events: {df['landslide_occurred'].sum()} ({pos_rate:.1f}%)")
