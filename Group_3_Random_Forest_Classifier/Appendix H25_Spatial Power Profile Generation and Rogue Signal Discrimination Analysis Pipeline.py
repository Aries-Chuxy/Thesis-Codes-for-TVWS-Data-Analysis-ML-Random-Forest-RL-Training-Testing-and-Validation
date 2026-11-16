"""
Description
This Python script constructs a synthetic multi-district validation dataset across the Enugu metropolis for Control Channel 22 to model spatial signal variations. It computes spatial mean power metrics across varying terrain classifications, performs quadratic B-spline interpolation (make_interp_spline) to visualize continuous spatial power curves, and generates publication-grade MATLAB-style plots discriminating between TVWS noise floors, standard DTT broadcasts, and localized rogue transmission anomalies against key decision thresholds (Control_Channel_22_Continuous_Plot.png).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
BASE_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer")
DATA_DIR = BASE_DIR / "Result_Sheet3" / "Result_Sheet3_Retrained"
OUTPUT_DIR = DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. GENERATE SYNTHETIC MATCHING DATASET (TARGET 1)
# ==========================================
np.random.seed(42)

districts = [
    "Coal Camp", "Iva Valley", "Independence Layout", 
    "GRA", "Ogui-New Layout", "Abakpa", "Garriki"
]

data = []
for loc_idx, loc in enumerate(districts):
    n_samples = 150
    if loc in ["Coal Camp", "Iva Valley"]:
        # Quiet TVWS Noise Floor (~ -104 dBm)
        powers = np.random.normal(loc=-104.0, scale=2.5, size=n_samples)
        classes = ["Vacant Channel (TVWS)"] * n_samples
    elif loc in ["Independence Layout", "GRA", "Ogui-New Layout"]:
        # Standard Primary DTT Broadcasts (~ -91 dBm)
        powers = np.random.normal(loc=-91.0, scale=2.0, size=n_samples)
        classes = ["Primary Broadcast DTT"] * n_samples
    else:  # Abakpa & Garriki
        # High-Power Localized Rogue Transmitter Anomaly Spikes (~ -78 dBm)
        p_anomaly = np.random.normal(loc=-78.0, scale=3.5, size=int(n_samples * 0.7))
        p_normal = np.random.normal(loc=-92.0, scale=2.0, size=int(n_samples * 0.3))
        powers = np.concatenate([p_anomaly, p_normal])
        classes = ["Unauthorized Localized Anomaly"] * len(p_anomaly) + ["Primary Broadcast DTT"] * len(p_normal)
    
    for p, c in zip(powers, classes):
        data.append({"Location_Idx": loc_idx, "Location": loc, "Mean_Power": p, "Predicted_Class": c})

df_synthetic = pd.DataFrame(data)

# Save generated dataset
csv_path = OUTPUT_DIR / "Control_Channel_22_Synthetic_Validation_Data.csv"
df_synthetic.to_csv(csv_path, index=False)

# ==========================================
# 3. COMPUTE CONTINUOUS SPATIAL PROFILES
# ==========================================
spatial_profile = df_synthetic.groupby(["Location_Idx", "Location"])["Mean_Power"].agg(["mean", "std", "min", "max"]).reset_index()

tvws_pts = spatial_profile[spatial_profile["Location_Idx"].isin([0, 1])]
dtt_pts = spatial_profile[spatial_profile["Location_Idx"].isin([2, 3, 4])]
anomaly_pts = spatial_profile[spatial_profile["Location_Idx"].isin([5, 6])]

x_dense = np.linspace(0, 6, 300)
spline = make_interp_spline(spatial_profile["Location_Idx"], spatial_profile["mean"], k=2)
power_curve_dense = spline(x_dense)

# ==========================================
# 4. MATLAB-STYLE CONTINUOUS LINE PLOT
# ==========================================
plt.style.use('classic')
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Plot continuous curved spatial mean power trace
ax.plot(x_dense, power_curve_dense, color='navy', linestyle='-', linewidth=2.0, label='Spatial Power Variation Profile P_mean(x)')

# Plot localized markers
ax.plot(tvws_pts["Location_Idx"], tvws_pts["mean"], 'o', color='#7570b3', markersize=8, label='Vacant TVWS Baseline (Quiet Zone)')
ax.plot(dtt_pts["Location_Idx"], dtt_pts["mean"], 's', color='#1b9e77', markersize=8, label='Standard Regional DTT Broadcasts')
ax.plot(anomaly_pts["Location_Idx"], anomaly_pts["mean"], '^', color='#d95f02', markersize=9, label='Rogue Localized Anomaly Spikes')

# Threshold Lines
ax.axhline(-95.0, color='red', linestyle='--', linewidth=2.0, label='Hard Occupancy Threshold (-95.0 dBm)')
ax.axhline(-88.0, color='black', linestyle=':', linewidth=1.8, label='Anomaly Trigger Threshold (-88.0 dBm)')

# Formatting plot elements
ax.set_title('Target 1: Spatial Power Profile & Rogue Signal Discrimination on Control Channel 22', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Measurement Location (Enugu Metropolis Districts)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Received Signal Power (dBm)', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(districts)))
ax.set_xticklabels(districts, rotation=25, ha='right', fontsize=9.5, fontweight='bold')
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
ax.set_ylim(-115, -60)
ax.set_xlim(-0.2, 6.2)

# Shaded Rogue Anomaly Alert Region
ax.axhspan(-88.0, -60.0, color='orange', alpha=0.12, label='Rogue Signal / Anomaly Detection Operating Region')

# Text Callout
ax.text(5.5, -67.0, 'Rogue Localized Transmission\n(Abakpa & Garriki: P > -88 dBm)', fontsize=9.5, fontweight='bold', 
        color='#d95f02', ha='center', bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#d95f02'))

# Legend positioning (Moved to Top Left Corner)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=9.0)

plt.tight_layout()

# Save image
plot_path = OUTPUT_DIR / "Control_Channel_22_MATLAB_Continuous_Plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n[✓] Target 1 MATLAB-style plot updated! Saved to:\n    {plot_path}")
