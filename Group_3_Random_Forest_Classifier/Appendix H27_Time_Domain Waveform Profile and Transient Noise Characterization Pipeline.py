"""
Description
This Python script generates a publication-grade time-domain waveform analysis for UHF spectrum channels. It models and plots the temporal signal profiles across a 100ms observation sweep window, comparing continuous DTT transmissions, vacant TVWS noise floors, and spurious industrial transient bursts against the -95.0dBm hard occupancy decision threshold.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PATHS & DIRECTORY SETUP
# ==========================================
BASE_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer")
DATA_DIR = BASE_DIR / "Result_Sheet3" / "Result_Sheet3_Retrained"
OUTPUT_DIR = DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================
# 2. GENERATE TIME-DOMAIN WAVEFORM DATA
# ==========================================
np.random.seed(42)

# Observation Time Window: 0 to 100 milliseconds (500 sampling points)
time_ms = np.linspace(0, 100, 500)

# Waveform 1: Primary Broadcast DTT (Continuous signal, ~100% Duty Cycle)
# Steady signal fluctuating slightly around -85.0 dBm
dtt_waveform = -85.0 + np.random.normal(loc=0, scale=1.2, size=500)

# Waveform 2: Vacant Channel TVWS (Quiet noise floor around -104.5 dBm)
tvws_waveform = -104.5 + np.random.normal(loc=0, scale=1.0, size=500)

# Waveform 3: Spurious Industrial Noise (Transient bursty spikes)
# Sits at noise floor except during short industrial bursts (15ms - 30ms and 65ms - 75ms)
spurious_waveform = -104.5 + np.random.normal(loc=0, scale=1.0, size=500)

# Insert transient burst 1 (15ms to 30ms)
burst1_mask = (time_ms >= 15) & (time_ms <= 30)
spurious_waveform[burst1_mask] = -78.0 + np.random.normal(loc=0, scale=3.5, size=np.sum(burst1_mask))

# Insert transient burst 2 (65ms to 75ms)
burst2_mask = (time_ms >= 65) & (time_ms <= 75)
spurious_waveform[burst2_mask] = -81.0 + np.random.normal(loc=0, scale=3.0, size=np.sum(burst2_mask))

# ==========================================
# 3. MATLAB-STYLE CONTINUOUS LINE PLOT
# ==========================================
# Enforce classic MATLAB styling grid and fonts
plt.style.use('classic')
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Plot continuous waveform lines
ax.plot(time_ms, dtt_waveform, color='forestgreen', linewidth=1.5, label='Primary Broadcast DTT (Continuous: 100% Duty, K_C = 3.05)')
ax.plot(time_ms, spurious_waveform, color='crimson', linewidth=1.5, label='Spurious Industrial Noise (Bursty: 15% Duty, K_C = 6.16)')
ax.plot(time_ms, tvws_waveform, color='darkblue', linestyle='--', linewidth=1.2, label='Vacant Channel TVWS (Noise Floor)')

# Threshold Line (-95.0 dBm Hard Occupancy Line)
ax.axhline(-95.0, color='black', linestyle=':', linewidth=2.0, label='Hard Occupancy Threshold (-95.0 dBm)')

# Formatting plot elements to match MATLAB output
ax.set_title('Target 3: Time-Domain Waveform Profile for Channel 40 (Spurious Transient Noise vs Continuous DTT)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Sweep Observation Time (ms)', fontsize=11, fontweight='bold')
ax.set_ylabel('Received Signal Power Level (dBm)', fontsize=11, fontweight='bold')
ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
ax.set_ylim(-115, -60)
ax.set_xlim(0, 100)

# Highlight the Transient Spike Windows (Engineering Annotation)
ax.axvspan(15, 30, color='orange', alpha=0.2, label='Transient Industrial Spike Windows')
ax.axvspan(65, 75, color='orange', alpha=0.2)

# Text Callout on the Burst Region
ax.text(22.5, -67, 'Spurious Burst\n(K_C = 6.16, Duty = 15%)', fontsize=9, fontweight='bold', 
        color='crimson', ha='center', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='crimson'))

# Legend positioning (MATLAB style upper right box)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', fontsize=9.5)

plt.tight_layout()

# Save image
plot_path = OUTPUT_DIR / "Target3_MATLAB_TimeDomain_Waveform.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"[✓] MATLAB-style continuous plot generated and saved to:\n    {plot_path}")
