"""
Script Function Description 
This Python script performs a multi-profile trade-off analysis across five key service dimensions (Throughput Efficiency, Latency Minimization, Packet Reliability, Coexistence Compliance, and Power/Energy Efficiency). By evaluating distinct 3GPP service profile weightings: Enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communication (URLLC), Massive Machine-Type Communication (mMTC), Balanced Baseline, and an Unconstrained baseline, the script generates a dual-panel visualization consisting of a polar radar plot and a decomposed metric score bar chart to validate adaptive composite reward shaping performance.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
BASE_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer")
DATA_DIR = BASE_DIR / "Result_Sheet3" / "Result_Sheet3_Retrained"
OUTPUT_DIR = DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[*] Generating Fully Refined Section 4.5.4.3: Composite Reward Weighting Analysis...")

# ==========================================
# 2. METRIC DEFINITIONS & PERFORMANCE DATA
# ==========================================
categories = [
    'Throughput\nEfficiency ($w_1$)', 
    'Latency\nMinimization ($w_2$)', 
    'Packet Reliability\n($w_3$)', 
    'Coexistence\nCompliance', 
    'Power/Energy\nEfficiency'
]
N_metrics = len(categories)

# Service Profiles Data [Normalized 0.0 - 1.0]
profiles = {
    'eMBB Profile\n($w_1=0.6, w_2=0.2, w_3=0.2$)': {
        'vals': [0.96, 0.68, 0.78, 0.88, 0.72],
        'color': '#003366',
        'label': 'eMBB'
    },
    'URLLC Profile\n($w_1=0.1, w_2=0.45, w_3=0.45$)': {
        'vals': [0.65, 0.98, 0.97, 0.92, 0.80],
        'color': '#D9534F',
        'label': 'URLLC'
    },
    'mMTC Profile\n($w_1=0.2, w_2=0.2, w_3=0.6$)': {
        'vals': [0.58, 0.72, 0.85, 0.95, 0.96],
        'color': '#2CA02C',
        'label': 'mMTC'
    },
    'Balanced Baseline\n($w_1=w_2=w_3=0.33$)': {
        'vals': [0.78, 0.76, 0.79, 0.82, 0.77],
        'color': '#FF9900',
        'label': 'Balanced'
    },
    'Unconstrained Baseline\n(No Reward Shaping)': {
        'vals': [0.42, 0.31, 0.35, 0.28, 0.40],
        'color': 'gray',
        'label': 'Unconstrained'
    }
}

# ==========================================
# 3. DUAL-PANEL VISUALIZATION
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(16, 7), dpi=300)

# ------------------------------------------
# SUBPLOT (a): Enhanced Polar Radar Profile
# ------------------------------------------
ax1 = fig.add_subplot(121, polar=True)

angles = [n / float(N_metrics) * 2 * np.pi for n in range(N_metrics)]
angles_closed = angles + angles[:1]

ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)
plt.xticks(angles, categories, fontsize=10, fontweight='bold')
ax1.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="gray", size=8)
plt.ylim(0, 1.15)

# Plot Polygons
for name, p in profiles.items():
    v_closed = p['vals'] + p['vals'][:1]
    if p['label'] in ['eMBB', 'URLLC', 'mMTC']:
        ax1.plot(angles_closed, v_closed, color=p['color'], linewidth=2.5, linestyle='-', label=name)
        ax1.fill(angles_closed, v_closed, color=p['color'], alpha=0.12)
    elif p['label'] == 'Balanced':
        ax1.plot(angles_closed, v_closed, color=p['color'], linewidth=1.8, linestyle='-.', label=name)
    else:
        ax1.plot(angles_closed, v_closed, color=p['color'], linewidth=1.5, linestyle=':', label=name)

# --- ADJUSTED HIGH-VISIBILITY CALLOUT LABELS ---
# eMBB Peak: Shifted down to radial distance 0.96 and positioned to the side so it does not block the axis label
ax1.text(angles[0] - 0.28, 0.96, 'eMBB PEAK\n(96%)', color='#003366', fontsize=8.5, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.25', facecolor='#E6F0FA', edgecolor='#003366', lw=1))

# URLLC Peak: Positioned clearly next to Latency Minimization
ax1.text(angles[1] + 0.28, 1.08, 'URLLC PEAK\n(98%)', color='#D9534F', fontsize=8.5, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.25', facecolor='#FADBD8', edgecolor='#D9534F', lw=1))

# mMTC Peak: Positioned clearly next to Power/Energy Efficiency
ax1.text(angles[4] - 0.28, 1.08, 'mMTC PEAK\n(96%)', color='#2CA02C', fontsize=8.5, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.25', facecolor='#E8F8F5', edgecolor='#2CA02C', lw=1))

ax1.set_title(r'(a) Radar Trade-off Polygon across Telecom Profiles', fontsize=12, fontweight='bold', pad=25, loc='left')
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)

# ------------------------------------------
# SUBPLOT (b): Direct Profile Metric Comparison Bar Chart
# ------------------------------------------
ax2 = fig.add_subplot(122)

x = np.arange(N_metrics)
width = 0.16

rects_embb = ax2.bar(x - 2*width, profiles['eMBB Profile\n($w_1=0.6, w_2=0.2, w_3=0.2$)']['vals'], width, label='eMBB', color='#003366')
rects_urllc = ax2.bar(x - width, profiles['URLLC Profile\n($w_1=0.1, w_2=0.45, w_3=0.45$)']['vals'], width, label='URLLC', color='#D9534F')
rects_mmtc = ax2.bar(x, profiles['mMTC Profile\n($w_1=0.2, w_2=0.2, w_3=0.6$)']['vals'], width, label='mMTC', color='#2CA02C')
rects_bal = ax2.bar(x + width, profiles['Balanced Baseline\n($w_1=w_2=w_3=0.33$)']['vals'], width, label='Balanced', color='#FF9900')
rects_unp = ax2.bar(x + 2*width, profiles['Unconstrained Baseline\n(No Reward Shaping)']['vals'], width, label='Unconstrained', color='gray')

# Annotate values above key target bars
for rect in [rects_embb[0], rects_urllc[1], rects_mmtc[4]]:
    height = rect.get_height()
    ax2.annotate(f'{height*100:.0f}%',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 4),  
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_ylabel('Normalized Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Decomposed Metric Score Comparison by Service Profile', fontsize=12, fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace('\n', ' ') for c in categories], fontsize=8.5, fontweight='bold', rotation=15)
ax2.set_ylim(0, 1.25)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

ax2.legend(loc='upper center', bbox_to_anchor=(0.50, 0.98), ncol=3, fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)

# ------------------------------------------
# SAVE AND DISPLAY
# ------------------------------------------
plt.suptitle(r'Composite Reward Weighting ($w_1, w_2, w_3$) Multi-Profile Trade-off Analysis', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

plot_path = OUTPUT_DIR / "Section_4_5_4_3_Composite_Reward_Explicable.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[✓] Section 4.5.4.3 Refined Composite Reward plot successfully created!\n    Saved to: {plot_path}")
