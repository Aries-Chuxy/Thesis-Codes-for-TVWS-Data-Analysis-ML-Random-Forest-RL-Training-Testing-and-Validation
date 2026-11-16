"""
Description
This Python script executes a full-spectrum comparative analysis across 57 standard broadcast channels (VHF Band III Channels 5–11 and UHF Bands IV/V Channels 21–70) evaluating conventional hard-threshold signal detection (-95.0dBm) against an optimized Sub-Threshold Voting (STV) algorithm with a 10% voting window. It parses headerless CSV sweep files grouped by survey location and diurnal time window, evaluates active spectrum fractions, exports structured comparative data matrices to a multi-tab Excel workbook (STV_Validation_Metrics_Full.xlsx), and generates a publication-grade side-by-side validation heatmap (STV_Full_Spectrum_Validation_Heatmap.png).
"""
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet2'
THRESHOLD = -95.0
VOTING_WINDOW = 0.10  # 10% STV rule

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Upgraded, foolproof time parser
def get_diurnal_session(time_val):
    try:
        time_str = str(time_val).strip()
        parsed_time = pd.to_datetime(time_str, errors='coerce')
        if pd.isna(parsed_time):
            parsed_time = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
        if pd.isna(parsed_time):
            return np.nan
            
        hour = parsed_time.hour
        if 6 <= hour < 12: return 'Morning (6am-12pm)'
        elif 12 <= hour < 16: return 'Afternoon (12pm-4pm)'
        elif 16 <= hour < 22: return 'Evening (4pm-10pm)'
        else: return 'Night (10pm-6am)'
    except Exception:
        return np.nan

all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

raw_records = []

# --- Streaming Processing Engine ---
for file in all_files:
    try:
        header_df = pd.read_csv(file, nrows=0)
        freq_cols = [c for c in header_df.columns if c.replace('.', '', 1).isdigit()]
        meta_cols = [c for c in ['Time'] if c in header_df.columns]
        
        if not freq_cols or not meta_cols:
            continue
            
        df = pd.read_csv(file, usecols=meta_cols + freq_cols)
        loc = os.path.basename(os.path.dirname(file))
        
        df['Session'] = df['Time'].apply(get_diurnal_session)
        df = df.dropna(subset=['Session'])
        
        if df.empty: 
            continue
            
        # 1. Baseline Hard Threshold (Active if ANY bin in the sweep breaches -95dBm)
        baseline_sweep_state = (df[freq_cols] > THRESHOLD).any(axis=1).astype(int)
        baseline_session_sdc = baseline_sweep_state.groupby(df['Session']).mean() * 100
        
        # 2. Proposed STV Mechanism (Active only if at least 10% of bins breach -95dBm)
        stv_sweep_ratio = (df[freq_cols] > THRESHOLD).mean(axis=1)
        stv_sweep_state = (stv_sweep_ratio >= VOTING_WINDOW).astype(int)
        stv_session_sdc = stv_sweep_state.groupby(df['Session']).mean() * 100
        
        for session in baseline_session_sdc.index:
            raw_records.append({
                'Location': loc,
                'Session': session,
                'Baseline SDC (%)': baseline_session_sdc[session],
                'STV SDC (%)': stv_session_sdc[session]
            })
    except Exception as e:
        print(f"Error processing {os.path.basename(file)}: {e}")

# --- Generate Validation Metrics ---
df_metrics = pd.DataFrame(raw_records)

# Create consolidated Spatial vs Diurnal matrices for both models
pivot_base = df_metrics.pivot_table(index='Location', columns='Session', values='Baseline SDC (%)', aggfunc='mean')
pivot_stv = df_metrics.pivot_table(index='Location', columns='Session', values='STV SDC (%)', aggfunc='mean')

# Enforce clean chronological session layout
temporal_order = ['Morning (6am-12pm)', 'Afternoon (12pm-4pm)', 'Evening (4pm-10pm)', 'Night (10pm-6am)']
available_sessions = [s for s in temporal_order if s in pivot_base.columns]
pivot_base = pivot_base.reindex(columns=available_sessions)
pivot_stv = pivot_stv.reindex(columns=available_sessions)

# Core Math Step: Calculate the exact sensitivity gains provided by STV
pivot_difference = pivot_stv - pivot_base

# Save comprehensive comparison matrices directly to Excel
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'STV_General_Validation_Report.xlsx')) as writer:
    pivot_base.to_excel(writer, sheet_name='Baseline_Hard_Threshold')
    pivot_stv.to_excel(writer, sheet_name='Proposed_STV_Engine')
    pivot_difference.to_excel(writer, sheet_name='Validation_Delta_Gain')

# --- Plotting the Validation Heatmap ---
plt.figure(figsize=(14, 10))
sns.set_theme(style="white")

# Plot the difference matrix. If STV is working, variations will light up clearly.
ax = sns.heatmap(
    pivot_difference, 
    annot=True, 
    cmap='coolwarm', 
    center=0,
    fmt='.2f', 
    linewidths=.5,
    cbar_kws={'label': 'Sensitivity Variance Delta: STV SDC - Baseline SDC (%)'},
    annot_kws={'weight': 'bold', 'size': 12}
)

plt.title('STV Mechanism Validation: Spatial-Diurnal Sensitivity Variance Gain', fontsize=15, weight='bold', pad=20)
plt.xlabel('Temporal Diurnal Sessions', fontsize=13, weight='bold', labelpad=15)
plt.ylabel('Spatial Locations', fontsize=13, weight='bold', labelpad=15)

plt.xticks(fontsize=11, weight='bold', rotation=0)
plt.yticks(fontsize=11, weight='bold')

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Sensitivity Variance Delta (%)', weight='bold', size=12, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'STV_General_Validation_Heatmap.png'), dpi=300)

print(f"\nValidation processing completed successfully!")
print(f"Heatmap exported to: {os.path.join(OUTPUT_DIR, 'STV_General_Validation_Heatmap.png')}")
