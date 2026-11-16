"""
Description
This Python script executes a comparative performance evaluation between traditional hard-threshold signal detection (-95.0dBm) and an optimized Sub-Threshold Voting (STV) noise suppression algorithm (10% voting window) across 50 UHF channels (470 - 870MHz). It processes headerless CSV sweep files grouped by survey location and diurnal time window, quantifies spectrum duty cycles (SDC) for both detection methods, exports the structured validation datasets into a multi-tab Excel workbook (STV_Validation_Metrics.xlsx), and renders a side-by-side comparative heatmap (STV_Mechanism_Validation_Heatmap.png).
"""
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\RFExplorer\Result_Sheet'
THRESHOLD = -95.0
VOTING_WINDOW = 0.10  # 10% STV rule

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate channels (Example for UHF IV/V 470-870MHz; easily fits full spectrum)
channel_definitions = []
uhf_start = 470
for i in range(50):
    ch_start = uhf_start + (i * 8)
    ch_stop = ch_start + 8
    channel_definitions.append({
        'Channel': f'Ch {21+i}', 'Range': f'{ch_start}-{ch_stop}',
        'Start': ch_start, 'Stop': ch_stop
    })

def get_diurnal_session(time_str):
    try:
        hour = int(time_str.split(':')[0])
        if 6 <= hour < 12: return 'Morning (6am-12pm)'
        elif 12 <= hour < 16: return 'Afternoon (12pm-4pm)'
        elif 16 <= hour < 22: return 'Evening (4pm-10pm)'
        else: return 'Night (10pm-6am)'
    except Exception: return np.nan

all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

location_groups = {}
for file in all_files:
    loc = os.path.basename(os.path.dirname(file))
    location_groups.setdefault(loc, []).append(file)

baseline_results, stv_results = [], []

# --- Processing Engine ---
for loc, files in location_groups.items():
    loc_dfs = []
    for file in files:
        try:
            header_df = pd.read_csv(file, nrows=0)
            freq_cols = [c for c in header_df.columns if c.replace('.', '', 1).isdigit()]
            meta_cols = [c for c in ['Sweep', 'Time'] if c in header_df.columns]
            if freq_cols:
                loc_dfs.append(pd.read_csv(file, usecols=meta_cols + freq_cols))
        except Exception: continue
            
    if not loc_dfs: continue
        
    try:
        combined_df = loc_dfs[0]
        for next_df in loc_dfs[1:]:
            if 'Sweep' in combined_df.columns and 'Sweep' in next_df.columns:
                combined_df = pd.merge(combined_df, next_df, on=['Sweep'], suffixes=('', '_dup'))
            else:
                combined_df = pd.concat([combined_df, next_df], axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith('_dup')]
        
        combined_df['Session'] = combined_df['Time'].apply(get_diurnal_session)
        combined_df = combined_df.dropna(subset=['Session'])
        if combined_df.empty: continue
            
        freq_cols = [c for c in combined_df.columns if c.replace('.', '', 1).isdigit()]
        freq_floats = np.array([float(c) for c in freq_cols])
        
        for ch in channel_definitions:
            band_mask = (freq_floats >= ch['Start']) & (freq_floats < ch['Stop'])
            matched_cols = [freq_cols[i] for i, matches in enumerate(band_mask) if matches]
            
            if matched_cols:
                # --- 1. BASELINE HARD THRESHOLD CRITERIA ---
                baseline_state = (combined_df[matched_cols] > THRESHOLD).any(axis=1).astype(int)
                base_sdc = baseline_state.groupby(combined_df['Session']).mean() * 100
                
                # --- 2. PROPOSED SUB-THRESHOLD VOTING CRITERIA ---
                bin_activation = (combined_df[matched_cols] > THRESHOLD).mean(axis=1)
                stv_state = (bin_activation >= VOTING_WINDOW).astype(int)
                stv_sdc = stv_state.groupby(combined_df['Session']).mean() * 100
                
                for session in base_sdc.index:
                    meta = {'Location': loc, 'Range': ch['Range'], 'Channel': ch['Channel'], 'Session': session}
                    baseline_results.append({**meta, 'SDC (%)': base_sdc[session]})
                    stv_results.append({**meta, 'SDC (%)': stv_sdc[session]})
    except Exception as e: print(f"Error at {loc}: {e}")

# --- Formatting Matrices ---
def build_pivot(raw_data):
    df = pd.DataFrame(raw_data)
    p = df.pivot_table(index=['Location', 'Session'], columns=['Range', 'Channel'], values='SDC (%)', aggfunc='mean')
    p = p.reindex(level='Session', labels=['Morning (6am-12pm)', 'Afternoon (12pm-4pm)', 'Evening (4pm-10pm)', 'Night (10pm-6am)'])
    return p.sort_index(axis=1, level=0, key=lambda x: [int(s.split('-')[0]) for s in x])

pivot_base = build_pivot(baseline_results)
pivot_stv = build_pivot(stv_results)

# Save validation data matrices to Excel sheets
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'STV_Validation_Metrics.xlsx')) as writer:
    pivot_base.to_excel(writer, sheet_name='Baseline_Hard_Threshold')
    pivot_stv.to_excel(writer, sheet_name='Proposed_STV_Engine')

# --- Side-by-Side Visual Validation Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(36, 14), sharey=True)
sns.set_theme(style="white")

# Plot 1: Baseline Hard Threshold Heatmap
sns.heatmap(pivot_base, ax=axes[0], annot=True, cmap='viridis', fmt='.1f', vmin=0, vmax=100,
            cbar=False, annot_kws={'weight': 'bold', 'size': 8})
axes[0].set_title('A) Baseline Detector Performance (Hard Threshold = -95 dBm)', fontsize=16, weight='bold', pad=15)
axes[0].set_xlabel('Channel Ranges (MHz) & Identifiers', fontsize=12, weight='bold')
axes[0].set_ylabel('Spatial Locations & Diurnal Sessions', fontsize=12, weight='bold')
axes[0].tick_params(axis='x', rotation=90, labelsize=9)

# Plot 2: Proposed Sub-Threshold Voting Heatmap
sns.heatmap(pivot_stv, ax=axes[1], annot=True, cmap='viridis', fmt='.1f', vmin=0, vmax=100,
            cbar_kws={'label': 'Spectrum Duty Cycle (%)'}, annot_kws={'weight': 'bold', 'size': 8})
axes[1].set_title('B) Proposed STV Detector Performance (Voting Window = 10%)', fontsize=16, weight='bold', pad=15)
axes[1].set_xlabel('Channel Ranges (MHz) & Identifiers', fontsize=12, weight='bold')
axes[1].set_ylabel('', visible=False) # Share Y axis
axes[1].tick_params(axis='x', rotation=90, labelsize=9)

# Font configurations
for ax in axes:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'STV_Mechanism_Validation_Heatmap.png'), dpi=300)
print(f"Validation framework complete. Visual assets generated in: {OUTPUT_DIR}")
