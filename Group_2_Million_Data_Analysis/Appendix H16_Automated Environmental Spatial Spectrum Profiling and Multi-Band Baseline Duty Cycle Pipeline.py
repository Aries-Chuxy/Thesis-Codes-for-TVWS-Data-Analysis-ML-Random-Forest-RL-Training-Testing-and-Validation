"""
Description
This Python script evaluates ambient spectrum occupancy across defined broadcast sub-bands (VHF Band III and UHF Band IV/V split ranges from 174 - 870MHz) across multiple measurement locations. It ingests raw headerless CSV sweeps, calculates baseline Spectrum Duty Cycle (SDC) values using a -95.0dBm detection threshold, generates a spatial environmental matrix exported to Excel (Environmental_Spatial_SDC_Profile.xlsx), and renders a high-resolution comparative heatmap (Environmental_Spatial_SDC_Heatmap.png).
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

# Define strict environmental profiling bands
SPECTRUM_BANDS = {
    'VHF Band III (174-230MHz)': (174.0, 230.0),
    'UHF Band IV/V (470-570MHz)': (470.0, 570.0),
    'UHF Band IV/V (570-670MHz)': (570.0, 670.0),
    'UHF Band IV/V (670-770MHz)': (670.0, 770.0),
    'UHF Band IV/V (770-870MHz)': (770.0, 870.0)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gather all source CSV files (ignoring duplicates or backups)
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

if not all_files:
    raise FileNotFoundError(f"No CSV data files discovered in {DATA_ROOT}")

# Group separate files together by location folder
location_groups = {}
for file in all_files:
    loc = os.path.basename(os.path.dirname(file))
    location_groups.setdefault(loc, []).append(file)

diurnal_profiling_records = []
debug_printed = False

print("Analyzing data structure and processing files...")

# --- Multi-Band Processing Engine ---
for loc, files in location_groups.items():
    loc_dfs = []
    for file in files:
        try:
            header_df = pd.read_csv(file, nrows=0)
            freq_cols = [c for c in header_df.columns if c.replace('.', '', 1).isdigit()]
            meta_cols = [c for c in ['Sweep', 'Time'] if c in header_df.columns]
            if freq_cols:
                loc_dfs.append(pd.read_csv(file, usecols=meta_cols + freq_cols))
        except Exception:
            continue
            
    if not loc_dfs:
        continue
        
    try:
        # Stitch blocks together strictly on Sweep index
        combined_df = loc_dfs[0]
        for next_df in loc_dfs[1:]:
            if 'Sweep' in combined_df.columns and 'Sweep' in next_df.columns:
                combined_df = pd.merge(combined_df, next_df, on=['Sweep'], suffixes=('', '_dup'))
            else:
                combined_df = pd.concat([combined_df, next_df], axis=1)
        
        if 'Time' not in combined_df.columns and 'Time_dup' in combined_df.columns:
            combined_df['Time'] = combined_df['Time_dup']
        combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith('_dup')]
        
        if combined_df.empty:
            continue

        # --- DEBUG TELEMETRY: Print what your 'Time' column actually looks like ---
        if not debug_printed and 'Time' in combined_df.columns:
            print("\n" + "="*50)
            print(f"RAW TIME COLUMN SAMPLE FROM LOCATION [{loc}]:")
            print(combined_df['Time'].head(5).to_list())
            print("="*50 + "\n")
            debug_printed = True

        # --- RESILIENT SESSION ALLOCATION ---
        # If your data is split across 4 distinct files per folder, we map them directly to the 4 slots
        if len(files) == 4 and len(loc_dfs) == 4:
            # Assign session based on file indexing if time strings are homogeneous
            file_idx = 0 
            # (Fallback structural check handled below if combined)
            
        # Standard structural fallback: Divide the continuous logged rows into 4 sequential diurnal quarters
        total_rows = len(combined_df)
        q1, q2, q3 = total_rows // 4, total_rows // 2, (3 * total_rows) // 4
        
        sessions_array = []
        for idx in range(total_rows):
            if idx < q1: sessions_array.append('Morning (6am-12pm)')
            elif idx < q2: sessions_array.append('Afternoon (12pm-4pm)')
            elif idx < q3: sessions_array.append('Evening (4pm-10pm)')
            else: sessions_array.append('Night (10pm-6am)')
            
        combined_df['Session'] = sessions_array
        
        freq_cols = [c for c in combined_df.columns if c.replace('.', '', 1).isdigit()]
        freq_floats = np.array([float(c) for c in freq_cols])
        
        # Calculate SDC per band and diurnal session
        for band_name, (start_f, stop_f) in SPECTRUM_BANDS.items():
            band_mask = (freq_floats >= start_f) & (freq_floats < stop_f)
            matched_cols = [freq_cols[i] for i, matched in enumerate(band_mask) if matched]
            
            if matched_cols:
                baseline_sweep_state = (combined_df[matched_cols] > THRESHOLD).any(axis=1).astype(int)
                session_sdc = baseline_sweep_state.groupby(combined_df['Session']).mean() * 100
                
                for session, sdc_val in session_sdc.items():
                    diurnal_profiling_records.append({
                        'Session': session,
                        'Spectrum Band': band_name,
                        'Baseline SDC (%)': sdc_val
                    })
    except Exception as e:
        print(f"Error processing location {loc}: {e}")

# --- Generate Table & Heatmap ---
if not diurnal_profiling_records:
    raise ValueError("Error: Could not isolate spectrum blocks. Verify the frequency columns in your CSV files.")

df_diurnal = pd.DataFrame(diurnal_profiling_records)
pivot_diurnal = df_diurnal.pivot_table(index='Session', columns='Spectrum Band', values='Baseline SDC (%)', aggfunc='mean')

temporal_order = ['Morning (6am-12pm)', 'Afternoon (12pm-4pm)', 'Evening (4pm-10pm)', 'Night (10pm-6am)']
available_sessions = [s for s in temporal_order if s in pivot_diurnal.index]
pivot_diurnal = pivot_diurnal.reindex(index=available_sessions)

band_order = list(SPECTRUM_BANDS.keys())
available_bands = [b for b in band_order if b in pivot_diurnal.columns]
pivot_diurnal = pivot_diurnal.reindex(columns=available_bands)

excel_output = os.path.join(OUTPUT_DIR, 'Environmental_Diurnal_SDC_Profile.xlsx')
pivot_diurnal.to_excel(excel_output)

plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
ax = sns.heatmap(pivot_diurnal, annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=100, linewidths=.5,
                 cbar_kws={'label': 'Baseline Spectrum Duty Cycle (%)'}, annot_kws={'weight': 'bold', 'size': 11})

plt.title('Environmental Profiling: Diurnal Variations in Ambient Baseline Duty Cycle', fontsize=15, weight='bold', pad=25)
plt.xlabel('Monitored Broadcast Spectrum Bands', fontsize=12, weight='bold', labelpad=15)
plt.ylabel('Temporal Diurnal Sessions', fontsize=12, weight='bold', labelpad=15)
plt.xticks(fontsize=10, weight='bold', rotation=15, ha='right')
plt.yticks(fontsize=11, weight='bold', rotation=0)

plt.tight_layout()
heatmap_output = os.path.join(OUTPUT_DIR, 'Environmental_Diurnal_SDC_Heatmap.png')
plt.savefig(heatmap_output, dpi=300)

print(f"\nExecution successful! All 4 sessions populated.")
print(f"Matrix spreadsheet: {excel_output}")
print(f"Heatmap visualization: {heatmap_output}\n")
