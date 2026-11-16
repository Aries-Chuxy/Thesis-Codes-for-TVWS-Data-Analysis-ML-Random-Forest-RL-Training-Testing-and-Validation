"""
Description
This Python script processes spectrum sweep data across the upper UHF sub-band (766–870MHz), mapping spectrum usage directly to Nigerian Communications Commission (NCC) Channels 58 through 70 (8MHz raster). It loads headerless CSV files grouped by survey location, computes power levels against a -95.0dBm threshold to determine channel occupancy rates, structures output columns sequentially by frequency range, and exports the spatial aggregation to an Excel spreadsheet (UHF_766-870_Channel_Analysis.xlsx) and a publication-quality heatmap (UHF_766-870_Horizontal_Heatmap.png).
"""
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet'
THRESHOLD = -95.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate channel data: Ch 58 (766MHz) to Ch 70 (870MHz)
channel_data = []
start_freq = 766
for i in range(13): # 13 channels of 8MHz covers 766MHz to 870MHz
    ch_start = start_freq + (i * 8)
    ch_stop = ch_start + 8
    channel_data.append({
        'Channel': f'Ch {58+i}',
        'Range': f'{ch_start}-{ch_stop}', # No 'MHz' to save space in headers
        'Start': ch_start,
        'Stop': ch_stop
    })

# Gather all processed CSV files
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

# Group files by location folder to handle cross-file stitching for Ch 58
location_groups = {}
for file in all_files:
    loc = os.path.basename(os.path.dirname(file))
    location_groups.setdefault(loc, []).append(file)

results = []

# Process location by location
for loc, files in location_groups.items():
    loc_dfs = []
    for file in files:
        try:
            temp_df = pd.read_csv(file)
            # Filter numerical frequency columns and Sweep tracker metadata
            freq_cols = [col for col in temp_df.columns if col.replace('.', '', 1).isdigit()]
            meta_cols = [col for col in ['Sweep', 'Date', 'Time'] if col in temp_df.columns]
            loc_dfs.append(temp_df[meta_cols + freq_cols])
        except Exception:
            continue
            
    if not loc_dfs:
        continue
        
    try:
        # Merge physical files to reconstruct a unified spectrum for the location
        combined_df = loc_dfs[0]
        for next_df in loc_dfs[1:]:
            combined_df = pd.merge(combined_df, next_df, on=['Sweep'], suffixes=('', '_dup'))
            
        combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith('_dup')]
        
        # Identify frequency columns
        freq_cols = [col for col in combined_df.columns if col.replace('.', '', 1).isdigit()]
        freq_floats = {col: float(col) for col in freq_cols}
        
        # Calculate occupancy rate per standard channel
        for ch in channel_data:
            matched_cols = [col for col, f_val in freq_floats.items() if ch['Start'] <= f_val < ch['Stop']]
            
            if matched_cols:
                occupancy_pct = (combined_df[matched_cols] > THRESHOLD).any(axis=1).mean() * 100
                results.append({
                    'Location': loc,
                    'Range': ch['Range'],
                    'Channel': ch['Channel'],
                    'Occupancy (%)': occupancy_pct
                })
    except Exception as e:
        print(f"Error compiling data for location {loc}: {e}")

# --- Pivot and Multi-Level Layout ---
df_results = pd.DataFrame(results)

pivot_table = df_results.pivot_table(
    index='Location', 
    columns=['Range', 'Channel'], 
    values='Occupancy (%)', 
    aggfunc='mean'
)

# Sort columns strictly by starting frequency sequence (766, 774, 782...)
pivot_table = pivot_table.sort_index(axis=1, level=0, key=lambda x: [int(s.split('-')[0]) for s in x])

# Save output data to hard disk
pivot_table.to_excel(os.path.join(OUTPUT_DIR, 'UHF_766-870_Channel_Analysis.xlsx'))

# --- Visualization ---
plt.figure(figsize=(24, 10))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})

# Final labeling adjustments
plt.title('Spatial Occupancy of UHF Band (766-870 MHz) - NCC/NBC Standard Compliant', fontsize=16)
plt.xlabel('UHF Frequency Ranges (MHz) and Channel Numbers', fontsize=14)
plt.ylabel('Measured Locations', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'UHF_766-870_Horizontal_Heatmap.png'))

print(f"Analysis complete. Boundary-safe files saved in: {OUTPUT_DIR}")
