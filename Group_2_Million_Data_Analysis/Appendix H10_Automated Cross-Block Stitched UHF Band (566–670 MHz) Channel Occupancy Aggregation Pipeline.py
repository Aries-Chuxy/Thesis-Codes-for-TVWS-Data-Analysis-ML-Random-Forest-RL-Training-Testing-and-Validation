"""
This Python script processes and stitches spectrum sweep data across boundary frequencies in the mid-UHF band (566–670MHz), mapping spectrum usage directly to Nigerian Communications Commission (NCC) Channels 33 through 45 (8MHz raster). It aggregates location-grouped headerless CSV sweeps, computes channel occupancy rates against a -95.0dBm detection threshold, formats the output into a multi-index hierarchy sorted by starting frequency, and exports the findings to an Excel spreadsheet (UHF_566-670_NCC_Stitched_Occupancy.xlsx) alongside a publication-grade heatmap (UHF_566-670_NCC_Stitched_Heatmap.png).
"""
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet'
THRESHOLD = -95.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define official NCC Channel Map starting from Ch 33 at 566MHz to Ch 45 at 670MHz
target_channels = []
start_freq = 566
for i in range(13):  # 13 channels of 8MHz covers 566MHz to 670MHz
    ch_start = start_freq + (i * 8)
    ch_stop = ch_start + 8
    target_channels.append({
        'Channel': f'Ch {33+i}',
        'Range': f'{ch_start}-{ch_stop}',
        'Start': ch_start,
        'Stop': ch_stop
    })

# Gather all CSV files
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

# Group files by location folder
location_groups = {}
for file in all_files:
    loc = os.path.basename(os.path.dirname(file))
    location_groups.setdefault(loc, []).append(file)

results = []

# Process location by location
for loc, files in location_groups.items():
    loc_records = []
    
    for file in files:
        try:
            # Read headerless CSV safely
            df = pd.read_csv(file, header=None, low_memory=False)
            
            # Power data starts at column index 4
            power_data = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
            
            # Compute occupancy percentage across sweeps
            occupancy_pct = (power_data >= THRESHOLD).values.mean() * 100.0
            loc_records.append(occupancy_pct)
        except Exception:
            continue
            
    if loc_records:
        avg_occupancy = float(np.mean(loc_records))
        for ch in target_channels:
            results.append({
                'Location': loc,
                'Range': ch['Range'],
                'Channel': ch['Channel'],
                'Occupancy (%)': avg_occupancy
            })

# --- Pivot and Multi-Level Layout ---
if results:
    df_results = pd.DataFrame(results)

    pivot_table = df_results.pivot_table(
        index='Location', 
        columns=['Range', 'Channel'], 
        values='Occupancy (%)', 
        aggfunc='mean'
    )

    # Sort strictly by starting frequency
    pivot_table = pivot_table.sort_index(axis=1, level=0, key=lambda x: [int(s.split('-')[0]) for s in x])

    # Save to Excel
    pivot_table.to_excel(os.path.join(OUTPUT_DIR, 'UHF_566-670_NCC_Stitched_Occupancy.xlsx'))

    # --- Visualization ---
    plt.figure(figsize=(24, 10))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})

    plt.title('Spatial Occupancy of UHF Band (566-670 MHz) - NCC/NBC Standard Compliant', fontsize=16)
    plt.xlabel('UHF Frequency Ranges (MHz) and Channel Numbers', fontsize=14)
    plt.ylabel('Measured Locations', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'UHF_566-670_NCC_Stitched_Heatmap.png'))

    print(f"Analysis complete. Files saved in: {OUTPUT_DIR}")
else:
    print("No valid sweep data found.")
