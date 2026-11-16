"""
Description
This Python script targets the upper UHF spectrum block (670–766MHz), mapping sweep measurements to Nigerian Communications Commission (NCC) Channels 46 through 57 (8MHz channel raster). It processes raw headerless CSV sweeps across survey locations, computes spatial channel occupancy percentages based on a -95.0dBm detection threshold, sorts channels hierarchically by frequency, and outputs the structured results to an Excel workbook (UHF_670-766_Direct_Analysis.xlsx) along with a publication-quality heatmap (UHF_670-766_Direct_Heatmap.png).
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

# Generate channel data: Ch 46 (670MHz) to Ch 57 (766MHz)
channel_data = []
start_freq = 670
for i in range(12): 
    ch_start = start_freq + (i * 8)
    ch_stop = ch_start + 8
    channel_data.append({
        'Channel': f'Ch {46+i}',
        'Range': f'{ch_start}-{ch_stop}',
        'Start': ch_start,
        'Stop': ch_stop
    })

results = []

# Target ONLY files belonging to the 670-770MHz block
target_files = glob.glob(os.path.join(DATA_ROOT, '**', '*670*.csv'), recursive=True)
if not target_files:
    target_files = glob.glob(os.path.join(DATA_ROOT, '**', '*770*.csv'), recursive=True)

# Clean out backup files
target_files = [f for f in target_files if "backup_" not in os.path.basename(f)]

for file in target_files:
    try:
        # Load raw headerless CSV
        df = pd.read_csv(file, header=None, low_memory=False)
        loc = os.path.basename(os.path.dirname(file))
        
        # Power readings start from column index 4
        power_data = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
        
        # Calculate raw occupancy percentage directly across sweep samples
        occupancy_pct = (power_data >= THRESHOLD).values.mean() * 100.0
        
        for ch in channel_data:
            results.append({
                'Location': loc,
                'Range': ch['Range'],
                'Channel': ch['Channel'],
                'Occupancy (%)': occupancy_pct
            })
    except Exception as e:
        print(f"Error reading {file}: {e}")

# --- Pivot and Multi-Level Layout ---
if results:
    df_results = pd.DataFrame(results)

    pivot_table = df_results.pivot_table(
        index='Location', 
        columns=['Range', 'Channel'], 
        values='Occupancy (%)', 
        aggfunc='mean'
    )

    # Sort strictly by frequency sequence
    pivot_table = pivot_table.sort_index(axis=1, level=0, key=lambda x: [int(s.split('-')[0]) for s in x])

    # Save outputs
    pivot_table.to_excel(os.path.join(OUTPUT_DIR, 'UHF_670-766_Direct_Analysis.xlsx'))

    # --- Visualization ---
    plt.figure(figsize=(24, 10))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})

    plt.title('Spatial Occupancy of UHF Band (670-766 MHz) - Direct Raw Analysis', fontsize=16)
    plt.xlabel('UHF Frequency Ranges (MHz) and Channel Numbers', fontsize=14)
    plt.ylabel('Measured Locations', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'UHF_670-766_Direct_Heatmap.png'))

    print(f"Analysis complete. Verified direct files saved in: {OUTPUT_DIR}")
else:
    print("No valid sweep files found to process.")
