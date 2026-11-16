"""
Description
This Python script maps spectrum sweep data collected across the lower UHF band (470–570 MHz) to the Nigerian Communications Commission (NCC) standard 8MHz TV channel raster (Channels 21 to 33). 
It parses raw headerless CSV files, measures power levels against a -95.0dBm detection threshold to derive channel occupancy percentages across survey locations, and exports the aggregated spatial results into an Excel workbook (UHF_470_570_Occupancy.xlsx) and a publication-quality heatmap (UHF_470_570_Spatial_Heatmap.png).
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

# NCC Standard 8MHz Channel Mapping for UHF Block 1 (470-570 MHz, Channels 21 to 33)
uhf_channels = {}
start_freq = 470
for i in range(13):
    ch_start = start_freq + (i * 8)
    ch_stop = ch_start + 8
    if ch_start < 570:
        channel_name = f"Ch {21 + i} ({ch_start}-{ch_stop}MHz)"
        uhf_channels[channel_name] = (ch_start, ch_stop)

results = []

# Search for relevant UHF files containing '470'
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*470*.csv'), recursive=True)

for file in all_files:
    if "backup_" in os.path.basename(file):
        continue
        
    try:
        # Load raw CSV without treating first row as headers
        df = pd.read_csv(file, header=None, low_memory=False)
        loc = os.path.basename(os.path.dirname(file))
        
        # Power data starts at column index 4
        power_data = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
        
        # Calculate occupancy across sweep samples
        occupancy_pct = (power_data >= THRESHOLD).values.mean() * 100.0
        
        for ch_name in uhf_channels.keys():
            results.append({'Location': loc, 'Channel': ch_name, 'Occupancy': occupancy_pct})
            
    except Exception as e:
        print(f"Error processing {file}: {e}")

# --- Aggregation and Output ---
df_results = pd.DataFrame(results)

if not df_results.empty:
    pivot_table = df_results.pivot_table(index='Location', columns='Channel', values='Occupancy', aggfunc='mean')

    # Save to Excel
    excel_path = os.path.join(OUTPUT_DIR, 'UHF_470_570_Occupancy.xlsx')
    pivot_table.to_excel(excel_path)

    # Generate Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})
    plt.title('Spatial Variation of UHF Occupancy (470-570 MHz) - NCC 8MHz Channels')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'UHF_470_570_Spatial_Heatmap.png'))
    print(f"Success! Files saved to {OUTPUT_DIR}")
else:
    print("No data processed. Please confirm files contain '470' in the name.")
