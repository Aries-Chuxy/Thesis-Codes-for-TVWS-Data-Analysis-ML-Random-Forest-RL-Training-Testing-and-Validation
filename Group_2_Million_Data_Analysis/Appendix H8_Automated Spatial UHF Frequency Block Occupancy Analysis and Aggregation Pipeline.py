"""
Description
This Python script processes sweep data across four core UHF spectrum sub-bands (470–570 MHz, 570–670 MHz, 670–770 MHz, and 770–870MHz) collected across multiple survey locations. It filters sweep files, computes spatial channel occupancy percentages based on a -95.0dBm detection threshold, aggregates the results into a location-versus-sub-band pivot table saved as an Excel spreadsheet (UHF_Spatial_Occupancy.xlsx), and visualizes spatial distribution patterns via a Seaborn heatmap (UHF_Spatial_Heatmap.png).
"""
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet'
THRESHOLD = -95.0

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UHF Frequency blocks
uhf_blocks = {
    '470-570MHz': (470, 570),
    '570-670MHz': (570, 670),
    '670-770MHz': (670, 770),
    '770-870MHz': (770, 870)
}

results = []

# Process all CSV files within the processed data directory
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)

for file in all_files:
    # Skip backup files created by previous processing scripts
    if "backup_" in os.path.basename(file):
        continue
        
    try:
        # Load raw file without assuming header names
        df = pd.read_csv(file, header=None, low_memory=False)
        loc = os.path.basename(os.path.dirname(file))
        
        # Power readings start from column index 4 onwards
        power_data = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
        
        # Determine frequency sub-band from filename or column count
        filename = os.path.basename(file).lower()
        
        for block_name, (start, stop) in uhf_blocks.items():
            # Match file to its respective UHF block based on filename
            if f"{start}" in filename or block_name.lower() in filename:
                # Calculate occupancy percentage (samples exceeding -95 dBm)
                occupancy_pct = (power_data >= THRESHOLD).values.mean() * 100.0
                results.append({'Location': loc, 'UHF_Block': block_name, 'Occupancy (%)': occupancy_pct})
                break

    except Exception as e:
        print(f"Error processing {file}: {e}")

# --- Data Aggregation ---
if results:
    df_results = pd.DataFrame(results)
    pivot_table = df_results.pivot_table(index='Location', columns='UHF_Block', values='Occupancy (%)', aggfunc='mean')

    # --- Save Outputs ---
    # 1. Save to Excel
    excel_path = os.path.join(OUTPUT_DIR, 'UHF_Spatial_Occupancy.xlsx')
    pivot_table.to_excel(excel_path)

    # 2. Generate and Save Heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})
    plt.title('Spatial Variation of UHF Occupancy by Frequency Block')
    plt.xlabel('UHF Frequency Blocks')
    plt.ylabel('Measured Locations')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'UHF_Spatial_Heatmap.png'))

    print(f"Analysis complete. Files saved to: {OUTPUT_DIR}")
else:
    print("No matching UHF sweep data found to process.")
