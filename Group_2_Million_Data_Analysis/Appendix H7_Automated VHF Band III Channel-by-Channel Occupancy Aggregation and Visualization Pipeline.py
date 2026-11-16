"""
Description
This Python script processes RF spectrum sweep data targeting VHF Band III (174–230MHz) across multiple survey locations. It maps sweep readings to discrete 8 MHz TV channels (Channels 6 through 12), measures channel occupancy rates against a -95.0dBm detection threshold, and aggregates the results into a spatial summary table saved as an Excel spreadsheet (VHF_Channel_Occupancy.xlsx) and a publication-quality heatmap (VHF_Channel_Heatmap.png).
"""
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# Your specific data and output paths
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet'

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

THRESHOLD = -95.0

# Define channels and their frequency ranges
channel_map = {
    'Ch 6 (174-182MHz)': (174, 182),
    'Ch 7 (182-190MHz)': (182, 190),
    'Ch 8 (190-198MHz)': (190, 198),
    'Ch 9 (198-206MHz)': (198, 206),
    'Ch 10 (206-214MHz)': (206, 214),
    'Ch 11 (214-222MHz)': (214, 222),
    'Ch 12 (222-230MHz)': (222, 230)
}

results = []

# Search for all CSV files in the data root
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*174*.csv'), recursive=True)

if not all_files:
    print(f"No files found. Please check your path: {DATA_ROOT}")
else:
    for file in all_files:
        try:
            df = pd.read_csv(file)
            loc = os.path.basename(os.path.dirname(file))
            
            for ch_name, (start, stop) in channel_map.items():
                # Filter columns that fall within the specific frequency range
                freq_cols = [col for col in df.columns[4:] if start <= float(col) < stop]
                if freq_cols:
                    # Calculate occupancy percentage
                    occupancy_pct = (df[freq_cols] > THRESHOLD).any(axis=1).mean() * 100
                    results.append({'Location': loc, 'Channel': ch_name, 'Occupancy (%)': occupancy_pct})
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # --- Pivot and Save ---
    df_results = pd.DataFrame(results)
    pivot_table = df_results.pivot_table(index='Location', columns='Channel', values='Occupancy (%)', aggfunc='mean')
    
    # Reorder columns to ensure 6-12 flow horizontally
    pivot_table = pivot_table[list(channel_map.keys())]

    # Save to Excel
    excel_path = os.path.join(OUTPUT_DIR, 'VHF_Channel_Occupancy.xlsx')
    pivot_table.to_excel(excel_path)

    # Generate and save Heatmap
    plt.figure(figsize=(18, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Occupancy Rate (%)'})
    plt.title('Spatial Occupancy of VHF Band III Channels (174-230 MHz)')
    plt.ylabel('Measured Locations')
    plt.xlabel('VHF Channels (Frequency Ranges)')
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'VHF_Channel_Heatmap.png')
    plt.savefig(heatmap_path)
    
    print(f"Success! Files saved in: {OUTPUT_DIR}")
