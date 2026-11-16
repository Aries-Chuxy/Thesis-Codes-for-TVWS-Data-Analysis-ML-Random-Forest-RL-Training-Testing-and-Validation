"""
Description
This Python script streams and aggregates full-spectrum sweep data across VHF and UHF bands to analyze general spectrum occupancy variations across both spatial locations and four diurnal time windows (Morning, Afternoon, Evening, Night). It parses headerless CSV files, computes row-wise active signal ratios against a -95.0dBm threshold, groups the metrics by location and temporal session, and exports the results into a structured Excel report (General_Diurnal_Spatial_Duty_Cycle.xlsx) and a publication-quality heatmap (General_Diurnal_Spatial_Duty_Cycle_Heatmap.png).
"""
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT = r'C:\Users\User\Documents\RFExplorer\Processed_Data'
OUTPUT_DIR = r'C:\Users\User\Documents\RFExplorer\Result_Sheet'
THRESHOLD = -95.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fast session helper mapping hours
def get_diurnal_session(time_str):
    try:
        hour = int(time_str.split(':')[0])
        if 6 <= hour < 12:
            return 'Morning (6am-12pm)'
        elif 12 <= hour < 16:
            return 'Afternoon (12pm-4pm)'
        elif 16 <= hour < 22:
            return 'Evening (4pm-10pm)'
        else:
            return 'Night (10pm-6am)'
    except Exception:
        return np.nan

# Gather files (excluding backups)
all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.csv'), recursive=True)
all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

raw_records = []

# Streaming File Processing (Extremely Memory-Efficient)
for file in all_files:
    try:
        # Check headers first to pull only frequency and metadata columns
        header_df = pd.read_csv(file, nrows=0)
        freq_cols = [c for c in header_df.columns if c.replace('.', '', 1).isdigit()]
        meta_cols = [c for c in ['Time'] if c in header_df.columns]
        
        if not freq_cols or not meta_cols:
            continue
            
        # Stream only the required columns
        df = pd.read_csv(file, usecols=meta_cols + freq_cols)
        loc = os.path.basename(os.path.dirname(file))
        
        # Categorize temporal sessions
        df['Session'] = df['Time'].apply(get_diurnal_session)
        df = df.dropna(subset=['Session'])
        
        if df.empty:
            continue
            
        # 1. Calculate the active fraction of bins for each sweep (row)
        # 2. Group by Session and compute the mean to get SDC (%)
        sweep_active_fraction = (df[freq_cols] > THRESHOLD).mean(axis=1) * 100
        session_sdc = sweep_active_fraction.groupby(df['Session']).mean()
        
        for session, sdc_val in session_sdc.items():
            raw_records.append({
                'Location': loc,
                'Session': session,
                'Duty Cycle (%)': sdc_val
            })
    except Exception as e:
        print(f"Error processing {os.path.basename(file)}: {e}")

# Build global DataFrame
summary_df = pd.DataFrame(raw_records)

# --- Pivot Processing (Spatial Y-axis, Diurnal X-axis) ---
# Group by Location and Session, taking the mean of all processed block data
pivot_table = summary_df.pivot_table(
    index='Location',
    columns='Session',
    values='Duty Cycle (%)',
    aggfunc='mean'
)

# Reorder columns to logical temporal flow
temporal_order = [
    'Morning (6am-12pm)', 
    'Afternoon (12pm-4pm)', 
    'Evening (4pm-10pm)', 
    'Night (10pm-6am)'
]
pivot_table = pivot_table.reindex(columns=temporal_order)

# Save clean, single-index layout to Excel
excel_output = os.path.join(OUTPUT_DIR, 'General_Diurnal_Spatial_Duty_Cycle.xlsx')
pivot_table.to_excel(excel_output)

# --- High-Resolution Heatmap Plot ---
plt.figure(figsize=(14, 10))

# Custom styling parameters for bold, publication-grade readability
sns.set_theme(style="white")
ax = sns.heatmap(
    pivot_table, 
    annot=True, 
    cmap='viridis', 
    fmt='.2f', 
    linewidths=.5,
    cbar_kws={'label': 'General Spectrum Duty Cycle (%)'},
    annot_kws={'weight': 'bold', 'size': 13} # Bold annotative percentages
)

# Force bold and highly readable fonts for academic presentation
plt.title('General Spatial-Diurnal Spectrum Duty Cycle (VHF & UHF Combined)', fontsize=16, weight='bold', pad=20)
plt.xlabel('Temporal Diurnal Sessions', fontsize=13, weight='bold', labelpad=15)
plt.ylabel('Measured Locations (Spatial)', fontsize=13, weight='bold', labelpad=15)

plt.xticks(fontsize=11, weight='bold', rotation=0)
plt.yticks(fontsize=11, weight='bold')

# Style the color bar label
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
cbar.set_label('General Spectrum Duty Cycle (%)', weight='bold', size=12, labelpad=15)

plt.tight_layout()
heatmap_output = os.path.join(OUTPUT_DIR, 'General_Diurnal_Spatial_Duty_Cycle_Heatmap.png')
plt.savefig(heatmap_output, dpi=300)

print(f"\nAnalysis complete!")
print(f"Excel summary saved to: {excel_output}")
print(f"Heatmap visualization saved to: {heatmap_output}")
