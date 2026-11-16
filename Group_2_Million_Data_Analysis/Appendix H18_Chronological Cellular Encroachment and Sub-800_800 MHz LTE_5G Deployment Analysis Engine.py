"""
Description
This Python script evaluates potential cellular encroachment and spectrum utilization across six chronologically ordered sub-bands within the 766MHz - 870MHz frequency range (including legacy TV channels, guard bands, duplex gaps, and LTE Band 20 / 5G n20 uplink and downlink allocations). It streams sweep data using headerless C-engine parsing, calculates threshold breach duty cycles ({PSD > -95.0dBm) and mean PSD levels per spatial location, exports full metrics to Excel (Cellular_Encroachment_766_870MHz_Analysis.xlsx), and plots an annotated encroachment heatmap (Cellular_Encroachment_766_870MHz_Heatmap.png).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Inline plotting for Jupyter Notebook environments
%matplotlib inline

# Target Cellular Encroachment / Legacy TV Band Allocations (Strictly 766 MHz - 870 MHz)
# Explicitly ordered chronologically from lowest to highest
CELLULAR_CHANNELS = {
    "1. Legacy_Ch_58_59 (766-782MHz)": (766.0, 782.0),
    "2. Guard_Band_Lower (782-791MHz)": (782.0, 791.0),
    "3. LTE_B20_5G_n20_DL (791-821MHz)": (791.0, 821.0),
    "4. Duplex_Gap (821-832MHz)": (821.0, 832.0),
    "5. LTE_B20_5G_n20_UL (832-862MHz)": (832.0, 862.0),
    "6. Guard_Band_Upper (862-870MHz)": (862.0, 870.0)
}

def analyze_cellular_encroachment_766_870(data_root, psd_threshold=-95.0, chunksize=50000):
    print("="*80)
    print("CELLULAR ENCROACHMENT ANALYSIS (STRICT RANGE: 766 MHz - 870 MHz)")
    print("="*80)

    all_files = glob.glob(os.path.join(data_root, '**', '*.csv'), recursive=True)
    all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]

    if not all_files:
        print(f"Error: No processed CSV files found in: {data_root}")
        return

    # Enforce exact chronological order for the X-axis
    ordered_bands = list(CELLULAR_CHANNELS.keys())
    records = []

    for file in all_files:
        loc = os.path.basename(os.path.dirname(file))
        
        try:
            # Sample header to isolate numeric frequency columns
            sample = pd.read_csv(file, nrows=2)
            numeric_cols = [c for c in sample.columns if str(c).replace('.', '', 1).isdigit()]
            
            if not numeric_cols:
                numeric_cols = list(sample.columns[1:])

            col_freqs = np.array([float(c) for c in numeric_cols], dtype=np.float32)

            # Strictly filter for columns between 766.0 MHz and 870.0 MHz
            strict_range_mask = (col_freqs >= 766.0) & (col_freqs <= 870.0)
            
            if not np.any(strict_range_mask):
                continue

            target_cols = [col for idx, col in enumerate(numeric_cols) if strict_range_mask[idx]]
            target_col_freqs = col_freqs[strict_range_mask]

            # Re-map sub-band masks to target_cols subset
            sub_masks = {
                b_name: (target_col_freqs >= f_start) & (target_col_freqs <= f_end)
                for b_name, (f_start, f_end) in CELLULAR_CHANNELS.items()
            }

            # Low-memory streaming C-engine processing
            chunks = pd.read_csv(file, usecols=target_cols, chunksize=chunksize, engine='c')

            file_breaches = {b: 0 for b in ordered_bands}
            file_psd_sum = {b: 0.0 for b in ordered_bands}
            file_total_bins = {b: 0 for b in ordered_bands}

            for chunk in chunks:
                mat = chunk.values.astype(np.float32)
                n_sweeps = mat.shape[0]

                for b_name, mask in sub_masks.items():
                    if not np.any(mask):
                        continue
                    sub_mat = mat[:, mask]
                    
                    active_mask = sub_mat > psd_threshold
                    file_breaches[b_name] += int(np.sum(active_mask))
                    file_total_bins[b_name] += n_sweeps * sub_mat.shape[1]
                    file_psd_sum[b_name] += float(np.sum(sub_mat))

            for b_name in ordered_bands:
                if file_total_bins[b_name] > 0:
                    records.append({
                        "Location": loc,
                        "SubBand": b_name,
                        "Active_Bins": file_breaches[b_name],
                        "Total_Bins": file_total_bins[b_name],
                        "Mean_PSD_dBm": file_psd_sum[b_name] / file_total_bins[b_name]
                    })

        except Exception as e:
            continue

    if not records:
        print("[CRITICAL ERROR] No data parsed within the 766 MHz - 870 MHz range.")
        return

    # Compile and calculate duty cycles
    df_raw = pd.DataFrame(records)
    df_grouped = df_raw.groupby(["Location", "SubBand"]).agg({
        "Active_Bins": "sum",
        "Total_Bins": "sum",
        "Mean_PSD_dBm": "mean"
    }).reset_index()

    df_grouped["Duty_Cycle_Pct"] = (df_grouped["Active_Bins"] / df_grouped["Total_Bins"]) * 100.0

    # Save Excel Assets
    output_dir = r"C:\Users\User\Documents\RFExplorer\Result_Sheet2"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, 'Cellular_Encroachment_766_870MHz_Analysis.xlsx')
    df_grouped.to_excel(excel_path, index=False)

    # Pivot table with strict chronological column reindexing
    pivot = df_grouped.pivot(index="Location", columns="SubBand", values="Duty_Cycle_Pct")
    
    # Ensure all ordered bands are present on X-axis
    for b in ordered_bands:
        if b not in pivot.columns:
            pivot[b] = 0.0
            
    pivot = pivot[ordered_bands].fillna(0.0)

    # Clean display labels for X-axis
    clean_x_labels = [b.split('. ')[1] for b in ordered_bands]

    # Plot Matrix Heatmap
    plt.figure(figsize=(18, 10))
    ax = sns.heatmap(
        pivot,
        annot=True,
        cmap='YlOrRd',
        fmt='.1f',
        vmin=0,
        vmax=100,
        linewidths=1.5,
        linecolor='white',
        cbar_kws={'label': 'Duty Cycle / Encroachment Rate (%)'},
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    
    ax.set_xticklabels(clean_x_labels, rotation=20, ha='right', weight='bold', fontsize=11)
    
    plt.title('Cellular Encroachment & LTE/5G Downlink Deployment (766 MHz - 870 MHz)', weight='bold', fontsize=16, pad=20)
    plt.xlabel('Chronological Frequency Sub-Bands (Lowest to Highest)', weight='bold', fontsize=13)
    plt.ylabel('Spatial Location', weight='bold', fontsize=13)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'Cellular_Encroachment_766_870MHz_Heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"[SUCCESS] Analysis complete! Excel report saved to: {excel_path}")
    print(f"[SUCCESS] Heatmap figure saved to: {heatmap_path}")

# Run execution
DATA_ROOT = r"C:\Users\User\Documents\RFExplorer\Processed_Data"
analyze_cellular_encroachment_766_870(DATA_ROOT, psd_threshold=-95.0)
