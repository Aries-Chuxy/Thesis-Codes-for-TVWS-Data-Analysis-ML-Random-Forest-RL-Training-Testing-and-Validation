"""
Description
This Python script evaluates spatial overlap and correlation in spectrum usage across surveyed geographical locations by computing a pairwise Jaccard Similarity Index matrix (0 -100%). It parses raw headerless CSV measurement sweeps, extracts active frequency bin profiles exceeding a -95.0dBm detection threshold for each location, exports the symmetric intersection matrix to an Excel workbook (Spatial_Channel_Intersection_Matrix.xlsx), and generates a high-resolution similarity heatmap (Spatial_Channel_Intersection_Heatmap.png).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Force inline plotting for Jupyter Notebook environments
%matplotlib inline

def get_diurnal_session_fast(time_val):
    """Robust hours-to-session binning framework."""
    try:
        time_str = str(time_val).strip().upper()
        is_pm = 'PM' in time_str
        is_am = 'AM' in time_str
        
        match = re.search(r'(\d+)', time_str)
        if not match:
            return np.nan
        
        hour = int(match.group(1))
        if is_pm or is_am:
            if is_pm and hour != 12: hour += 12
            elif is_am and hour == 12: hour = 0
        else:
            if ':' in time_str:
                hour = int(time_str.split(':')[0])
                
        if 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 16: return 'Afternoon'
        elif 16 <= hour < 22: return 'Evening'
        else: return 'Night'
    except Exception:
        return np.nan

def compute_spatial_diurnal_intersection(data_root, threshold=-95.0):
    print("Initializing Spatial-Diurnal Intersection engine...")
    
    all_files = glob.glob(os.path.join(data_root, '**', '*.csv'), recursive=True)
    all_files = [f for f in all_files if "backup_" not in os.path.basename(f)]
    
    if not all_files:
        print(f"Error: No data found in {data_root}")
        return
        
    loc_groups = {}
    for f in all_files:
        loc = os.path.basename(os.path.dirname(f))
        loc_groups.setdefault(loc, []).append(f)
        
    spatial_diurnal_profiles = {}
    
    # 1. Profile active channels per Location + Diurnal Session combination
    for loc, files in loc_groups.items():
        for file in files:
            try:
                # Read headerless CSV safely
                df = pd.read_csv(file, header=None, low_memory=False)
                
                # Column 2 holds Time; Columns 4+ hold numeric power values
                time_series = df.iloc[:, 2]
                power_data = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
                
                sessions = time_series.apply(get_diurnal_session_fast)
                
                for session, group in power_data.groupby(sessions):
                    if pd.isna(session):
                        continue
                    combo_key = f"{loc}_{session}"
                    
                    # Determine which frequency bins breach the threshold within this session slice
                    is_active = (group > threshold).any(axis=0)
                    active_channels = set(is_active[is_active].index)
                    
                    if combo_key not in spatial_diurnal_profiles:
                        spatial_diurnal_profiles[combo_key] = active_channels
                    else:
                        spatial_diurnal_profiles[combo_key].update(active_channels)
            except Exception as e:
                continue

    if not spatial_diurnal_profiles:
        print("Error: Could not construct spatial-diurnal profiles.")
        return

    # 2. Compute the cross-intersection matrix (Jaccard Index)
    keys = sorted(list(spatial_diurnal_profiles.keys()))
    n_keys = len(keys)
    matrix_data = np.zeros((n_keys, n_keys), dtype=np.float32)
    
    for i in range(n_keys):
        set_i = spatial_diurnal_profiles[keys[i]]
        len_i = len(set_i)
        for j in range(i, n_keys):
            set_j = spatial_diurnal_profiles[keys[j]]
            
            if len_i == 0 or len(set_j) == 0:
                similarity = 0.0
            else:
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                similarity = (intersection / union) * 100.0 if union > 0 else 0.0
                
            matrix_data[i, j] = similarity
            matrix_data[j, i] = similarity
            
    df_matrix = pd.DataFrame(matrix_data, index=keys, columns=keys)
    
    # 3. Print out directly on Jupyter Notebook Environment
    print("\n" + "="*50)
    print("SPATIAL-DIURNAL PROFILE INTERSECTION MATRIX (%)")
    print("="*50)
    display(df_matrix.round(2))
    print("="*50 + "\n")
    
    # 4. Save to designated path
    output_dir = r"C:\Users\User\Documents\RFExplorer\Result_Sheet2"
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, 'Spatial_Diurnal_Intersection_Matrix.xlsx')
    heatmap_path = os.path.join(output_dir, 'Spatial_Diurnal_Intersection_Heatmap.png')
    
    df_matrix.to_excel(excel_path)
    
    # 5. Plot Configuration
    plt.figure(figsize=(16, 14))
    sns.set_theme(style="white")
    
    sns.heatmap(
        df_matrix, 
        annot=True, 
        cmap='magma', 
        fmt='.1f', 
        vmin=0, 
        vmax=100, 
        linewidths=.3,
        cbar_kws={'label': 'Profile Overlap / Jaccard Similarity Index (%)'},
        annot_kws={'size': 9, 'weight': 'bold'}
    )
    
    plt.title('Spatial-Diurnal Spectrum Channel Occupancy Intersection Matrix', fontsize=14, weight='bold', pad=25)
    plt.xticks(rotation=45, ha='right', fontsize=9, weight='bold')
    plt.yticks(rotation=0, fontsize=9, weight='bold')
    plt.tight_layout()
    
    plt.savefig(heatmap_path, dpi=300)
    plt.show()
    
    print(f"[SUCCESS] Spreadsheet matrix saved to: {excel_path}")
    print(f"[SUCCESS] High-resolution heatmap saved to: {heatmap_path}")

# Run analysis
DATA_ROOT = r"C:\Users\User\Documents\RFExplorer\Processed_Data"
compute_spatial_diurnal_intersection(DATA_ROOT, threshold=-95.0)
