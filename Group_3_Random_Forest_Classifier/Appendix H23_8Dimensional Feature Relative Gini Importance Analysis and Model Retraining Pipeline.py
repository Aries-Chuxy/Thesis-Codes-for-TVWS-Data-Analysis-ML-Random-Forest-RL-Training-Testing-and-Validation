"""
Description
This Python script extracts an 8-dimensional spatial-diurnal and spectral feature set (P_C, σ_C^2, P_max, 〖Lat〗_i, 〖Long〗_i, h_elev, D_spatial, H_k) from training datasets using Polars lazy evaluation. It retrains a Random Forest Classifier evaluated at the -95.0dBm PU occupancy threshold, computes feature Gini importance ranks, and generates an annotated visualization bar chart (Gini_Importance_vs_Feature_Dimension.png).
"""
import os
import gc
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PATHS & SETUP
# ==========================================
BASE_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer")
DATA_DIR = BASE_DIR / "Result_Sheet3" / "Result_Sheet3_Retrained"
OUTPUT_DIR = DATA_DIR

train_parquet = DATA_DIR / "RF_ML_Dataset_Train.parquet"

lazy_df = pl.scan_parquet(train_parquet)
schema_cols = lazy_df.collect_schema().names()

def find_col(keywords):
    for col in schema_cols:
        if any(k in col.lower() for k in keywords):
            return col
    return None

lat_col = find_col(["lat"])
long_col = find_col(["long", "lon"])
elev_col = find_col(["elev", "elav"])
density_col = find_col(["density", "spatial_d", "d_spatial"])

lat_expr = pl.col(lat_col).alias("Latitude") if lat_col else pl.lit(0.0).alias("Latitude")
long_expr = pl.col(long_col).alias("Longitude") if long_col else pl.lit(0.0).alias("Longitude")
elev_expr = pl.col(elev_col).alias("Elevation") if elev_col else pl.lit(0.0).alias("Elevation")
density_expr = pl.col(density_col).alias("Spatial_Density") if density_col else pl.lit(0.0).alias("Spatial_Density")

meta_cols = [c for c in [lat_col, long_col, elev_col, density_col, "Sweep", "Date", "Time", "Milliseconds", "Location", "Diurnal_Session", "Band_Range", "Occupancy_Label"] if c]
freq_cols = [c for c in schema_cols if c not in meta_cols]

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
mean_power_expr = pl.mean_horizontal(freq_cols).alias("Mean_Power")
squared_diff_exprs = [(pl.col(c) - mean_power_expr) ** 2 for c in freq_cols]
variance_expr = pl.mean_horizontal(squared_diff_exprs).alias("Spectral_Variance")
peak_power_expr = pl.max_horizontal(freq_cols).alias("Peak_Power")

if "Time" in schema_cols:
    hour_expr = pl.col("Time").str.strptime(pl.Time, "%H:%M:%S", strict=False).dt.hour().fill_null(12).alias("Hour_Index")
else:
    hour_expr = pl.lit(12).alias("Hour_Index")

df_features = lazy_df.select([
    mean_power_expr,
    variance_expr,
    peak_power_expr,
    lat_expr,
    long_expr,
    elev_expr,
    density_expr,
    hour_expr
]).collect()

X_8D = df_features.to_pandas().fillna(0)
del df_features
gc.collect()

rename_dict = {
    "Mean_Power": r"Mean Power ($\bar{P}_C$)",
    "Spectral_Variance": r"Spectral Variance ($\sigma^2_C$)",
    "Peak_Power": r"Peak Power ($P_{\mathrm{max}}$)",
    "Latitude": r"Latitude ($\mathrm{Lat}_i$)",
    "Longitude": r"Longitude ($\mathrm{Long}_i$)",
    "Elevation": r"Terrain Elevation ($h_{\mathrm{elev}}$)",
    "Spatial_Density": r"Spatial Density ($D_{\mathrm{spatial}}$)",
    "Hour_Index": r"Hour Index ($H_k$)"
}
X_8D.rename(columns=rename_dict, inplace=True)

# ==========================================
# 3. OCCUPANCY LABEL (-95 dBm HARD THRESHOLD)
# ==========================================
THRESHOLD_DBM = -95.0

# Priority 1: Use existing 'Occupancy_Label' if pre-calculated in Parquet schema
if "Occupancy_Label" in schema_cols:
    y = lazy_df.select("Occupancy_Label").collect().to_pandas()["Occupancy_Label"]
    print(f"[✓] Using pre-existing Occupancy_Label from dataset schema.")
else:
    # Priority 2: Apply exact -95 dBm threshold on Mean Power
    y = (X_8D[r"Mean Power ($\bar{P}_C$)"] > THRESHOLD_DBM).astype(int)
    print(f"[✓] Applied strict threshold: Mean Power > {THRESHOLD_DBM} dBm")

# ==========================================
# 4. RANDOM FOREST RETRAINING & PLOTTING
# ==========================================
print("[*] Retraining Random Forest Classifier at -95 dBm threshold...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_8D, y)

importance_df = pd.DataFrame({
    'Feature Dimension': X_8D.columns,
    'Relative Gini Importance': rf.feature_importances_
}).sort_values(by='Relative Gini Importance', ascending=False)

plt.figure(figsize=(12, 6), dpi=300)
sns.set_theme(style="whitegrid")

bars = plt.bar(
    importance_df['Feature Dimension'], 
    importance_df['Relative Gini Importance'], 
    color='#1f77b4', 
    edgecolor='black', 
    alpha=0.85
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height + 0.01, 
        f'{height:.4f}', 
        ha='center', 
        va='bottom', 
        fontsize=9, 
        fontweight='bold'
    )

plt.title('Relative Gini Importance vs. Feature Dimension (-95 dBm Threshold)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Feature Dimension', fontsize=12, fontweight='bold')
plt.ylabel('Relative Gini Importance', fontsize=12, fontweight='bold')
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.ylim(0, max(importance_df['Relative Gini Importance']) * 1.15)
plt.tight_layout()

plot_path = OUTPUT_DIR / "Gini_Importance_vs_Feature_Dimension.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"[✓] Retrained model and updated plot saved to: {plot_path}")
