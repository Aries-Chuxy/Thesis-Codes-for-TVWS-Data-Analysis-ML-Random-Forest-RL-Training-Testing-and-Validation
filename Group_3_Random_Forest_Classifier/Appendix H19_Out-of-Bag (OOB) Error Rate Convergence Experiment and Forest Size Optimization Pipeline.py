"""
Description
This Python script evaluates the Out-of-Bag (OOB) generalization error trajectory of a Random Forest Classifier across increasing ensemble sizes (B = 10 to 150 trees at max_depth=15). By ingesting pre-processed spectrum features and spatial-diurnal metadata from a Parquet dataset via Polars, the pipeline determines the optimal tree count convergence threshold, outputs tabular OOB accuracy metrics, and exports a publication-quality OOB convergence plot (OOB_Error_Convergence_Plot.png).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. LOAD PREPARED PARQUET DATASET
# ==========================================
data_path = Path(
    r"C:\RFExplorer_Final_Results\Prepared_RF_ML_Data.parquet"
)
if not data_path.exists():
    data_path = Path(
        r"C:\Users\User\Documents\RFExplorer\Result_Sheet3\Prepared_RF_ML_Data.parquet"
    )

print(f"[+] Loading prepared RF dataset from: {data_path}")
df = pl.read_parquet(data_path)

# Extract spatial/diurnal & spectral features for ML training
feature_cols = [
    "Active_Bin_Ratio",
    "Mean_Power_dBm",
    "Variance_Power_dB2",
    "Max_Power_dBm",
    "Power_Range_H_k",
]

# Include latitude/longitude/elevation if present
for col in ["Latitude", "Longitude", "Elevation"]:
    if col in df.columns:
        feature_cols.append(col)

# Include raw frequency channel features
freq_cols = [c for c in df.columns if "mhz" in c.lower()]
feature_cols.extend(freq_cols)

X = df.select(feature_cols).to_pandas()
y = df["Occupancy_Label"].to_pandas()

print(f"[+] Feature matrix shape: {X.shape}")
print(f"[+] Target variable distribution: {y.value_counts().to_dict()}")

# ==========================================
# 2. OOB CONVERGENCE EXPERIMENT (B = 10 to 150)
# ==========================================
trees_range = range(10, 151, 10)
oob_errors = []
oob_accuracies = []

print(
    "\n[+] Training Random Forest models (max_depth=15) and calculating OOB scores..."
)
print("---------------------------------------------------------------")
print(f"{'Trees (B)':<10} | {'OOB Score (Acc)':<18} | {'OOB Error Rate':<15}")
print("---------------------------------------------------------------")

for n_trees in trees_range:
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=15,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        warm_start=False,
    )
    rf.fit(X, y)

    oob_acc = rf.oob_score_
    oob_err = 1.0 - oob_acc

    oob_accuracies.append(oob_acc)
    oob_errors.append(oob_err)

    print(f"{n_trees:<10} | {oob_acc:<18.5f} | {oob_err:<15.5f}")

# ==========================================
# 3. PLOT OOB ERROR CONVERGENCE CURVE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(
    trees_range,
    oob_errors,
    marker="o",
    linewidth=2.5,
    color="#1f77b4",
    label="OOB Error Rate",
)
plt.axvline(
    x=150,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Selected Ensemble Size (B = 150)",
)

plt.title(
    "Random Forest OOB Error Score Convergence (max_depth = 15)", fontsize=14
)
plt.xlabel("Ensemble Size (Number of Trees, B)", fontsize=12)
plt.ylabel("Out-Of-Bag (OOB) Error Rate", fontsize=12)
plt.xticks(list(trees_range))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Save plot
plot_path = (
    Path(r"C:\RFExplorer_Final_Results") / "OOB_Error_Convergence_Plot.png"
)
plt.savefig(plot_path, dpi=300)
plt.show()

print("\n==============================================")
print(f"[✓] OOB Analysis Completed!")
print(f"[✓] Convergence Plot Saved: {plot_path}")
print("==============================================")
