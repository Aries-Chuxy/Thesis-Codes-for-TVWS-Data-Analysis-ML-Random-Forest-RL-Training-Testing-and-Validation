"""
Description
This Python script evaluates the trade-off between the Probability of Detection (P_d) and Probability of False Alarm (P_fa) by computing the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) metric for the Random Forest classifier on the holdout test partition (D_test). It streams data using Polars, calculates decision threshold metrics exported to CSV (ROC_Curve_Metrics_Dtest.csv), and generates publication-grade 2D ROC plots highlighting the PU operational decision point (ROC_Curve_Dtest.png and .pdf).
"""
import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Set academic paper styling
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

# ==========================================
# 1. PATHS & SETUP
# ==========================================
result_sheet3_dir = Path(r"C:\Users\User\Documents\RFExplorer\Result_Sheet3")
easy_dest_dir = Path(r"C:\RFExplorer_Final_Results")

parquet_path = result_sheet3_dir / "Prepared_RF_ML_1M_Data.parquet"
if not parquet_path.exists():
    parquet_path = easy_dest_dir / "Prepared_RF_ML_1M_Data.parquet"

print(f"[+] Loading Prepared Dataset: {parquet_path.name}")

# ==========================================
# 2. FAST POLARS DATA LOADING & SPLITTING
# ==========================================
df_lazy = pl.scan_parquet(parquet_path)
all_cols = df_lazy.collect_schema().names()

exclude_cols = {
    "Occupancy_Label", "Sweep_Active_State_Y", "Active_Bin_Ratio",
    "Date", "Time", "Sweep", "Milliseconds", "Diurnal_Session",
    "Location", "Location_ID", "Site", "Time_Of_Day", "Band_Range"
}
feature_cols = [c for c in all_cols if c not in exclude_cols]

# Efficient Float32 casting
df_full = df_lazy.select(
    [pl.col(c).cast(pl.Float32) for c in feature_cols] + 
    [pl.col("Occupancy_Label").cast(pl.Int8)]
).collect()

pos_df = df_full.filter(pl.col("Occupancy_Label") == 1)
neg_df = df_full.filter(pl.col("Occupancy_Label") == 0)

pos_len, neg_len = len(pos_df), len(neg_df)

# Exact 80% Train / 10% Validation / 10% Test Partitioning
p_train = pos_df.slice(0, int(pos_len * 0.80))
p_test  = pos_df.slice(int(pos_len * 0.90), pos_len - int(pos_len * 0.90))

n_train = neg_df.slice(0, int(neg_len * 0.80))
n_test  = neg_df.slice(int(neg_len * 0.90), neg_len - int(neg_len * 0.90))

df_train = pl.concat([p_train, n_train]).sample(fraction=1.0, shuffle=True, seed=42)
df_test  = pl.concat([p_test, n_test]).sample(fraction=1.0, shuffle=True, seed=42)

# Free Memory Immediately
del df_full, pos_df, neg_df, p_train, p_test, n_train, n_test
gc.collect()

X_train = df_train.select(feature_cols).to_numpy()
y_train = df_train["Occupancy_Label"].to_numpy()

X_test = df_test.select(feature_cols).to_numpy()
y_test = df_test["Occupancy_Label"].to_numpy()

del df_train, df_test
gc.collect()

# ==========================================
# 3. MODEL FIT & FAST ROC-AUC EVALUATION
# ==========================================
print("[+] Fitting Random Forest Model...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=15,
    max_features=0.15,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("[+] Computing Prediction Probabilities on D_test...")
# Extract probability for active class (1) in Float32 to save RAM
y_scores = rf.predict_proba(X_test)[:, 1].astype(np.float32)

# Calculate FPR (P_fa), TPR (P_d), and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n[✓] Computed ROC-AUC Score: {roc_auc:.4f}")

# Save metrics CSV
roc_metrics_df = pd.DataFrame({"FPR_Pfa": fpr, "TPR_Pd": tpr, "Thresholds": thresholds})
roc_csv = result_sheet3_dir / "ROC_Curve_Metrics_Dtest.csv"
roc_metrics_df.to_csv(roc_csv, index=False)
shutil.copy2(roc_csv, easy_dest_dir / roc_csv.name)

# ==========================================
# 4. PUBLICATION-READY ROC PLOT
# ==========================================
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

# Plot ROC curve with Cognitive Radio notation (P_d vs. P_fa)
ax.plot(
    fpr, tpr, 
    color="#005580", 
    lw=2.5, 
    label=rf"Random Forest ROC Curve ($\mathrm{{AUC}} = {roc_auc:.4f}$)"
)

# Plot Random Chance Baseline
ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier Baseline ($\mathrm{AUC} = 0.50$)")

# Mark Operational Operating Point (P_fa = 0.00, P_d = 0.9743)
ax.scatter([0.00], [0.9743], color="crimson", s=60, zorder=5, label=r"Operating Point ($P_{\mathrm{fa}}=0.00\%, P_{\mathrm{d}}=97.43\%$)")

# Annotations & Formatting
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel(r"Probability of False Alarm, $P_{\mathrm{fa}}$ (False Positive Rate)", fontsize=11, weight="bold")
ax.set_ylabel(r"Probability of Detection, $P_{\mathrm{d}}$ (True Positive Rate)", fontsize=11, weight="bold")
ax.set_title(r"Receiver Operating Characteristic (ROC) Curve ($\mathcal{D}_{\mathrm{test}}$)", fontsize=12, weight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()

# Save Figures
roc_png = result_sheet3_dir / "ROC_Curve_Dtest.png"
roc_pdf = result_sheet3_dir / "ROC_Curve_Dtest.pdf"

plt.savefig(roc_png, dpi=300, bbox_inches="tight")
plt.savefig(roc_pdf, format="pdf", bbox_inches="tight")

print(f"[✓] ROC Curve Plot saved to:\n    {roc_png}")
plt.show()
