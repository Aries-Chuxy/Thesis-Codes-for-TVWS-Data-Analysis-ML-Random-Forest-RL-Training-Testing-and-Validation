"""
Description
This Python script executes the final holdout evaluation of the optimized Random Forest Classifier on an unseen 10% test dataset (D_test) comprising 100,000 spectrum sweeps. It performs class-stratified partition slicing using Polars, evaluates primary Cognitive Radio metrics—specifically Probability of Detection (P_d) for PU protection and Probability of False Alarm (P_fa) for SU spectrum harvesting, exports structured numerical results to CSV (Confusion_Matrix_Dtest_10pct.csv), and generates publication-grade 2D confusion matrix heatmaps (Confusion_Matrix_Dtest_10pct_Heatmap.png and .pdf).
"""
import gc
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Set academic styling
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

# ==========================================
# 1. PATHS & DIRECTORY SETUP
# ==========================================
result_sheet3_dir = Path(r"C:\Users\User\Documents\RFExplorer\Result_Sheet3")
result_sheet3_dir.mkdir(parents=True, exist_ok=True)

easy_dest_dir = Path(r"C:\RFExplorer_Final_Results")
easy_dest_dir.mkdir(parents=True, exist_ok=True)

parquet_path = result_sheet3_dir / "Prepared_RF_ML_1M_Data.parquet"
if not parquet_path.exists():
    parquet_path = easy_dest_dir / "Prepared_RF_ML_1M_Data.parquet"

print(f"[+] Loading Prepared 1,000,000 Dataset from:\n    {parquet_path}")

# ==========================================
# 2. DATA SPLITTING (STRICT 80 / 10 / 10 SPLIT)
# ==========================================
df_lazy = pl.scan_parquet(parquet_path)
all_cols = df_lazy.collect_schema().names()

exclude_cols = {
    "Occupancy_Label", "Sweep_Active_State_Y", "Active_Bin_Ratio",
    "Date", "Time", "Sweep", "Milliseconds", "Diurnal_Session",
    "Location", "Location_ID", "Site", "Time_Of_Day", "Band_Range"
}
feature_cols = [c for c in all_cols if c not in exclude_cols]

df_full = df_lazy.select(
    [pl.col(c).cast(pl.Float32) for c in feature_cols] + 
    [pl.col("Occupancy_Label").cast(pl.Int8)]
).collect()

# Separate Positive and Negative Classes
pos_df = df_full.filter(pl.col("Occupancy_Label") == 1)
neg_df = df_full.filter(pl.col("Occupancy_Label") == 0)

pos_len, neg_len = len(pos_df), len(neg_df)

# Proportional Slices for 80% Train / 10% Validation / 10% Test
p_train = pos_df.slice(0, int(pos_len * 0.80))
p_val   = pos_df.slice(int(pos_len * 0.80), int(pos_len * 0.10))
p_test  = pos_df.slice(int(pos_len * 0.90), pos_len - int(pos_len * 0.90))

n_train = neg_df.slice(0, int(neg_len * 0.80))
n_val   = neg_df.slice(int(neg_len * 0.80), int(neg_len * 0.10))
n_test  = neg_df.slice(int(neg_len * 0.90), neg_len - int(neg_len * 0.90))

# Combine and Shuffle Partitions
df_train = pl.concat([p_train, n_train]).sample(fraction=1.0, shuffle=True, seed=42)
df_val   = pl.concat([p_val, n_val]).sample(fraction=1.0, shuffle=True, seed=42)
df_test  = pl.concat([p_test, n_test]).sample(fraction=1.0, shuffle=True, seed=42)

del df_full, pos_df, neg_df, p_train, n_train, p_val, n_val, p_test, n_test
gc.collect()

# Extract NumPy Arrays
X_train = df_train.select(feature_cols).to_numpy()
y_train = df_train["Occupancy_Label"].to_numpy()

X_test = df_test.select(feature_cols).to_numpy()
y_test = df_test["Occupancy_Label"].to_numpy()

del df_train, df_val, df_test
gc.collect()

print(f"[+] Data Partitions Ready:")
print(f"    - Training Partition (80%)      : {X_train.shape}")
print(f"    - Strict Holdout D_test (10%)   : {X_test.shape}\n")

# ==========================================
# 3. FIT FINAL MODEL & EVALUATE ON D_TEST
# ==========================================
print("[+] Training Final Random Forest Classifier...")
rf_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=15,
    max_features=0.15,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_train, y_train)

print("[+] Evaluating Model on Unseen 10% Test Dataset (D_test)...")
y_pred = rf_final.predict(X_test)

# Compute 2x2 Confusion Matrix [TN, FP / FN, TP]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Metrics in Cognitive Radio Context
total_samples = len(y_test)
accuracy = (tp + tn) / total_samples * 100
pd_detection = (tp / (tp + fn)) * 100        # P_d (Sensitivity / Recall)
pfa_false_alarm = (fp / (fp + tn)) * 100     # P_fa (1 - Specificity)

print("\n" + "=" * 60)
print("  2x2 CONFUSION MATRIX ON STRICT 10% TEST DATASET (D_test)")
print("=" * 60)
print(f"  True Negatives  (TN - Vacant TVWS Correct): {tn:,}")
print(f"  False Positives (FP - False Alarm)       : {fp:,}")
print(f"  False Negatives (FN - Missed Detection)  : {fn:,}")
print(f"  True Positives  (TP - Active DTT Correct) : {tp:,}")
print("-" * 60)
print(f"  Overall Accuracy              : {accuracy:.2f}%")
print(f"  Probability of Detection (P_d): {pd_detection:.2f}%  (Protection >= 99%)")
print(f"  Probability of False Alarm(P_fa): {pfa_false_alarm:.2f}% (Harvesting <= 1%)")
print("=" * 60 + "\n")

# Save Confusion Matrix as CSV
cm_df = pd.DataFrame(
    cm, 
    index=["Actual Vacant (0)", "Actual Active (1)"], 
    columns=["Pred Vacant (0)", "Pred Active (1)"]
)

cm_csv = result_sheet3_dir / "Confusion_Matrix_Dtest_10pct.csv"
cm_df.to_csv(cm_csv)
shutil.copy2(cm_csv, easy_dest_dir / cm_csv.name)

# ==========================================
# 4. GENERATE & SAVE HEATMAP VISUALIZATION
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

group_names = ["True Negative (TN)\n[Vacant Spectrum]", "False Positive (FP)\n[False Alarm]", 
               "False Negative (FN)\n[Missed Detection]", "True Positive (TP)\n[Active Protection]"]
group_counts = [f"{value:,}" for value in cm.flatten()]
group_percentages = [f"{value/np.sum(cm):.2%}" for value in cm.flatten()]

labels = [f"{v1}\n\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(
    cm, 
    annot=labels, 
    fmt="", 
    cmap="Blues", 
    cbar=False,
    linewidths=2,
    linecolor="white",
    ax=ax,
    annot_kws={"size": 11, "weight": "bold"}
)

ax.set_title(r"2$\times$2 Confusion Matrix ($\mathcal{D}_{\mathrm{test}}$ 10% Test Set)", fontsize=13, weight="bold", pad=15)
ax.set_xlabel("Predicted Spectrum State", fontsize=11, weight="bold", labelpad=10)
ax.set_ylabel("Actual Spectrum State", fontsize=11, weight="bold", labelpad=10)
ax.set_xticklabels(["Vacant (0)", "Active (1)"], fontsize=10, weight="bold")
ax.set_yticklabels(["Vacant (0)", "Active (1)"], fontsize=10, weight="bold", rotation=0)

plt.tight_layout()

cm_png = result_sheet3_dir / "Confusion_Matrix_Dtest_10pct_Heatmap.png"
cm_pdf = result_sheet3_dir / "Confusion_Matrix_Dtest_10pct_Heatmap.pdf"

plt.savefig(cm_png, dpi=300, bbox_inches="tight")
plt.savefig(cm_pdf, format="pdf", bbox_inches="tight")

print(f"[✓] Confusion Matrix Plot saved to:\n    {cm_png}")
plt.show()
