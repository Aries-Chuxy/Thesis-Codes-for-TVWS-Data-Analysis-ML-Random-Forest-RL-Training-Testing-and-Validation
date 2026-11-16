"""
H3_RF Data Feature Engineering and Predictive Model Training Script
Extracts frequency/signal features, trains supervised machine learning regressors (e.g., Random Forest), and evaluates RF propagation performance metrics.
"""
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Allocate 4 threads to keep OS and Chrome responsive
os.environ["POLARS_MAX_THREADS"] = "4"

# ==========================================
# 1. INPUT & DIRECT OUTPUT DIRECTORY PATHS
# ==========================================
input_dir = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer\Processed_Data")
if not input_dir.exists():
    input_dir = Path(r"C:\Users\User\Documents\RFExplorer\Processed_Data")

# Primary output location
output_dir = Path(r"C:\Users\User\Documents\RFExplorer\Result_Sheet3")
output_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(input_dir.rglob("*.csv"))

print(f"[+] Input Directory : {input_dir}")
print(f"[+] Output Directory: {output_dir}")
print(f"[+] Found {len(csv_files)} CSV files.")

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{input_dir}'!")

# ==========================================
# 2. FREQUENCY SPECTRUM BLOCK FILTERING
# ==========================================
print("[+] Scanning and filtering targeted VHF/UHF spectrum blocks...")

metadata_cols = {"Date", "Time", "Latitude", "Longitude", "Elevation", "Sweep"}
raw_freq_colnames = set()

for f in csv_files:
    header_df = pl.read_csv(f, n_rows=5)
    freqs = [c for c in header_df.columns if c not in metadata_cols]
    raw_freq_colnames.update(freqs)


def parse_freq_mhz(col_name):
    """Extract numeric frequency in MHz from column header string."""
    try:
        clean_name = col_name.lower().replace("mhz", "").strip()
        return float(clean_name)
    except ValueError:
        return None


def get_spectrum_block(freq_mhz):
    """Map frequency to VHF or UHF sub-blocks."""
    if 174.0 <= freq_mhz <= 230.0:
        return "VHF_174_230"
    elif 470.0 <= freq_mhz < 570.0:
        return "UHF_470_570"
    elif 570.0 <= freq_mhz < 670.0:
        return "UHF_570_670"
    elif 670.0 <= freq_mhz < 770.0:
        return "UHF_670_770"
    elif 770.0 <= freq_mhz <= 870.0:
        return "UHF_770_870"
    return None


valid_freq_cols = []
block_counts = {
    "VHF_174_230": 0,
    "UHF_470_570": 0,
    "UHF_570_670": 0,
    "UHF_670_770": 0,
    "UHF_770_870": 0,
}

for c in raw_freq_colnames:
    freq_val = parse_freq_mhz(c)
    if freq_val is not None:
        block = get_spectrum_block(freq_val)
        if block is not None:
            valid_freq_cols.append((freq_val, c, block))
            block_counts[block] += 1

valid_freq_cols.sort(key=lambda x: x[0])
freq_colnames = [col_tuple[1] for col_tuple in valid_freq_cols]

print(f"[+] Total Valid VHF/UHF Channels: {len(freq_colnames)}")
for block, count in block_counts.items():
    print(f"    - {block}: {count} channels")

# Lazy scanning
lazy_dfs = [pl.scan_csv(f, infer_schema_length=10000) for f in csv_files]
full_lazy = pl.concat(lazy_dfs, how="diagonal_relaxed")


# ==========================================
# 3. DATA PREPARATION & FEATURE ENGINEERING
# ==========================================
def prepare_rf_data_for_random_forest(full_lazy, freq_colnames, out_dir):
    print("[+] Building 7D Spatial-Diurnal Feature Matrix & SDC Features...")

    THRESHOLD_DBM = -95.0
    VOTING_TAU = 0.10  # 10% sub-threshold voting

    # 1. Cast frequency columns safely and fill unobserved channels with -110 dBm
    cast_freq_exprs = [
        pl.col(c).cast(pl.Float64, strict=False) for c in freq_colnames
    ]
    full_lazy_prep = full_lazy.with_columns(cast_freq_exprs)

    fill_null_exprs = [pl.col(c).fill_null(-110.0) for c in freq_colnames]
    full_lazy_prep = full_lazy_prep.with_columns(fill_null_exprs)

    # 2. Sub-Threshold Voting Mechanism (\phi and Active_Bin_Ratio)
    phi_exprs = [
        (pl.col(c) > THRESHOLD_DBM).cast(pl.Float32).alias(f"phi_{c}")
        for c in freq_colnames
    ]

    full_lazy_prep = full_lazy_prep.with_columns(phi_exprs).with_columns(
        (
            pl.sum_horizontal([f"phi_{c}" for c in freq_colnames])
            / len(freq_colnames)
        ).alias("Active_Bin_Ratio")
    )

    # Sweep Active State Y_(i,C) & Occupancy Label
    full_lazy_prep = full_lazy_prep.with_columns([
        pl.when(pl.col("Active_Bin_Ratio") >= VOTING_TAU)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("Sweep_Active_State_Y"),
        pl.when(pl.col("Active_Bin_Ratio") >= VOTING_TAU)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("Occupancy_Label"),
    ])

    # 3. Compute Spectral Features
    freq_cols_expr = [pl.col(c) for c in freq_colnames]

    full_lazy_prep = full_lazy_prep.with_columns([
        pl.mean_horizontal(freq_cols_expr)
        .cast(pl.Float32)
        .alias("Mean_Power_dBm"),
        pl.max_horizontal(freq_cols_expr)
        .cast(pl.Float32)
        .alias("Max_Power_dBm"),
        pl.min_horizontal(freq_cols_expr)
        .cast(pl.Float32)
        .alias("Min_Power_dBm"),
        pl.concat_list(freq_colnames).list.var().alias("Variance_Power_dB2"),
    ])

    full_lazy_prep = full_lazy_prep.with_columns(
        (pl.col("Max_Power_dBm") - pl.col("Min_Power_dBm")).alias(
            "Power_Range_H_k"
        )
    )

    # 4. Diurnal Session Classification (Corrected Polars is_between syntax)
    if "Time" in full_lazy_prep.collect_schema().names():
        time_col = pl.col("Time").str.to_time(strict=False)
        full_lazy_prep = full_lazy_prep.with_columns(
            pl.when(
                time_col.is_between(
                    pl.time(6, 0, 0), pl.time(11, 59, 59), closed="both"
                )
            )
            .then(pl.lit("Morning"))
            .when(
                time_col.is_between(
                    pl.time(12, 0, 0), pl.time(15, 59, 59), closed="both"
                )
            )
            .then(pl.lit("Afternoon"))
            .when(
                time_col.is_between(
                    pl.time(16, 0, 0), pl.time(21, 59, 59), closed="both"
                )
            )
            .then(pl.lit("Evening"))
            .otherwise(pl.lit("Night"))
            .alias("Diurnal_Session")
        )

    # Drop temporary phi columns to optimize storage
    phi_cols_to_drop = [f"phi_{c}" for c in freq_colnames]
    full_lazy_prep = full_lazy_prep.drop(phi_cols_to_drop)

    # ==========================================
    # STREAMING DIRECTLY TO DISK
    # ==========================================
    parquet_path = out_dir / "Prepared_RF_ML_Data.parquet"
    excel_path = out_dir / "Prepared_RF_ML_Data.xlsx"

    print("[+] Streaming output directly to Parquet (Low Memory Footprint)...")
    full_lazy_prep.sink_parquet(parquet_path, compression="zstd")
    print(f"[✓] Parquet file written: {parquet_path}")

    # ==========================================
    # EXCEL EXPORT (ZIP64 ENABLED)
    # ==========================================
    print(
        "[+] Sampling dataset for Excel export with ZIP64 extensions enabled..."
    )
    summary_cols = [
        "Sweep_Active_State_Y",
        "Active_Bin_Ratio",
        "Mean_Power_dBm",
        "Variance_Power_dB2",
        "Max_Power_dBm",
        "Power_Range_H_k",
        "Occupancy_Label",
    ]

    schema_names = full_lazy_prep.collect_schema().names()
    if "Diurnal_Session" in schema_names:
        summary_cols.append("Diurnal_Session")
    if "Latitude" in schema_names:
        summary_cols.extend(["Latitude", "Longitude", "Elevation"])

    sample_df = pl.read_parquet(parquet_path, n_rows=10000)

    print(f"[+] Exporting Excel Visualization file: {excel_path}")
    sample_df.write_excel(excel_path, use_zip64=True)

    # ==========================================
    # FEATURE CORRELATION HEATMAP (FIXED FOR STRING COLUMNS)
    # ==========================================
    print("[+] Generating feature correlation heatmap...")
    heatmap_cols = summary_cols + freq_colnames[:10]
    valid_heatmap_cols = [c for c in heatmap_cols if c in sample_df.columns]

    corr_df = (
        sample_df.select(valid_heatmap_cols).to_pandas().corr(numeric_only=True)
    )

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr_df, annot=False, cmap="coolwarm", linewidths=0.5, cbar=True
    )
    plt.title("VHF/UHF Spectrum Correlation Heatmap", fontsize=13)
    plt.tight_layout()

    heatmap_path = out_dir / "RF_Features_Coordinates_Heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # ==========================================
    # EXTRA COPY TO EASY PHYSICAL DRIVE LOCATION
    # ==========================================
    easy_dest_dir = Path(r"C:\RFExplorer_Final_Results")
    easy_dest_dir.mkdir(parents=True, exist_ok=True)

    for p in [parquet_path, excel_path, heatmap_path]:
        shutil.copy2(p, easy_dest_dir / p.name)

    print("\n==============================================")
    print("[✓] DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"    1. Parquet Dataset : {parquet_path}")
    print(f"    2. Excel Dataset   : {excel_path}")
    print(f"    3. Heatmap Image   : {heatmap_path}")
    print("----------------------------------------------")
    print(
        f"[✓] Direct Physical Access Copy: {easy_dest_dir / 'Prepared_RF_ML_Data.parquet'}"
    )
    print("==============================================")

    # Automatically open the output folder on screen
    os.startfile(easy_dest_dir)


if __name__ == "__main__":
    prepare_rf_data_for_random_forest(full_lazy, freq_colnames, output_dir)

