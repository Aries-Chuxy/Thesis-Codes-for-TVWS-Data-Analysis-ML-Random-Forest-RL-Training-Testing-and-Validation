"""
Appendix H.4_Spatial Metadata Integration and Dataset Splitting Pipeline (H.4_dataset_prep_spatial_split.py)
Description:
This script executes high-speed ingestion and spatial-spectral normalization across raw survey files to build partitioned datasets for the Reinforcement Learning (RL) environment and Random Forest classifiers. Utilizing polars for memory-efficient lazy evaluation, the script scans CSV traces, restricts telemetry to the primary 2,500 sweeps, parses temporal sessions (Morning, Afternoon, Evening, Night) and frequency bands (174–870 MHz) from file paths, and performs a relational join with GIS spatial coordinates from an external Excel file. Finally, it divides the 20 survey locations into a spatially non-overlapping 80/10/10 split (16 Training, 2 Testing, 2 Validation locations) and exports compressed Apache Parquet datasets using diagonal relaxed concatenation to resolve varying channel schemas.
"""
import gc
import os
from pathlib import Path
import pandas as pd
import polars as pl

# ==========================================
# 1. DIRECTORY CONFIGURATION & PATHS
# ==========================================
INPUT_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer\Processed_Data")
OUTPUT_DIR = Path(r"C:\Users\User\OneDrive\Documents\RFExplorer\Result_Sheet3\Result_Sheet3_Retrained")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[*] Reading data from: {INPUT_DIR}")
print(f"[*] Output directory ready at: {OUTPUT_DIR}\n")

# ==========================================
# 2. READ & PREPARE EXCEL COORDINATES
# ==========================================
xlsx_files = list(INPUT_DIR.glob("*.xlsx")) + list(INPUT_DIR.glob("*.xls"))
if not xlsx_files:
    raise FileNotFoundError(f"No Excel coordinates file found inside: {INPUT_DIR}")

COORDS_FILE = xlsx_files[0]
print(f"[+] Loaded spatial metadata Excel: {COORDS_FILE.name}")

coords_df = pd.read_excel(COORDS_FILE)
coords_df.columns = [str(c).strip() for c in coords_df.columns]

# Auto-detect Location column in Excel
loc_col = [c for c in coords_df.columns if "loc" in c.lower() or "site" in c.lower() or "name" in c.lower()][0]

# Convert Excel coordinates to Polars DataFrame for high-speed joining
coords_polars = pl.from_pandas(coords_df)

# ==========================================
# 3. DISCOVER & SPATIAL SPLIT LOCATIONS (16 / 2 / 2)
# ==========================================
all_locations = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
total_locations = len(all_locations)

if total_locations == 0:
    raise FileNotFoundError(f"No location subfolders found in {INPUT_DIR}.")

print(f"[+] Total Locations Found ({total_locations}): {all_locations}")

# 80/10/10 Spatial Split across 20 locations
train_locs = set(all_locations[:16])
test_locs = set(all_locations[16:18])
val_locs = set(all_locations[18:20])

print(f"   - Train Locations ({len(train_locs)}): {sorted(list(train_locs))}")
print(f"   - Test Locations ({len(test_locs)}): {sorted(list(test_locs))}")
print(f"   - Val Locations ({len(val_locs)}): {sorted(list(val_locs))}\n")

def get_split_category(loc_name: str) -> str:
    if loc_name in train_locs:
        return "train"
    elif loc_name in test_locs:
        return "test"
    else:
        return "val"

def parse_filename_metadata(file_name: str):
    """Extracts session and band range metadata from filename."""
    name_upper = file_name.upper()

    if "MORNING" in name_upper:
        session = "Morning"
    elif "AFTERNOON" in name_upper:
        session = "Afternoon"
    elif "EVENING" in name_upper:
        session = "Evening"
    elif "NIGHT" in name_upper:
        session = "Night"
    else:
        session = "Unknown"

    if "174-230" in name_upper:
        band = "174-230MHz"
    elif "470-570" in name_upper:
        band = "470-570MHz"
    elif "570-670" in name_upper:
        band = "570-670MHz"
    elif "670-770" in name_upper:
        band = "670-770MHz"
    elif "770-870" in name_upper:
        band = "770-870MHz"
    else:
        band = "Unknown"

    return session, band

# ==========================================
# 4. FAST CSV SCANNING & METADATA MERGING
# ==========================================
train_frames = []
test_frames = []
val_frames = []

print("[*] Processing CSV files and integrating spatial coordinates...")

total_files_processed = 0

for loc_folder in INPUT_DIR.iterdir():
    if not loc_folder.is_dir():
        continue

    loc_name = loc_folder.name
    split_cat = get_split_category(loc_name)

    for csv_file in loc_folder.glob("*.csv"):
        try:
            # Fast scan via Polars
            df_lazy = pl.scan_csv(csv_file, ignore_errors=True)
            session, band = parse_filename_metadata(csv_file.name)

            # Filter sweeps and attach Location ID and metadata
            df_processed = df_lazy.filter(pl.col("Sweep") < 2500).with_columns(
                [
                    pl.lit(loc_name).alias("Location"),
                    pl.lit(session).alias("Diurnal_Session"),
                    pl.lit(band).alias("Band_Range"),
                ]
            )

            # Collect to DataFrame
            df_curr = df_processed.collect()

            # Merge spatial metadata from Coordinates Excel
            df_curr = df_curr.join(coords_polars, left_on="Location", right_on=loc_col, how="left")

            if split_cat == "train":
                train_frames.append(df_curr)
            elif split_cat == "test":
                test_frames.append(df_curr)
            elif split_cat == "val":
                val_frames.append(df_curr)

            total_files_processed += 1

        except Exception as e:
            print(f"Warning: Error reading {csv_file.name}: {e}")

print(f"[✓] Successfully read, merged coordinates, and parsed {total_files_processed} CSV files.\n")

# ==========================================
# 5. CONCATENATE WITH RELAXED MODE & SAVE
# ==========================================
splits = [("Train", train_frames), ("Test", test_frames), ("Val", val_frames)]

for split_name, frame_list in splits:
    if frame_list:
        print(f"[*] Concatenating {split_name} dataset...")
        
        # how="diagonal_relaxed" resolves ShapeError when CSV column names differ across frequency bands
        df_split = pl.concat(frame_list, how="diagonal_relaxed")

        out_parquet = OUTPUT_DIR / f"RF_ML_Dataset_{split_name}.parquet"
        df_split.write_parquet(out_parquet, compression="snappy")
        
        print(f"   [✓] Saved {split_name} Parquet ({len(df_split):,} rows) -> {out_parquet.name}")

        del df_split, frame_list
        gc.collect()

print("\n" + "=" * 65)
print("  DATASET PREPARATION & COORDINATE INTEGRATION COMPLETED SUCCESSFULLY!")
print(f"  All outputs saved to: {OUTPUT_DIR}")
print("=" * 65)
