"""
Appendix H1. 
RF Explorer CSV Data Trimming and Directory Processing Script
Note: This script automates directory traversal across all Enugu measurement locations. Lines 29–30 bypass device metadata header rows (rows 1–5), while lines 32–33 standardize all dataset inputs by capping signal sweep recordings to 2,500 rows (N = 2,500). The resulting structured .csv datasets are exported to Processed_Data and duplicated in Backup_Data to guarantee raw data integrity during analysis.
"""
import pandas as pd
import shutil
from pathlib import Path

# --- Configuration ---
# Using the absolute path you provided
BASE_DIR = Path(r'C:\Users\User\Documents\RFExplorer\Enugu_Project')
OUTPUT_DIR = Path(r'C:\Users\User\Documents\RFExplorer\Processed_Data')
BACKUP_DIR = Path(r'C:\Users\User\Documents\RFExplorer\Backup_Data')

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def process_files():
    if not BASE_DIR.exists():
        print(f"CRITICAL ERROR: Folder not found at {BASE_DIR}")
        return

    # Iterate through all location sub-folders
    for location_folder in BASE_DIR.iterdir():
        if location_folder.is_dir():
            print(f"--- Processing location: {location_folder.name} ---")
            
            # Prepare destination sub-folders
            (OUTPUT_DIR / location_folder.name).mkdir(exist_ok=True)
            (BACKUP_DIR / location_folder.name).mkdir(exist_ok=True)
            
            for csv_file in location_folder.glob('*.csv'):
                try:
                    # Skip metadata rows 1-5, read from row 6
                    df = pd.read_csv(csv_file, skiprows=5)
                    
                    # Filter: Keep only sweeps 0 to 2499 (total 2500 rows)
                    df_trimmed = df[df['Sweep'] < 2500].copy()
                    
                    # File paths
                    out_path = OUTPUT_DIR / location_folder.name / csv_file.name
                    back_path = BACKUP_DIR / location_folder.name / csv_file.name
                    
                    # Save to processed directory
                    df_trimmed.to_csv(out_path, index=False)
                    
                    # Create duplicate in backup
                    shutil.copy2(out_path, back_path)
                    
                    print(f"Successfully processed: {csv_file.name}")
                    
                except Exception as e:
                    print(f"FAILED to process {csv_file.name}: {e}")

    print("\nAll tasks completed. Data is stored in 'Processed_Data' and 'Backup_Data'.")

if __name__ == "__main__":
    process_files()
