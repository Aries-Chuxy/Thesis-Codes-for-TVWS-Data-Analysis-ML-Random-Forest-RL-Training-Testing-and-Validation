"""
Appendix H2_Coordinates Transformation Script (DMS to Decimal Degrees)
Converts field survey GPS coordinates from Degrees-Minutes-Seconds (DMS) format into continuous Decimal Degrees for geospatial modelling. These were the standpoints through which TV Band Frequency measurements were taken.
"""
import os
import glob
import re
import pandas as pd

def parse_dms_robust(val):
    """
    Robust DMS to Decimal converter handling all variations of 
    degree, minute, second symbols and directional indicators.
    """
    if pd.isna(val):
        return None
    
    val_str = str(val).strip()
    
    # 1. Standardize non-standard Unicode quote characters
    # Replace single quote variants: ‘ ’ ` ' 
    val_str = re.sub(r"[‘’`']", "'", val_str)
    # Replace double quote variants: “ ” ″ "
    val_str = re.sub(r'[“”″"]', '"', val_str)
    # Replace degree symbol variants: ° º
    val_str = re.sub(r"[°º]", "°", val_str)

    # 2. Extract digits for Deg, Min, Sec, Direction using Regex
    pattern = r"(\d+)\s*°\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"\s*([NSEWnsew])"
    match = re.search(pattern, val_str)
    
    if match:
        deg, minutes, seconds, direction = match.groups()
        decimal = float(deg) + (float(minutes) / 60.0) + (float(seconds) / 3600.0)
        if direction.upper() in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 6)
    
    # 3. Fallback: If numbers are separated by spaces or punctuation without symbols
    fallback_pattern = r"(\d+)\D+(\d+)\D+(\d+(?:\.\d+)?)\D*([NSEWnsew])?"
    fb_match = re.search(fallback_pattern, val_str)
    if fb_match:
        deg, minutes, seconds, direction = fb_match.groups()
        decimal = float(deg) + (float(minutes) / 60.0) + (float(seconds) / 3600.0)
        if direction and direction.upper() in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 6)

    # Return original if already a valid float
    try:
        return float(val_str)
    except ValueError:
        return val

def process_and_convert_enugu_coordinates():
    input_dir = r"C:\Users\User\Documents\RFExplorer\Enugu_Project"
    output_dir = r"C:\Users\User\Documents\RFExplorer\Processed_Data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
    
    if not excel_files:
        print(f"[ERROR] No Excel file found in: {input_dir}")
        return

    input_file = excel_files[0]
    print(f"[INFO] Reading: {input_file}")

    df = pd.read_excel(input_file)
    
    # Strip whitespace from column names to prevent matching failures
    df.columns = [str(c).strip() for c in df.columns]

    # Find Latitude and Longitude columns regardless of case
    lat_col = [c for c in df.columns if 'lat' in c.lower()]
    lon_col = [c for c in df.columns if 'lon' in c.lower()]

    if lat_col:
        df[lat_col[0]] = df[lat_col[0]].apply(parse_dms_robust)
    if lon_col:
        df[lon_col[0]] = df[lon_col[0]].apply(parse_dms_robust)

    output_path = os.path.join(output_dir, "Enugu_Coordinates_Decimal.xlsx")
    
    # Save processed Excel sheet
    df.to_excel(output_path, index=False)
    
    print("\n" + "="*60)
    print("DECIMAL CONVERSION VERIFICATION OUTPUT:")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"File saved to: {output_path}\n")

# Run conversion
process_and_convert_enugu_coordinates()
