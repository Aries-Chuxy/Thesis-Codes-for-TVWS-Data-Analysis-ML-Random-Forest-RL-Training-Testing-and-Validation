"""
Descriptive Note: This Python script generates and visualizes a spatial-temporal spectrum occupancy heatmap for 20 locations across Enugu Metropolis using a simulated dataset of 1,000,000 records. 
It optimizes memory consumption and execution speed through vectorized Pandas operations (pd.cut, categorical data types, downcasted integers), bins timestamps into four diurnal sessions (Morning, Afternoon, Evening, Night), aggregates occupancy rates via pivot table, and outputs a publication-quality Seaborn heatmap (Enugu_Metropolis_Occupancy_Heatmap.png).
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. OPTIMIZED DATA SIMULATION / LOADING
# -------------------------------------------------------------------------
# Assuming 'Processed_Data' contains columns: 'Location', 'Timestamp', and 'Is_Occupied'
# To be memory friendly, we force categories and downcast numeric values.

# If reading from a CSV, use these parameters to minimize RAM usage:
# df = pd.read_csv('Processed_Data.csv', dtype={'Location': 'category', 'Is_Occupied': 'int8'}, parse_dates=['Timestamp'])

# Generating dummy data to match your 1,000,000 dataset layout
np.random.seed(42)
n_rows = 1_000_000

locations = [
    "Abakpa", "GRA", "IndependenceLayout", "AcharaLayout", "Amaechi", 
    "AwkunanawGarriki", "ChinaTown", "Emene", "IbagwaNike", "IvaValley", 
    "MaryLand", "NewHaven", "OgbeteKingsway", "OguiAsata", "ThinkerCorner", 
    "TransEkulu", "Ugwuaji", "UmuChigbo", "Uwani", "CoalCamp"
]

data = {
    'Location': pd.Categorical(np.random.choice(locations, size=n_rows)),
    'Timestamp': pd.date_range(start='2026-01-01', periods=n_rows, freq='5s'),
    'Is_Occupied': np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4]).astype(np.int8)
}
df = pd.DataFrame(data)

# -------------------------------------------------------------------------
# 2. FAST VECTORIZED DIURNAL BINNING (Memory & Speed Optimized)
# -------------------------------------------------------------------------
# Extract hour natively via vectorized int extraction (much faster than string conversion)
hours = df['Timestamp'].dt.hour

# Define boundaries and custom labels according to your specification:
# Morning: 6am-12pm; Afternoon: 12pm-4pm; Evening: 4pm-10pm; Night: 10pm-6am
bins = [-1, 5, 11, 15, 21, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Night']

# Vectorized categorization using pd.cut
# ordered=False allows merging categories (like the two 'Night' slices) instantly
diurnal_series = pd.cut(hours, bins=bins, labels=labels, ordered=False).astype('category')

# Insert back into dataframe efficiently
df['Diurnal_Session'] = diurnal_series

# -------------------------------------------------------------------------
# 3. FAST SPATIAL-TEMPORAL AGGREGATION
# -------------------------------------------------------------------------
# Pivot tables built on categorized columns complete instantly and consume negligible memory
occupancy_matrix = df.pivot_table(
    index='Location', 
    columns='Diurnal_Session', 
    values='Is_Occupied', 
    aggfunc='mean'
)

# Reorder columns to sequential chronological order
session_order = ['Morning', 'Afternoon', 'Evening', 'Night']
occupancy_matrix = occupancy_matrix.reindex(columns=session_order)

# -------------------------------------------------------------------------
# 4. PLOT HIGH-RESOLUTION COMPLIANCE HEATMAP
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 8), dpi=150)
sns.heatmap(
    occupancy_matrix, 
    annot=True, 
    fmt=".2%", 
    cmap="YlOrRd", 
    linewidths=0.5, 
    cbar_kws={'label': 'Spectrum Occupancy Rate / Compliance'}
)

plt.title('General Enugu Metropolis Spectrum Occupancy Matrix\nSpatial Variations vs. Temporal Diurnal Sessions', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Measured Locations (Enugu Metropolis)', fontsize=10)
plt.xlabel('Diurnal Sessions', fontsize=10)
plt.tight_layout()

# Save image efficiently
plt.savefig('Enugu_Metropolis_Occupancy_Heatmap.png', dpi=300)
plt.show()
