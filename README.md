# Spatial-Spectral Data Analysis Scripts (Enugu Metropolis)

## Project Summary
This repository contains the data processing, preparatory, and spatial-spectral analysis scripts developed for the Enugu Metropolis spectrum research study. The codebase is organized into four primary groups handling telemetry parsing, data cleaning, spatial-spectral analysis, and dataset generation.

## Zenodo Archival & DOI Information
* **Zenodo DOI:** https://doi.org/10.5281/zenodo.XXXXXXX *(Replace XXXXXXX with your actual DOI after release)*
* **Repository Version:** v1.0.0
* **Citation:** If using or referencing these scripts, please cite the Zenodo dataset release using the DOI provided above.

## Codebase Organization
* `Group_1_Preparatory_Scripts/`: This collection of scripts handles raw data ingestion, telemetry parsing, error filtering, and preprocessing to standardize the field measurement datasets for downstream model consumption.
* `Group_2_1Million_Spatial_Spectral_Data_Analysis_Scripts/`: This section contains the core analytical scripts dedicated to processing, spatially indexing, and extracting spectral features from the consolidated 1-million-entry radio frequency (TV Band) sweep dataset across the study region.
* `Group_3_Random_Forest_Classifier_Model_Analysis_Scripts/`: This suite of scripts details the dataset partitioning, hyperparameter tuning, feature importance evaluation, and performance evaluation metrics for the supervised Random Forest classification model.
* `Group_4_1M_Dataset_Processing/`: This group of scripts implements the reinforcement learning environment, agent policy training, reward function calculations, and performance evaluation pipelines for dynamic spectrum allocation and decision-making analysis.

## Dependencies & Requirements
To run these scripts, ensure you have the following software and Python libraries installed:

* **Python:** Version 3.8 or higher
* **Required Libraries:**
  * `pandas` (>= 1.3.0)
  * `numpy` (>= 1.21.0)
  * `pathlib` (Built-in standard library)
  * `shutil` (Built-in standard library)
  