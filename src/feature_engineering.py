import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data

def clean_and_prepare_data(df):
    """
    This function takes the raw exoplanet data and performs the following steps:
    1. Selects key features for habitability analysis.
    2. Renames columns for clarity.
    3. Handles missing values.
    4. Calculates stellar luminosity.
    5. Calculates stellar luminosity.
    It returns a cleaned but unscaled DataFrame.
    """
    # 1. Select and rename columns
    feature_map = {
        'pl_name': 'planet_name',
        'pl_rade': 'planet_radius',      # Planet radius in Earth radii
        'pl_orbper': 'orbital_period',   # Orbital period in days
        'st_teff': 'star_temp_k',        # Stellar effective temperature in Kelvin
        'pl_orbsmax': 'orbit_semi_major_axis', # Orbit semi-major axis in AU
        'st_rad': 'star_radius_solar', # Stellar radius in Solar radii
        'st_lum': 'star_luminosity' # Stellar luminosity in Solar luminosity
    }
    
    # Filter for columns that exist in the dataframe
    existing_features = [col for col in feature_map.keys() if col in df.columns]
    df_clean = df[existing_features].copy()
    df_clean = df_clean.rename(columns=feature_map)
    
    print("1. Selected and renamed columns:")
    print(df_clean.head())
    
    # 2. Handle missing values
    # For this example, we will use median imputation for simplicity.
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
            
    print("\n2. Handled missing values (using median imputation):")
    print(df_clean.isnull().sum())
    
    # 3. Calculate Stellar Luminosity (if not present)
    # Using the Stefan-Boltzmann law: L = 4 * pi * R^2 * sigma * T^4
    # We will express it in terms of solar units.
    if 'star_luminosity' not in df_clean.columns:
        # L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
        # T_sun is approx. 5778 K
        df_clean['star_luminosity'] = (df_clean['star_radius_solar']**2) * ((df_clean['star_temp_k'] / 5778)**4)
        print("\n3. Calculated stellar luminosity.")

    print("\nData cleaned and prepared (unscaled).")
    
    return df_clean

if __name__ == "__main__":
    # Load the raw data
    raw_df = load_data()
    
    if raw_df is not None:
        print("--- Starting Data Cleaning and Preparation ---")
        
        # Filter for default_flag == 1 to get the best measurement for each planet
        print(f"Original dataset shape: {raw_df.shape}")
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
            print(f"Dataset shape after filtering for default_flag == 1: {raw_df.shape}")
        else:
            print("Warning: 'default_flag' column not found. Skipping filtering.")

        # Clean and prepare the data
        prepared_df = clean_and_prepare_data(raw_df)
        
        print("\n--- Final Cleaned and Prepared Dataset ---")
        print(prepared_df.head())
    else:
        print("Could not load data to start the cleaning process.")
