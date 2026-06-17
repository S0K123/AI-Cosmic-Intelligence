import pandas as pd
import numpy as np
import os
import requests

def download_exoplanet_data(filename='data/PS_exoplanets.csv'):
    """
    Downloads the NASA Exoplanet Archive dataset (Planetary Systems table)
    """
    if not os.path.exists('data'):
        os.makedirs('data')
        
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+1000+pl_name,hostname,pl_rade,pl_orbper,st_teff,pl_orbsmax,st_rad,st_lum,st_mass+from+ps&format=csv"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return False

def load_and_clean_data(filename='data/PS_exoplanets.csv'):
    """
    Loads and cleans the NASA Exoplanet dataset.
    """
    if not os.path.exists(filename):
        if not download_exoplanet_data(filename):
            return None
            
    df = pd.read_csv(filename, comment='#')
    
    # Feature mapping
    feature_map = {
        'pl_name': 'planet_name',
        'pl_rade': 'planet_radius',
        'pl_orbper': 'orbital_period',
        'st_teff': 'star_temp_k',
        'pl_orbsmax': 'orbit_semi_major_axis',
        'st_rad': 'star_radius_solar',
        'st_lum': 'star_luminosity',
        'st_mass': 'star_mass_solar'
    }
    
    df = df.rename(columns=feature_map)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Calculate luminosity if missing
    if 'star_luminosity' not in df.columns or df['star_luminosity'].isnull().all():
        df['star_luminosity'] = (df['star_radius_solar']**2) * ((df['star_temp_k'] / 5778)**4)
        
    # Create habitability label
    # Flux = L / r^2
    df['stellar_flux'] = df['star_luminosity'] / (df['orbit_semi_major_axis']**2)
    df['habitability_label'] = (
        (df['planet_radius'] >= 0.5) & (df['planet_radius'] <= 2.0) & 
        (df['stellar_flux'] >= 0.2) & (df['stellar_flux'] <= 2.2)
    ).astype(int)
    
    return df
