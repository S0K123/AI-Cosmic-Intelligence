import pandas as pd
import requests
import os

def download_exoplanet_data(filename='data/PS_exoplanets.csv'):
    """
    Downloads the NASA Exoplanet Archive dataset (Planetary Systems table)
    """
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Query for the Planetary Systems table (ps) which is often faster for basic queries
    # Selecting a few key columns for the initial setup
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+100+pl_name,hostname,pl_letter,discoverymethod,pl_orbper,pl_rade,pl_masse,st_teff,st_rad,st_mass+from+ps&format=csv"
    
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        print(f"Data received. Size: {len(response.content)} bytes")
        with open(filename, 'wb') as f:
            f.write(response.content)
            
        print(f"Dataset saved to {filename}")
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def load_data(filename='data/PS_exoplanets.csv'):
    """
    Loads the exoplanet dataset into a Pandas DataFrame.
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found. Attempting download...")
        if not download_exoplanet_data(filename):
            return None
            
    # Load dataset, skipping initial comment lines if any (NASA CSVs often have '#' headers)
    # The pscomppars table usually doesn't have '#' if requested via TAP format=csv
    df = pd.read_csv(filename, comment='#')
    return df

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Successfully loaded {len(df)} records.")
        print("\nColumns in the dataset:")
        print(df.columns.tolist())
    else:
        print("Failed to load dataset.")
