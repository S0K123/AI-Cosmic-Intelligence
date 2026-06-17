from data_loader import load_data

def main():
    print("AI Cosmic Intelligence Engine - Data Exploration\n")
    
    # Load dataset
    df = load_data()
    
    if df is not None:
        print(f"Dataset Shape: {df.shape}")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        print("\nColumns in the dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
            
        # Display some interesting statistics
        print("\nBasic Dataset Information:")
        print(df.info())
        
        # Count non-null values for key columns
        # pl_name (Planet Name), pl_orbper (Orbital Period), pl_rade (Planet Radius in Earth Radii), pl_masse (Planet Mass in Earth Masses)
        # st_teff (Stellar Effective Temperature), st_rad (Stellar Radius), st_mass (Stellar Mass)
        key_cols = ['pl_name', 'pl_orbper', 'pl_rade', 'pl_masse', 'st_teff', 'st_rad', 'st_mass']
        existing_key_cols = [c for c in key_cols if c in df.columns]
        
        if existing_key_cols:
            print("\nStatistics for key astrophysical parameters:")
            print(df[existing_key_cols].describe())
    else:
        print("Dataset could not be loaded. Please check your internet connection or data folder.")

if __name__ == "__main__":
    main()
