import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from train_model import create_habitability_label


def plot_dataset_exploration(df):
    """Part 1: Plot distributions of key features."""
    print("\n--- Part 1: Dataset Exploration Visualizations ---")
    
    plt.figure(figsize=(18, 5))
    
    # Planet Radius Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df['planet_radius'], bins=50, kde=True)
    plt.title('Distribution of Planet Radius (Earth Radii)')
    plt.xlabel('Planet Radius')
    plt.ylabel('Frequency')
    plt.xscale('log')

    # Orbital Period Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df['orbital_period'], bins=50, kde=True)
    plt.title('Distribution of Orbital Periods (Days)')
    plt.xlabel('Orbital Period')
    plt.xscale('log')

    # Stellar Temperature Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(df['star_temp_k'], bins=50, kde=True)
    plt.title('Distribution of Stellar Temperatures (K)')
    plt.xlabel('Stellar Temperature')
    
    plt.tight_layout()
    plt.savefig('plots/part1_dataset_exploration.png')
    print("Saved dataset exploration plots.")

def plot_habitability_comparison(df):
    """Part 2: Compare habitable vs. non-habitable planets."""
    print("\n--- Part 2: Habitability Comparison ---")
    
    plt.figure(figsize=(14, 6))
    
    # Bar chart of habitability labels
    plt.subplot(1, 2, 1)
    sns.countplot(x='habitability_label', data=df)
    plt.title('Habitability Label Distribution')
    plt.xlabel('Is Potentially Habitable?')
    plt.ylabel('Count')

    # Scatter plot of habitable zone
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='star_luminosity', y='orbit_semi_major_axis', hue='habitability_label', palette={0: 'gray', 1: 'cyan'}, alpha=0.7)
    plt.title('Habitable Zone Visualization')
    plt.xlabel('Stellar Luminosity (Solar Units)')
    plt.ylabel('Orbit Semi-Major Axis (AU)')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/part2_habitability_comparison.png')
    print("Saved habitability comparison plots.")

def plot_feature_relationships(df):
    """Part 3: Visualize relationships between key features."""
    print("\n--- Part 3: Feature Relationships ---")
    
    plt.figure(figsize=(18, 5))
    
    # planet_radius vs orbital_period
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='orbital_period', y='planet_radius', hue='habitability_label', palette={0: 'gray', 1: 'cyan'}, alpha=0.7)
    plt.title('Planet Radius vs. Orbital Period')
    plt.xlabel('Orbital Period (Days)')
    plt.ylabel('Planet Radius (Earth Radii)')
    plt.xscale('log')
    plt.yscale('log')

    # planet_radius vs star_temp_k
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='star_temp_k', y='planet_radius', hue='habitability_label', palette={0: 'gray', 1: 'cyan'}, alpha=0.7)
    plt.title('Planet Radius vs. Star Temperature')
    plt.xlabel('Star Temperature (K)')
    plt.ylabel('Planet Radius (Earth Radii)')
    plt.yscale('log')

    # orbit_semi_major_axis vs star_luminosity
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='star_luminosity', y='orbit_semi_major_axis', hue='habitability_label', palette={0: 'gray', 1: 'cyan'}, alpha=0.7)
    plt.title('Orbit vs. Star Luminosity')
    plt.xlabel('Stellar Luminosity (Solar Units)')
    plt.ylabel('Orbit Semi-Major Axis (AU)')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/part3_feature_relationships.png')
    print("Saved feature relationship plots.")

def plot_model_insights(df):
    """Part 4: Plot feature importance from the model."""
    print("\n--- Part 4: AI Model Insights ---")
    
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    X = df[features]
    y = df['habitability_label']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance for Habitability Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig('plots/part4_feature_importance.png')
    print("Saved feature importance plot.")

def plot_3d_star_system(df):
    """Part 5: Create a 3D visualization of a planetary system."""
    print("\n--- Part 5: 3D Star System Visualization ---")
    
    # Select a system, e.g., TRAPPIST-1
    system_name = 'TRAPPIST-1'
    system_df = df[df['planet_name'].str.contains(system_name, na=False)]
    
    if system_df.empty:
        print(f"System '{system_name}' not found in the dataset. Skipping 3D plot.")
        return

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Star at the center
    ax.scatter(0, 0, 0, s=200, c='yellow', marker='*', label='Star (TRAPPIST-1)')

    # Get habitable zone for this star
    star_lum = system_df['star_luminosity'].iloc[0]
    inner_hz = np.sqrt(star_lum / 1.1)
    outer_hz = np.sqrt(star_lum / 0.53)

    # Plot orbits and planets
    for i, row in system_df.iterrows():
        r = row['orbit_semi_major_axis']
        theta = np.linspace(0, 2 * np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)
        ax.plot(x, y, z, linestyle='--', color='gray', alpha=0.5)
        
        # Plot planet position
        planet_color = 'cyan' if row['habitability_label'] == 1 else 'blue'
        ax.scatter(r, 0, 0, s=50, c=planet_color, label=row['planet_name'])

    # Plot habitable zone
    theta = np.linspace(0, 2 * np.pi, 100)
    x_inner = inner_hz * np.cos(theta)
    y_inner = inner_hz * np.sin(theta)
    ax.plot(x_inner, y_inner, 0, color='green', linestyle='-', alpha=0.3, label='Habitable Zone')
    x_outer = outer_hz * np.cos(theta)
    y_outer = outer_hz * np.sin(theta)
    ax.plot(x_outer, y_outer, 0, color='green', linestyle='-', alpha=0.3)
    
    ax.set_title(f'3D Visualization of the {system_name} System')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.legend()
    
    plt.savefig('plots/part5_3d_star_system.png')
    print(f"Saved 3D visualization for {system_name}.")

if __name__ == "__main__":
    # Load and prepare data
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()

        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        # Generate all plots
        plot_dataset_exploration(labeled_df)
        plot_habitability_comparison(labeled_df)
        plot_feature_relationships(labeled_df)
        plot_model_insights(labeled_df)
        plot_3d_star_system(labeled_df)
        
        print("\nAll visualizations have been generated and saved to the 'plots/' directory.")
