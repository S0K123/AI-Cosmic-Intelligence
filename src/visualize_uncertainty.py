import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_uncertainty_results(df):
    """Generates and saves visualizations for habitability prediction uncertainty."""
    if df.empty:
        print("DataFrame is empty, cannot generate uncertainty plots.")
        return

    print("\n--- Generating Uncertainty Visualizations ---")

    # 1. Histogram of Prediction Uncertainties
    plt.figure(figsize=(10, 6))
    sns.histplot(df['prediction_uncertainty'], bins=20, kde=True)
    plt.title('Distribution of Prediction Uncertainties')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.savefig('plots/uncertainty_distribution.png')
    print("Saved uncertainty distribution plot.")

    # 2. Scatter Plot of Probability vs. Uncertainty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='habitability_probability', y='prediction_uncertainty', hue='confidence_score', palette='viridis', size='confidence_score', sizes=(20, 200))
    plt.title('Habitability Probability vs. Prediction Uncertainty')
    plt.xlabel('Mean Habitability Probability')
    plt.ylabel('Prediction Uncertainty')
    plt.axvline(0.6, color='r', linestyle='--', label='Habitability Threshold')
    plt.axhline(0.15, color='orange', linestyle='--', label='Uncertainty Threshold')
    plt.legend()
    plt.savefig('plots/probability_vs_uncertainty.png')
    print("Saved probability vs. uncertainty scatter plot.")

if __name__ == "__main__":
    # This is for standalone testing. The main call will be from planet_discovery.py
    try:
        results_df = pd.read_csv('results/habitability_with_uncertainty.csv')
        visualize_uncertainty_results(results_df)
    except FileNotFoundError:
        print("Could not find results/habitability_with_uncertainty.csv. Run planet_discovery.py first.")
