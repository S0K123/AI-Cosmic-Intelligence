import pandas as pd
import numpy as np
import torch

from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from train_model import create_habitability_label, get_trained_models, enable_dropout
from gnn_model import get_trained_gnn, create_graph_dataset
from space_simulator import CelestialBody, simulate_step
from visualize_uncertainty import visualize_uncertainty_results

# --- Step 2 & 3: AI Habitability Filtering with Uncertainty ---
def filter_with_ai_ensemble(candidates, rf_model, dnn_model, gnn_model, scaler, original_df):
    if candidates.empty:
        return pd.DataFrame()

    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    X_cand_scaled = scaler.transform(candidates[features])
    X_cand_tensor = torch.from_numpy(X_cand_scaled).float()

    # RF predictions
    prob_rf = rf_model.predict_proba(X_cand_scaled)[:, 1]

    # Bayesian DNN predictions (Monte Carlo Dropout)
    dnn_model.eval()
    enable_dropout(dnn_model)
    mc_predictions = []
    with torch.no_grad():
        for _ in range(100):
            preds = torch.sigmoid(dnn_model(X_cand_tensor)).numpy().flatten()
            mc_predictions.append(preds)
    
    mc_predictions = np.array(mc_predictions)
    mean_prob_dnn = mc_predictions.mean(axis=0)
    uncertainty_dnn = mc_predictions.std(axis=0)

    # GNN predictions
    prob_gnn = []
    for i, cand in candidates.iterrows():
        temp_df = pd.concat([original_df, cand.to_frame().T], ignore_index=True)
        temp_graph = create_graph_dataset(temp_df)[0]
        with torch.no_grad():
            out = gnn_model(temp_graph)
            prob = torch.sigmoid(out[-1]).item()
            prob_gnn.append(prob)

    # Ensemble score
    final_score = (prob_rf + mean_prob_dnn + np.array(prob_gnn)) / 3
    candidates['habitability_probability'] = final_score
    candidates['prediction_uncertainty'] = uncertainty_dnn
    candidates['confidence_score'] = final_score * (1 - uncertainty_dnn)

    # Filter based on new criteria
    accepted_candidates = candidates[
        (candidates['habitability_probability'] > 0.6) & 
        (candidates['prediction_uncertainty'] < 0.15)
    ].copy()
    
    return accepted_candidates

if __name__ == "__main__":
    # ... (rest of the script is the same until the end)
    
    # 6. Final results
    final_results_df = pd.DataFrame(stable_planets)
    print(f"\n--- AI Planet Discovery Complete ---")
    print(f"Found {len(final_results_df)} stable and potentially habitable new planets.")
    
    if not final_results_df.empty:
        print(final_results_df)
        final_results_df.to_csv('results/habitability_with_uncertainty.csv', index=False)
        print("\nResults saved to results/habitability_with_uncertainty.csv")
        
        # 7. Visualize uncertainty results
        visualize_uncertainty_results(final_results_df)
