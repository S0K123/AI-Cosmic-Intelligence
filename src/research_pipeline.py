import pandas as pd
import numpy as np
import torch
from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from train_model import create_habitability_label, get_trained_models, enable_dropout
from gnn_model import get_trained_gnn
from self_supervised import train_self_supervised, get_ssl_embeddings
from generative_models import train_vae, generate_synthetic_system
from rl_discovery import train_rl_agent, PlanetDiscoveryEnv
from xai_analysis import generate_shap_explanations
from active_learning import active_learning_iteration
from planet_discovery import is_orbit_stable, filter_with_ai_ensemble
from visualize_uncertainty import visualize_uncertainty_results
import os

def run_advanced_research_pipeline():
    print("\n--- Starting AI Cosmic Intelligence Engine: Advanced Research Pipeline ---")
    
    # Create directories
    for d in ['results', 'plots', 'simulations']:
        if not os.path.exists(d):
            os.makedirs(d)

    # 1. Load Data
    raw_df = load_data()
    if raw_df is None: return
    if 'default_flag' in raw_df.columns:
        raw_df = raw_df[raw_df['default_flag'] == 1].copy()
    
    # 2. Self-Supervised Learning (SSL) Pretraining
    prepared_df = clean_and_prepare_data(raw_df)
    ssl_model, ssl_scaler = train_self_supervised(prepared_df)
    ssl_embeddings = get_ssl_embeddings(ssl_model, ssl_scaler, prepared_df)
    
    # 3. Physics Feature Engineering & Labeling
    labeled_df = create_habitability_label(prepared_df)
    
    # 4. Train Models: RF, Bayesian DNN, PINN, GNN
    rf, dnn, pinn, scaler, X_test, y_test = get_trained_models(labeled_df)
    gnn = get_trained_gnn(labeled_df)
    
    # 5. Generative Model: VAE
    vae_model, vae_scaler = train_vae(labeled_df)
    synthetic_system = generate_synthetic_system(vae_model, vae_scaler, num_planets=5)
    print("\nGenerated Synthetic Planetary System (VAE):")
    print(synthetic_system.head())
    
    # 6. Reinforcement Learning (RL) Planet Discovery
    rl_agent = train_rl_agent(labeled_df, rf, dnn, pinn, scaler)
    
    # 7. Active Learning Loop (Iteration 1)
    updated_df = active_learning_iteration(labeled_df, rf, dnn, pinn, scaler)
    # Retrain (simplified)
    rf, dnn, pinn, scaler, X_test, y_test = get_trained_models(updated_df)
    
    # 8. Explainable AI (XAI) Analysis
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    generate_shap_explanations(rf, dnn, X_test, features)
    
    # 9. Discovery & Stability Testing (Integration)
    # Use TRAPPIST-1 as an example system
    system_name = 'TRAPPIST-1'
    star_system_df = labeled_df[labeled_df['hostname'].str.contains(system_name, na=False)].copy()
    
    print(f"\n--- Testing RL Discovery for {system_name} ---")
    env = PlanetDiscoveryEnv(star_system_df.iloc[0].to_dict(), {'rf': rf, 'dnn': dnn, 'pinn': pinn}, scaler)
    obs, _ = env.reset()
    action, _ = rl_agent.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    candidate = pd.DataFrame([info['planet']])
    
    # Filter with AI Ensemble (RF + Bayesian DNN + GNN)
    # (Simplified for demonstration)
    print(f"Proposed Candidate Planet: {candidate}")
    
    is_stable = is_orbit_stable(star_system_df, candidate.iloc[0])
    print(f"Orbital Stability Result: {'STABLE' if is_stable else 'UNSTABLE'}")
    
    # 10. Visualization
    # (Assuming visualize_uncertainty handles results/habitability_with_uncertainty.csv)
    # We'll save a mock result file to demonstrate visualization
    candidate['habitability_probability'] = 0.95
    candidate['prediction_uncertainty'] = 0.05
    candidate['confidence_score'] = 0.95 * (1 - 0.05)
    candidate.to_csv('results/habitability_with_uncertainty.csv', index=False)
    visualize_uncertainty_results(candidate)

    print("\nAdvanced Research Pipeline complete. Results saved in results/, plots/, and simulations/.")

if __name__ == "__main__":
    run_advanced_research_pipeline()
