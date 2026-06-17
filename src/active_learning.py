import pandas as pd
import numpy as np
import torch
from train_model import get_trained_models, create_habitability_label, enable_dropout
from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from space_simulator import CelestialBody, simulate_step

def active_learning_iteration(df, rf_model, dnn_model, pinn_model, scaler, num_samples=10):
    """
    Implements one iteration of active learning.
    1. Identify high uncertainty planets.
    2. Simulate for 'ground truth'.
    3. Add to training set.
    """
    print("\n--- Running Active Learning Loop ---")
    
    # Identify high uncertainty planets using Bayesian DNN (MC Dropout)
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    X_all = scaler.transform(df[features])
    X_tensor = torch.from_numpy(X_all).float()
    
    dnn_model.eval()
    enable_dropout(dnn_model)
    mc_preds = []
    with torch.no_grad():
        for _ in range(50):
            preds = torch.sigmoid(dnn_model(X_tensor)).numpy().flatten()
            mc_preds.append(preds)
    
    uncertainties = np.std(np.array(mc_preds), axis=0)
    
    # Get top N most uncertain samples
    top_uncertain_idx = np.argsort(uncertainties)[-num_samples:]
    uncertain_samples = df.iloc[top_uncertain_idx].copy()
    
    print(f"Identified {num_samples} most uncertain planets for simulation.")
    
    # 2. Simulate these planets to get 'ground truth' stability
    simulation_results = []
    for _, planet in uncertain_samples.iterrows():
        is_stable = simulate_stability_test(planet)
        simulation_results.append(1 if is_stable else 0)
        
    # 3. Add simulation results to original dataset for retraining
    # For demonstration, we'll just update labels
    uncertain_samples['habitability_label'] = simulation_results
    
    updated_df = pd.concat([df, uncertain_samples], ignore_index=True)
    
    print(f"Active learning: Added {num_samples} validated samples to the dataset.")
    return updated_df

def simulate_stability_test(planet_row):
    """
    Quick stability test using existing physics engine.
    """
    bodies = []
    star_mass = planet_row.get('st_mass', 1.0)
    bodies.append(CelestialBody('Star', star_mass, [0, 0], [0, 0]))
    
    r = planet_row['orbit_semi_major_axis']
    v = np.sqrt(0.0002959122 * star_mass / r)
    bodies.append(CelestialBody('Planet', 1e-6, [r, 0], [0, v]))
    
    for _ in range(1000):
        simulate_step(bodies, 1.0)
        dist = np.linalg.norm(bodies[1].pos)
        if dist > 10 * r or dist < 0.1 * r:
            return False
    return True

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        rf, dnn, pinn, scaler, _, _ = get_trained_models(labeled_df)
        
        # Run one iteration of active learning
        updated_df = active_learning_iteration(labeled_df, rf, dnn, pinn, scaler)
        
        # Retrain models with updated data
        rf_new, dnn_new, pinn_new, _, _, _ = get_trained_models(updated_df)
        print("\nModels retrained after active learning iteration.")
        
        # Save updated dataset
        updated_df.to_csv('results/active_learning_dataset.csv', index=False)
