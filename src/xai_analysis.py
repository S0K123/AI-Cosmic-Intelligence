import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from train_model import get_trained_models, create_habitability_label
from data_loader import load_data
from feature_engineering import clean_and_prepare_data

def generate_shap_explanations(rf_model, dnn_model, X_test, feature_names):
    """
    Computes SHAP values for Random Forest and DNN to explain habitability predictions.
    """
    print("\n--- Generating Explainable AI (XAI) Analysis (SHAP) ---")
    
    # 1. SHAP for Random Forest
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values_rf = explainer_rf.shap_values(X_test)
    
    # SHAP Summary Plot for RF
    plt.figure(figsize=(10, 6))
    # Note: shap_values_rf is a list of [neg_class, pos_class] for binary
    shap.summary_plot(shap_values_rf[1], X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot: Random Forest')
    plt.savefig('plots/xai_shap_rf_summary.png')
    print("Saved SHAP summary plot for RF.")
    
    # 2. SHAP for DNN (using KernelExplainer or DeepExplainer)
    # Since dnn_model is BayesianDNN (torch), we can use KernelExplainer on a subset
    # Or DeepExplainer for torch
    import torch
    
    def dnn_predict_proba(X):
        dnn_model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            # Sigmoid for probability
            return torch.sigmoid(dnn_model(X_tensor)).numpy()
    
    # KernelExplainer is model-agnostic
    background = X_test[:50] # Subset for speed
    explainer_dnn = shap.KernelExplainer(dnn_predict_proba, background)
    shap_values_dnn = explainer_dnn.shap_values(X_test[:20]) # Small subset for speed
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_dnn, X_test[:20], feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance: Deep Neural Network')
    plt.savefig('plots/xai_shap_dnn_importance.png')
    print("Saved SHAP feature importance for DNN.")
    
    # 3. Planet-level explanation (Local Explanation)
    plt.figure(figsize=(12, 4))
    shap.force_plot(explainer_rf.expected_value[1], shap_values_rf[1][0,:], X_test[0,:], feature_names=feature_names, matplotlib=True, show=False)
    plt.title('Local SHAP Explanation: Candidate Planet')
    plt.savefig('plots/xai_local_explanation.png')
    print("Saved local SHAP explanation plot.")

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        rf, dnn, pinn, scaler, X_test, y_test = get_trained_models(labeled_df)
        features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
        
        generate_shap_explanations(rf, dnn, X_test, features)
