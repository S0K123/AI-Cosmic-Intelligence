
"""
Complete AI pipeline with model comparison, habitability ranking, and results export!
"""
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import enable_dropout


def compute_uncertainty_estimation(model, X, n_samples=100):
    if model is None:
        return np.zeros(len(X)), np.zeros(len(X))
    model.eval()
    enable_dropout(model)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_tensor)
            if logits.shape[-1] > 1:
                prob = F.softmax(logits, dim=1)[:, 1].numpy()
            else:
                prob = torch.sigmoid(logits).numpy().flatten()
            predictions.append(prob)
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    return mean_pred, std_pred


def predict_with_all_models(rf, dnn, pinn, scaler, X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # RF
    rf_pred = np.zeros(len(X_scaled))
    rf_prob = np.zeros(len(X_scaled))
    if rf:
        rf_pred = rf.predict(X_scaled)
        if hasattr(rf, 'predict_proba'):
            rf_prob = rf.predict_proba(X_scaled)[:, 1]
    
    # DNN/PINN
    dnn_pred = np.zeros(len(X_scaled))
    dnn_prob = np.zeros(len(X_scaled))
    dnn_std = np.zeros(len(X_scaled))
    if dnn is not None:
        dnn_mean, dnn_std = compute_uncertainty_estimation(dnn, X_scaled)
        dnn_prob = dnn_mean
        dnn_pred = (dnn_prob > 0.5).astype(int)
        
    pinn_pred = np.zeros(len(X_scaled))
    pinn_prob = np.zeros(len(X_scaled))
    if pinn is not None:
        X_t = torch.from_numpy(X_scaled).float()
        pinn.eval()
        with torch.no_grad():
            h_p, _ = pinn(X_t)
            pinn_prob = torch.sigmoid(h_p).numpy().flatten()
        pinn_pred = (pinn_prob > 0.5).astype(int)
        
    # Weighted average score
    dnn_weight = np.clip(1 - (dnn_std*2), 0, 1)
    weights = np.array([0.35, 0.35, 0.3])
    total_w = np.ones((len(X_scaled), 3)) * weights
    total_w[:,1] *= dnn_weight
    total_w = total_w / total_w.sum(axis=1, keepdims=True)
    overall = np.hstack([
        rf_prob.reshape(-1, 1),
        dnn_prob.reshape(-1, 1),
        pinn_prob.reshape(-1,1)
    ])
    overall_score = (overall * total_w).sum(axis=1)
    confidence = 1 - dnn_std

    return {
        "rf_prediction": rf_pred,
        "rf_confidence": rf_prob,
        "dnn_prediction": dnn_pred,
        "dnn_confidence": dnn_prob,
        "dnn_uncertainty": dnn_std,
        "pinn_prediction": pinn_pred,
        "pinn_confidence": pinn_prob,
        "overall_habitability_score": overall_score,
        "prediction_confidence": confidence
    }


def get_model_comparison_metrics(rf, dnn, X_test, y_test):
    metrics = {}
    if rf is not None:
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        metrics['Random Forest'] = {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred, zero_division=0),
            "Recall": recall_score(y_test, rf_pred),
            "F1-Score": f1_score(y_test, rf_pred),
            "ROC-AUC": roc_auc_score(y_test, rf_prob)
        }
        
    if dnn is not None:
        X_t = torch.from_numpy(X_test).float()
        dnn.eval()
        with torch.no_grad():
            dnn_logits = dnn(X_t)
            dnn_prob = torch.sigmoid(dnn_logits).numpy().flatten()
        dnn_pred = (dnn_prob > 0.5).astype(int)
        metrics['Bayesian DNN'] = {
            "Accuracy": accuracy_score(y_test, dnn_pred),
            "Precision": precision_score(y_test, dnn_pred, zero_division=0),
            "Recall": recall_score(y_test, dnn_pred),
            "F1-Score": f1_score(y_test, dnn_pred),
            "ROC-AUC": roc_auc_score(y_test, dnn_prob)
        }
        
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index').round(3)
    rf_p = metrics['Random Forest'].get('ROC-AUC', 0) if rf else 0
    return df_metrics, rf_prob, dnn_prob


def rank_planets_by_habitability(df, results):
    df = df.copy()
    for key in ['rf_prediction', 'rf_confidence', 'dnn_prediction', 'dnn_confidence',
               'dnn_uncertainty', 'pinn_prediction', 'pinn_confidence',
               'overall_habitability_score', 'prediction_confidence']:
        df[key] = results[key]
    df['habitability_rank'] = df['overall_habitability_score'].rank(ascending=False).astype(int)
    return df.sort_values('overall_habitability_score', ascending=False)


def export_ranked_results(df_ranked, output_path="results/ranked_planets.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ranked.to_csv(output_path, index=False)
    return output_path

