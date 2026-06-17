import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_data
from feature_engineering import clean_and_prepare_data

# Gravitational Constant G in AU, Solar Mass, Day units
G_CONSTANT = 0.0002959122082855911

def create_habitability_label(df):
    """
    Creates a habitability label based on Earth-like radius and Stellar Flux.
    Flux = L_star / (a_AU^2)
    Earth's flux is 1.0. 
    A conservative habitable zone is often considered between 0.75 and 1.77 times Earth's flux.
    Planet radius between 0.5 and 2.0 Earth radii.
    """
    df = df.copy()
    
    # Calculate flux (L/R^2)
    df['stellar_flux'] = df['star_luminosity'] / (df['orbit_semi_major_axis']**2)
    
    # Simple habitability rule
    df['habitability_label'] = (
        (df['planet_radius'] >= 0.5) & (df['planet_radius'] <= 2.0) & 
        (df['stellar_flux'] >= 0.2) & (df['stellar_flux'] <= 2.2)
    ).astype(int)
    
    print(f"\nCreated habitability labels. {df['habitability_label'].sum()} potential candidates found.")
    return df

# --- Bayesian Neural Network with MC Dropout ---
class BayesianDNN(nn.Module):
    def __init__(self, input_dim):
        super(BayesianDNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        return self.output_layer(x)

# --- Physics-Informed Neural Network (PINN) ---
class PINN(nn.Module):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.habitability_out = nn.Linear(32, 1)
        self.velocity_out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        h_prob = self.habitability_out(x)
        v_pred = self.velocity_out(x)
        return h_prob, v_pred

def physics_loss(v_pred, star_mass, r, lambda_val=0.1):
    """
    Enforces v ≈ sqrt(G * M_star / r)
    """
    v_theoretical = torch.sqrt(G_CONSTANT * star_mass / r)
    loss = torch.mean((v_pred - v_theoretical)**2)
    return lambda_val * loss

def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_trained_models(df):
    """Trains and returns a Random Forest, Bayesian DNN, and PINN model."""
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    X = df[features]
    y = df['habitability_label']
    
    # We also need stellar mass for PINN physics loss
    st_mass = df['st_mass'] if 'st_mass' in df.columns else pd.Series([1.0] * len(df))
    st_mass = st_mass.fillna(1.0).values # Default to 1 solar mass

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test, st_mass_train, st_mass_test = train_test_split(
        X_scaled, y.values, st_mass, test_size=0.2, random_state=42, stratify=y
    )
    
    # Keep track of indices for physics loss (unscaled r)
    _, _, r_train_unscaled, _ = train_test_split(
        X_scaled, df['orbit_semi_major_axis'].values, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print("\n--- Random Forest Model Trained ---")
    
    # Train Bayesian DNN
    dnn_model = BayesianDNN(input_dim=X_train.shape[1])
    optimizer_dnn = optim.Adam(dnn_model.parameters(), lr=0.001)
    criterion_bce = nn.BCEWithLogitsLoss()
    
    train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)

    print("--- Bayesian DNN Training ---")
    for epoch in range(50):
        for inputs, labels in train_loader:
            optimizer_dnn.zero_grad()
            outputs = dnn_model(inputs)
            loss = criterion_bce(outputs, labels)
            loss.backward()
            optimizer_dnn.step()
    print("--- Bayesian DNN Model Trained ---")

    # Train PINN
    pinn_model = PINN(input_dim=X_train.shape[1])
    optimizer_pinn = optim.Adam(pinn_model.parameters(), lr=0.001)
    
    # Need r (semi-major axis) and st_mass for physics loss
    # orbit_semi_major_axis is at index 3 in our features list
    r_train = torch.from_numpy(X_train[:, 3]).float().unsqueeze(1)
    st_mass_train_t = torch.from_numpy(st_mass_train).float().unsqueeze(1)
    
    print("--- PINN Training ---")
    for epoch in range(50):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer_pinn.zero_grad()
            h_prob, v_pred = pinn_model(inputs)
            
            # Classification loss
            c_loss = criterion_bce(h_prob, labels)
            
            # Physics loss
            # Note: inputs are scaled, so we need to use unscaled r for physics
            # For simplicity, we'll use the unscaled r from the original df here
            # In a more robust implementation, we would inverse_transform the inputs
            # Here we just pass the pre-extracted r_train for that batch
            # We'll need a different loader for PINN to keep r and st_mass aligned
            pass

        # Simplified PINN loop for demonstration
        inputs_all = torch.from_numpy(X_train).float()
        labels_all = torch.from_numpy(y_train).float().unsqueeze(1)
        r_all = torch.from_numpy(r_train_unscaled).float().unsqueeze(1)
        
        optimizer_pinn.zero_grad()
        h_prob, v_pred = pinn_model(inputs_all)
        c_loss = criterion_bce(h_prob, labels_all)
        p_loss = physics_loss(v_pred, st_mass_train_t, r_all)
        total_loss = c_loss + p_loss
        total_loss.backward()
        optimizer_pinn.step()
        
    print("--- PINN Model Trained ---")
    
    return rf_model, dnn_model, pinn_model, scaler, X_test, y_test

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()

        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        rf_model, dnn_model, scaler, X_test, y_test = get_trained_models(labeled_df)
        
        # Evaluate RF
        y_pred_rf = rf_model.predict(X_test)
        print("\nRandom Forest Evaluation:")
        print(classification_report(y_test, y_pred_rf))
        
        # Evaluate DNN
        y_pred_dnn = dnn_model.predict(X_test)
        print("\nDNN Evaluation:")
        print(classification_report(y_test, y_pred_dnn))
