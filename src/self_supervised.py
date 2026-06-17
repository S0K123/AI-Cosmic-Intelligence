import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
from feature_engineering import clean_and_prepare_data

class FeatureMaskingModel(nn.Module):
    def __init__(self, input_dim):
        super(FeatureMaskingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_self_supervised(df, epochs=50):
    """
    Trains a self-supervised model by masking some features and predicting them.
    """
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    data = df[features].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = torch.tensor(data_scaled, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = FeatureMaskingModel(input_dim=len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("\n--- Self-Supervised Pretraining (Feature Masking) ---")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs = batch[0]
            
            # Mask some features (e.g., mask 30% of features randomly)
            mask = torch.rand(inputs.shape) > 0.3
            masked_inputs = inputs * mask
            
            optimizer.zero_grad()
            outputs, _ = model(masked_inputs)
            loss = criterion(outputs, inputs) # Try to reconstruct original
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
            
    print("Self-supervised pretraining complete.")
    return model, scaler

def get_ssl_embeddings(model, scaler, df):
    """
    Extracts pretrained embeddings from the encoder.
    """
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    data = df[features].values
    data_scaled = scaler.transform(data)
    
    X = torch.tensor(data_scaled, dtype=torch.float32)
    with torch.no_grad():
        _, embeddings = model(X)
    return embeddings.numpy()

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        prepared_df = clean_and_prepare_data(raw_df)
        model, scaler = train_self_supervised(prepared_df)
        embeddings = get_ssl_embeddings(model, scaler, prepared_df)
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Save results for downstream tasks
        torch.save(model.state_dict(), 'results/ssl_model.pth')
        print("Pretrained SSL model saved.")
