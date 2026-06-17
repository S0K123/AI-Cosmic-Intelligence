import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
from feature_engineering import clean_and_prepare_data

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Scale features to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(df, epochs=50):
    """
    Trains a VAE to learn the distribution of real planetary systems.
    """
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    data = df[features].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = torch.tensor(data_scaled, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = VAE(input_dim=len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n--- Training Generative Planetary System Model (VAE) ---")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader.dataset):.4f}")
            
    print("VAE training complete.")
    return model, scaler

def generate_synthetic_system(model, scaler, num_planets=5):
    """
    Generates a synthetic planetary system using the trained VAE.
    """
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_planets, model.fc_mu.out_features)
        sample_scaled = model.decoder(z).numpy()
        
    sample = scaler.inverse_transform(sample_scaled)
    features = ['planet_radius', 'orbital_period', 'star_temp_k', 'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
    return pd.DataFrame(sample, columns=features)

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        prepared_df = clean_and_prepare_data(raw_df)
        vae_model, scaler = train_vae(prepared_df)
        synthetic_system = generate_synthetic_system(vae_model, scaler)
        print("\nGenerated Synthetic Planetary System:")
        print(synthetic_system)
        
        # Save model
        torch.save(vae_model.state_dict(), 'results/vae_model.pth')
