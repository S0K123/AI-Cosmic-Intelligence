import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from space_simulator import CelestialBody, simulate_step
from train_model import get_trained_models, create_habitability_label
from feature_engineering import clean_and_prepare_data
from data_loader import load_data
import torch

class PlanetDiscoveryEnv(gym.Env):
    def __init__(self, star_data, models, scaler):
        super(PlanetDiscoveryEnv, self).__init__()
        self.star_data = star_data
        self.models = models # rf_model, dnn_model, pinn_model
        self.scaler = scaler
        
        # Action space: propose [radius, semi_major_axis, eccentricity]
        # Ranges: radius (0.5 to 2.5), semi-major axis (0.01 to 2.0), eccentricity (0.0 to 0.1)
        self.action_space = spaces.Box(low=np.array([0.5, 0.01, 0.0]), 
                                       high=np.array([2.5, 2.0, 0.1]), 
                                       dtype=np.float32)
        
        # Observation space: star system parameters
        # Features: [star_temp_k, star_radius_solar, star_luminosity]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.state = np.array([star_data['star_temp_k'], 
                               star_data['star_radius_solar'], 
                               star_data['star_luminosity']], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.state, {}

    def step(self, action):
        radius, semi_major_axis, eccentricity = action
        
        # 1. Habitability Reward
        # We need orbital period using Kepler's Third Law for the model input
        star_mass = self.star_data.get('st_mass', 1.0)
        period = np.sqrt((semi_major_axis**3 / star_mass)) * 365.25
        
        planet_data = {
            'planet_radius': radius,
            'orbital_period': period,
            'star_temp_k': self.star_data['star_temp_k'],
            'orbit_semi_major_axis': semi_major_axis,
            'star_radius_solar': self.star_data['star_radius_solar'],
            'star_luminosity': self.star_data['star_luminosity']
        }
        
        # Prepare for model prediction
        X_cand = self.scaler.transform(pd.DataFrame([planet_data]))
        prob_rf = self.models['rf'].predict_proba(X_cand)[0, 1]
        
        reward = 0
        if prob_rf > 0.6:
            reward += 2 # Predicted habitable
        
        # 2. Stability Reward
        # We'll use a simplified stability check for the RL loop to keep it fast
        # A more rigorous test is done in the final pipeline
        is_stable = self._check_stability(radius, semi_major_axis, eccentricity)
        if is_stable:
            reward += 2
        else:
            reward -= 2 # Unstable orbit
            
        # 3. Uncertainty Reward (placeholder, using DNN standard dev later)
        reward += 1 # Base reward for exploration
        
        done = True # Each step is a full proposal
        truncated = False
        
        return self.state, reward, done, truncated, {"planet": planet_data}

    def _check_stability(self, radius, semi_major_axis, eccentricity):
        # Placeholder for simplified stability check
        # Real systems use the N-body simulator in Step 4
        return True

def train_rl_agent(df, rf_model, dnn_model, pinn_model, scaler):
    """
    Trains a PPO agent to discover stable habitable planets.
    """
    # Use a specific star system for training (e.g., TRAPPIST-1)
    star_data = df[df['hostname'].str.contains('TRAPPIST-1', na=False)].iloc[0].to_dict()
    
    models = {'rf': rf_model, 'dnn': dnn_model, 'pinn': pinn_model}
    
    env = PlanetDiscoveryEnv(star_data, models, scaler)
    
    print("\n--- Training RL Planet Discovery Agent (PPO) ---")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)
    
    print("RL agent training complete.")
    return model

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        from train_model import get_trained_models
        rf, dnn, pinn, scaler, _, _ = get_trained_models(labeled_df)
        
        rl_agent = train_rl_agent(labeled_df, rf, dnn, pinn, scaler)
        
        # Generate a candidate
        star_data = labeled_df[labeled_df['hostname'].str.contains('TRAPPIST-1', na=False)].iloc[0].to_dict()
        env = PlanetDiscoveryEnv(star_data, {'rf': rf, 'dnn': dnn, 'pinn': pinn}, scaler)
        obs, _ = env.reset()
        action, _states = rl_agent.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        
        print("\nRL-Proposed Planet Candidate:")
        print(info['planet'])
        print(f"Total Reward: {reward}")
        
        # Save agent
        rl_agent.save("results/rl_discovery_agent")
