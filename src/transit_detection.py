import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

def simulate_light_curve(has_transit=True, noise_level=0.001, transit_depth=0.01, transit_duration=50):
    """Generates a synthetic light curve.
    - has_transit: If True, a transit dip is added.
    - noise_level: The standard deviation of the Gaussian noise.
    - transit_depth: The percentage of brightness drop during a transit.
    - transit_duration: The number of data points the transit lasts.
    """
    time = np.arange(0, 1000)
    flux = np.ones_like(time, dtype=float)
    
    if has_transit:
        start = np.random.randint(100, 800)
        flux[start:start + transit_duration] -= transit_depth
        
    # Add Gaussian noise
    flux += np.random.normal(0, noise_level, size=time.shape)
    return time, flux

def extract_features(flux):
    """Extracts features from a light curve.
    - Finds the most prominent dip in brightness.
    - Returns its depth, duration, and area.
    """
    # Invert the flux to find peaks (which are dips in brightness)
    inverted_flux = 1.0 - flux
    # Smooth the data to reduce noise
    smoothed_flux = pd.Series(inverted_flux).rolling(window=10, center=True).mean().fillna(0)
    
    # Find peaks (dips)
    peaks, properties = find_peaks(smoothed_flux, prominence=0.001)
    
    if len(peaks) == 0:
        return [0, 0, 0] # No significant dip found
        
    # Get the most prominent dip
    most_prominent_peak_idx = np.argmax(properties['prominences'])
    peak_loc = peaks[most_prominent_peak_idx]
    
    # Extract features
    depth = properties['prominences'][most_prominent_peak_idx]
    duration = properties['widths'][most_prominent_peak_idx]
    area = depth * duration # A simple shape feature
    
    return [depth, duration, area]

def train_transit_model(n_samples=500):
    """Trains a model to classify light curves.
    - Creates a dataset of light curves with and without transits.
    - Extracts features from each.
    - Trains a RandomForestClassifier.
    """
    print("\n--- Training Transit Detection Model ---")
    features = []
    labels = []
    
    # Create a balanced dataset
    for i in range(n_samples):
        has_transit = i % 2 == 0
        _, flux = simulate_light_curve(has_transit=has_transit)
        feats = extract_features(flux)
        features.append(feats)
        labels.append(1 if has_transit else 0)
        
    X = np.array(features)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Transit detection model trained with an accuracy of {accuracy:.2f}")
    
    return model

def predict_and_visualize(model):
    """Generates a new light curve and uses the model to predict if it has a transit."""
    print("\n--- Predicting on a New Light Curve ---")
    # Create a new light curve with a transit
    test_time, test_flux = simulate_light_curve(has_transit=True, transit_depth=0.015, transit_duration=40)
    
    # Extract features and predict
    test_features = extract_features(test_flux)
    prediction = model.predict([test_features])[0]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(test_time, test_flux, '.', alpha=0.6, label='Simulated Star Brightness')
    plt.title('Light Curve Analysis')
    plt.xlabel('Time')
    plt.ylabel('Relative Brightness')
    
    result_text = "Planet Detected!" if prediction == 1 else "No Planet Detected."
    color = 'lightgreen' if prediction == 1 else 'salmon'
    plt.text(50, np.min(test_flux), result_text, fontsize=15, bbox=dict(facecolor=color, alpha=0.8))
    
    plt.legend()
    plt.savefig('plots/transit_detection_result.png')
    print("Saved transit detection plot.")
    print(f"Model Prediction: {result_text}")

if __name__ == "__main__":
    # Train the model
    transit_model = train_transit_model()
    
    # Predict on a new sample and visualize
    predict_and_visualize(transit_model)
    
    print("\nTransit detection module finished.")
