import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mapping dictionary for labels
LABEL_MAP = {
    0: "Normal ECG",
    1: "ST-T Abnormality",
    2: "LV Hypertrophy"
}

def load_data(filepath):
    """
    Loads ECG data from a CSV file.
    Assumes the last column is the label.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # The last column is the label, all other columns are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def clean_signal(X):
    """
    Applies basic preprocessing/cleaning to the signal.
    Standardizes each signal sample individually across its time steps.
    """
    print("Standardizing signals per sample...")
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    X_scaled = (X - mean) / std
    return X_scaled, None

def prepare_data(filepath, test_size=0.2, random_state=42):
    """
    Loads, cleans, and splits the data into training and testing sets.
    Reshapes data for CNN+LSTM input.
    """
    X, y = load_data(filepath)
    X_scaled, scaler = clean_signal(X)
    
    # Reshape X for CNN+LSTM: (samples, time_steps, features)
    # Our data is 1D signal, so features=1
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    print(f"Data shape after reshaping: {X_reshaped.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the preprocessing module standalone
    try:
        X_train, X_test, y_train, y_test, scaler = prepare_data("data/dummy_data.csv")
        print("Data preprocessing completed successfully.")
    except FileNotFoundError:
        print("Dataset not found. Please run create_dummy_data.py first.")
