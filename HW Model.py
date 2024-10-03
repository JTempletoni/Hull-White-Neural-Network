# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:16:44 2024

@author: jackt
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.fft import fft
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyDOE import lhs
from scipy.stats import norm

# Define the number of samples
n_samples = 100000

# Define the ranges for each parameter using realistic distributions
param_ranges = {
    'S0': (10, 350),      # Underlying asset price
    'K': (10, 350),       # Strike price
    'T': (0.1, 3),        # Time to expiry in years
    'sigma': (0.01, 0.5),    # Volatility
    'r': (0.01, 0.2)      # Risk-free interest rate
}

# Use Latin Hypercube Sampling to generate samples
lhs_samples = lhs(len(param_ranges), samples=n_samples)
param_grid = np.column_stack([lhs_samples[:, i] * (param_ranges[param][1] - param_ranges[param][0]) + param_ranges[param][0] for i, param in enumerate(param_ranges)])

# Use Latin Hypercube Sampling to generate samples
lhs_samples = lhs(len(param_ranges), samples=n_samples)
param_grid = np.column_stack([lhs_samples[:, i] * (param_ranges[param][1] - param_ranges[param][0]) + param_ranges[param][0] for i, param in enumerate(param_ranges)])

def hull_white_characteristic_function(S0, r, sigma, T, u):
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    var = T * sigma**2
    return np.exp(1j * u * mu - 0.5 * u**2 * var)

def modified_call_fft(S0, r, sigma, T, v, alpha):
    denom = (alpha**2 + alpha - v**2) + (2 * alpha + 1) * v * 1j
    phi_T_v = hull_white_characteristic_function(S0, r, sigma, T, v - (alpha + 1) * 1j)
    return np.exp(-r * T) * phi_T_v / denom

def log_strike_partition(eta=0.25, N=4096):
    b = np.pi / eta
    lamb = 2 * np.pi / (eta * N)
    k = -b + lamb * np.arange(0, N)
    return b, lamb, k

def fft_price(S0, r, sigma, K, T, alpha=1.5, eta=0.25, N=4096):
    V = np.arange(0, N * eta, eta)
    b, lamb, k = log_strike_partition(eta, N)
    pm_one = np.empty((N,))
    pm_one[::2] = -1
    pm_one[1::2] = 1
    Weights = 3 + pm_one
    Weights[0] -= 1
    Weights = (eta / 3) * Weights
    x = np.exp(1j * b * V) * modified_call_fft(S0, r, sigma, T, V, alpha) * Weights
    callPrices = np.real((np.exp(-alpha * k) / np.pi) * fft(x))
    strikes = np.exp(k)
    call_price_at_K = np.interp(K, strikes, callPrices)
    return call_price_at_K

def calculate_hw_prices(S0, K, T, sigma, r):
    return fft_price(S0, r, sigma, K, T, alpha=1.5, eta=0.25, N=4096)

# Generate synthetic option prices
HW_prices = np.array([calculate_hw_prices(*params) for params in param_grid])

# Extract time to expiry (T) and moneyness (S0/K) as features
T = param_grid[:, 2]
moneyness = param_grid[:, 0] / param_grid[:, 1]

# Combine the features and target into a DataFrame
data = pd.DataFrame({
    'T': T,
    'Moneyness': moneyness,
    'HW_price': HW_prices
})

# Validation of the synthetic data
print("Synthetic Data Validation:")
print(data.describe())
print("\nChecking correlations...")
print(data.corr())

# Save to CSV (optional)
data.to_csv('hull_white_option_pricing_data.csv', index=False)

# Load the dataset (optional if using saved data)
data = pd.read_csv('hull_white_option_pricing_data.csv')

# Split the dataset into inputs and targets
X_HW= data[['T', 'Moneyness']]
y_HW= data['HW_price']

# Split into training and test sets
X_train, X_test, y_HW_train, y_HW_test = train_test_split(X_HW, y_HW, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model with 5 layers
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 128, 64, 32),  # 5 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=2500,
    random_state=42
)

# Fit the model
mlp.fit(X_train_scaled, y_HW_train)

# Predict and evaluate
y_HW_pred = mlp.predict(X_test_scaled)
print(f'Hull-White Model MSE: {mean_squared_error(y_HW_test, y_HW_pred)}')

# Save the model and scaler
joblib.dump(mlp, 'hull_white_best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Plot the training loss curve
plt.figure(figsize=(12, 6))
plt.plot(mlp.loss_curve_, label='Hull-White Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Plot Prediction vs Actual
plt.figure(figsize=(12, 6))
plt.scatter(y_HW_test, y_HW_pred, alpha=0.5, label='Hull-White Predictions')
plt.plot([min(y_HW_test), max(y_HW_test)], [min(y_HW_test), max(y_HW_test)], 'r', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Prediction vs Actual')
plt.legend()
plt.show()

# Plot Error Distribution
errors_HW = y_HW_test - y_HW_pred

plt.figure(figsize=(12, 6))
sns.histplot(errors_HW, bins=30, kde=True, color='blue', label='Hull-White Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.show()

