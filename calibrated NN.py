# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:07:16 2024

@author: jackt
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import norm, gaussian_kde
from pyDOE import lhs
import seaborn as sns

# ----------------------------------------------
# Step 1: Load and Clean Real Data
# ----------------------------------------------

# Load the real data
file_path = r'C:\Users\jackt\Downloads\spy_sample-1.csv'
df = pd.read_csv(file_path)

# Select and rename the relevant columns and calculate features like Moneyness
calib_data = df[['UNDERLYING_LAST', 'QUOTE_DATE', 'EXPIRE_DATE', 'C_IV', 'C_ASK', 
                'STRIKE', 'C_RHO', 'C_DELTA', 'C_VEGA', 'C_GAMMA']].copy()

calib_data.rename(columns={
    'UNDERLYING_LAST': 'S_real',    # Stock price (real)
    'STRIKE': 'K_real',             # Strike price (real)
    'C_ASK': 'C_real',              # Real call option price
    'C_IV': 'SIGMA_real',           # Implied volatility (real)
    'C_RHO': 'rho_real',
    'C_DELTA': 'delta_real',
    'C_VEGA': 'vega_real',
    'C_GAMMA': 'gamma_real'         # Greeks (real)
}, inplace=True)

# Convert dates to datetime format
calib_data['QUOTE_DATE'] = pd.to_datetime(calib_data['QUOTE_DATE'])
calib_data['EXPIRE_DATE'] = pd.to_datetime(calib_data['EXPIRE_DATE'])

# Calculate time to expiry in trading years (252 trading days per year)
calib_data['DAYS_TO_EXPIRY_real'] = (calib_data['EXPIRE_DATE'] - calib_data['QUOTE_DATE']).dt.days
calib_data['T_real'] = calib_data['DAYS_TO_EXPIRY_real'] / 252

# Calculate moneyness for real data
calib_data['M_real'] = calib_data['S_real'] / calib_data['K_real']

# Drop any rows with missing values
calib_data.dropna(inplace=True)

# Handle zero option prices in `C_real`
calib_data = calib_data[calib_data['C_real'] > 0]  # Remove rows with zero option prices

# Calculate the changes in volatility (Δσ_t)
calib_data['sigma_t_next_real'] = calib_data['SIGMA_real'].shift(-1)  # Shift volatility to simulate next time step
calib_data['delta_sigma_real'] = calib_data['sigma_t_next_real'] - calib_data['SIGMA_real']
calib_data.dropna(subset=['delta_sigma_real'], inplace=True)  # Drop the last row due to NaN in σ_t_next

# ----------------------------------------------
# Step 2: Train Neural Networks on Real Data (for a(σ_t, t) and b(σ_t, t))
# ----------------------------------------------

# Input features for a(σ_t, t) and b(σ_t, t) for real data
X_real = calib_data[['S_real', 'K_real', 'T_real', 'SIGMA_real', 'gamma_real', 'delta_real', 'vega_real', 'rho_real']]
y_a_real = calib_data['delta_sigma_real']  # Target for a(σ_t, t)
y_b_real = calib_data['delta_sigma_real']  # Target for b(σ_t, t)

# Split data into training and test sets for real data
X_train_real, X_test_real, y_a_train, y_a_test = train_test_split(X_real, y_a_real, test_size=0.2, random_state=42)
X_train_real, X_test_real, y_b_train, y_b_test = train_test_split(X_real, y_b_real, test_size=0.2, random_state=42)

# Standardize the input features for real data
scaler_real = StandardScaler()
X_train_scaled_real = scaler_real.fit_transform(X_train_real)
X_test_scaled_real = scaler_real.transform(X_test_real)

# Define and train models for a(σ_t, t) and b(σ_t, t)
mlp_a = MLPRegressor(hidden_layer_sizes=(256, 128, 128, 64, 32), activation='relu', max_iter=2500, random_state=42)
mlp_b = MLPRegressor(hidden_layer_sizes=(256, 128, 128, 64, 32), activation='relu', max_iter=2500, random_state=42)

mlp_a.fit(X_train_scaled_real, y_a_train)
mlp_b.fit(X_train_scaled_real, y_b_train)

# ----------------------------------------------
# Step 3: Hull-White Fourier-Based Pricing with a(σ_t, t) and b(σ_t, t)
# ----------------------------------------------

# Use the trained models to predict a(σ_t, t) and b(σ_t, t) dynamically during pricing
def a_func(sigma_t, t, S, K, T, rho, delta, vega, gamma):
    return mlp_a.predict([[S, K, T, sigma_t, rho, delta, vega, gamma]])[0]

def b_func(sigma_t, t, S, K, T, rho, delta, vega, gamma):
    return mlp_b.predict([[S, K, T, sigma_t, rho, delta, vega, gamma]])[0]

# Preprocessing function to remove NaN rows before calculations
def remove_nan_rows(*arrays):
    mask = np.any([np.isnan(array) for array in arrays], axis=0)
    return [array[~mask] for array in arrays]

def hull_white_characteristic_function(S, K, T, u, sigma, r, rho, delta, vega, gamma, a_func, b_func):
    mu = np.log(S) + (r - 0.5 * sigma ** 2) * T
    variance = 0
    N_steps = 100
    dt = T / N_steps
    sigma_t = sigma

    for t in np.linspace(0, T, N_steps):
        drift = a_func(sigma_t, t, S, K, T, rho, delta, vega, gamma) * dt
        diffusion_term = b_func(sigma_t, t, S, K, T, rho, delta, vega, gamma)
        
        # Ensure dt is positive for the square root
        if dt > 0:
            diffusion = diffusion_term * np.random.normal(0, np.sqrt(dt))
        else:
            diffusion = 0
        
        sigma_t += drift + diffusion
        sigma_t = np.clip(sigma_t, 1e-6, 1)  # Clip sigma_t to reasonable values
        
        # Accumulate variance with clipping
        variance = np.clip(variance + sigma_t ** 2 * dt, 0, 1e6)

    exp_input = 1j * u * mu - 0.5 * u ** 2 * variance
    exp_input_clipped = np.clip(exp_input, -700, 700)  # Avoid overflow in exp
    return np.exp(exp_input_clipped)

# Fourier pricing function
def fft_price(S, K, T, sigma, r, rho, delta, vega, gamma, alpha=1.5, eta=0.25, N=4096):
    V = np.arange(0, N * eta, eta)
    b, lamb, k = log_strike_partition(eta, N)
    pm_one = np.empty((N,))
    pm_one[::2] = -1
    pm_one[1::2] = 1
    Weights = 3 + pm_one
    Weights[0] -= 1
    Weights = (eta / 3) * Weights
    x = np.exp(1j * b * V) * hull_white_characteristic_function(S, K, T, V - (alpha + 1) * 1j, sigma, r, rho, delta, vega, gamma, a_func, b_func) * Weights

    # Clip exponential input to avoid overflow
    exp_input = np.clip(-alpha * k, -700, 700)
    callPrices = np.real((np.exp(exp_input) / np.pi) * fft(x))

    strikes = np.exp(k)
    return np.interp(K, strikes, callPrices)


def log_strike_partition(eta=0.25, N=4096):
    b = np.pi / eta
    lamb = 2 * np.pi / (eta * N)
    k = -b + lamb * np.arange(0, N)

    # Clip k values to prevent overflow in later calculations
    k = np.clip(k, -700, 700)

    return b, lamb, k


# ----------------------------------------------
# Step 4: Generate Synthetic Data Using KDE
# ----------------------------------------------

# Use KDE to approximate real data distributions
T_real_kde = gaussian_kde(calib_data['T_real'])
gamma_real_kde = gaussian_kde(calib_data['gamma_real'])
vega_real_kde = gaussian_kde(calib_data['vega_real'])
delta_real_kde = gaussian_kde(calib_data['delta_real'])
rho_real_kde = gaussian_kde(calib_data['rho_real'])
S_real_kde = gaussian_kde(calib_data['S_real'])
K_real_kde = gaussian_kde(calib_data['K_real'])

# Generate synthetic data
n_samples = 200000

# Sample synthetic data based on KDE
synthetic_T = T_real_kde.resample(n_samples).reshape(-1)
synthetic_gamma = gamma_real_kde.resample(n_samples).reshape(-1)
synthetic_vega = vega_real_kde.resample(n_samples).reshape(-1)
synthetic_delta = delta_real_kde.resample(n_samples).reshape(-1)
synthetic_rho = rho_real_kde.resample(n_samples).reshape(-1)


synthetic_S = S_real_kde.resample(n_samples).reshape(-1)  # Assuming S is underlying
synthetic_K = K_real_kde.resample(n_samples).reshape(-1)

synthetic_moneyness = S_real_kde.resample(n_samples).reshape(-1) / K_real_kde.resample(n_samples).reshape(-1)

# Clean synthetic data before pricing (remove NaNs)
synthetic_S_clean, synthetic_K_clean, synthetic_T_clean, synthetic_gamma_clean, \
    synthetic_rho_clean, synthetic_delta_clean, synthetic_vega_clean = remove_nan_rows(
        synthetic_S, synthetic_K, synthetic_T, synthetic_gamma, synthetic_rho, synthetic_delta, synthetic_vega
)

# Now call the FFT pricing method
HW_prices = np.array([
    fft_price(
        S=synthetic_S_clean[i], K=synthetic_K_clean[i], T=synthetic_T_clean[i], 
        sigma=synthetic_gamma_clean[i], r=0.05, rho=synthetic_rho_clean[i], 
        delta=synthetic_delta_clean[i], vega=synthetic_vega_clean[i], gamma=synthetic_gamma_clean[i]
    ) for i in range(len(synthetic_S_clean))
])

# ----------------------------------------------
# Step 5: Train on Synthetic Data and Validate
# ----------------------------------------------


# Step 1: Create the synthetic dataset for training the option pricing model
X_BS_greeks_HW = pd.DataFrame({
    'T': synthetic_T_clean, 
    'Moneyness': synthetic_S_clean / synthetic_K_clean, 
    'Gamma': synthetic_gamma_clean, 
    'Vega': synthetic_vega_clean, 
    'Rho': synthetic_rho_clean, 
    'Delta': synthetic_delta_clean,
    'HW_price': HW_prices  # Add HW_prices to the DataFrame
})

# Step 2: Filter out rows with extreme HW_prices values
# Remove rows where HW_price is less than 0.1 or greater than 600
X_BS_greeks_HW_cleaned = X_BS_greeks_HW[(X_BS_greeks_HW['HW_price'] >= 0.01) & (X_BS_greeks_HW['HW_price'] <= 600)]

# Step 3: Separate the features (X) and target (y)
X_cleaned = X_BS_greeks_HW_cleaned[['T', 'Moneyness', 'Gamma', 'Vega', 'Rho', 'Delta']]  # Features
y_cleaned = X_BS_greeks_HW_cleaned['HW_price']  # Target (prices)

# Check the cleaned dataset
print(X_cleaned.head())
print(y_cleaned.head())


# Split cleaned synthetic data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Standardize the input features for synthetic data
scaler_synthetic = StandardScaler()
X_train_scaled = scaler_synthetic.fit_transform(X_train)
X_test_scaled = scaler_synthetic.transform(X_test)

# Define and train the neural network model for synthetic data
mlp_price = MLPRegressor(hidden_layer_sizes=(256, 128, 128, 64, 32), activation='relu', max_iter=4500, random_state=42)
mlp_price.fit(X_train_scaled, y_train)

# Predict on the synthetic test data
y_pred = mlp_price.predict(X_test_scaled)
print(f'Hull-White Model MSE on synthetic data: {mean_squared_error(y_test, y_pred)}')


# Plot the training loss curve
plt.figure(figsize=(12, 6))
plt.plot(mlp_price.loss_curve_, label='Hull-White Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Plot Prediction vs Actual
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Hull-White Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Prediction vs Actual')
plt.legend()
plt.show()

# Plot Error Distribution
errors_HW = y_test - y_pred

plt.figure(figsize=(12, 6))
sns.histplot(errors_HW, bins=450, kde=True, color='blue', label='Hull-White Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.show()

