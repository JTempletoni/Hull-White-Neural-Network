# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:38:53 2024

@author: jackt
"""

import numpy as np
import pandas as pd
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

def black_scholes_price(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Generate synthetic option prices
black_scholes_prices = np.array([black_scholes_price(*params) for params in param_grid])

# Extract time to expiry (T) and moneyness (S0/K) as features
T = param_grid[:, 2]
moneyness = param_grid[:, 0] / param_grid[:, 1]

# Combine the features and target into a DataFrame
data = pd.DataFrame({
    'T': T,
    'Moneyness': moneyness,
    'BS_price': black_scholes_prices
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
X_BS= data[['T', 'Moneyness']]
y_BS = data['BS_price']

# Split into training and test sets
X_train, X_test, y_BS_train, y_BS_test = train_test_split(X_BS, y_BS, test_size=0.2, random_state=42)

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
mlp.fit(X_train_scaled, y_BS_train)


# Predict and evaluate
y_BS_pred = mlp.predict(X_test_scaled)
print(f'Black-Scholes Model MSE: {mean_squared_error(y_BS_test, y_BS_pred)}')

# Save the model and scaler
joblib.dump(mlp, 'hull_white_best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Plot the training loss curve
plt.figure(figsize=(12, 6))
plt.plot(mlp.loss_curve_, label='Black-Scholes Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Plot Prediction vs Actual
plt.figure(figsize=(12, 6))
plt.scatter(y_BS_test, y_BS_pred, alpha=0.5, label='Black-Scholes  Predictions')
plt.plot([min(y_BS_test), max(y_BS_test)], [min(y_BS_test), max(y_BS_test)], 'r', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Prediction vs Actual')
plt.legend()
plt.show()

# Plot Error Distribution
errors_BS = y_BS_test - y_BS_pred

plt.figure(figsize=(12, 6))
sns.histplot(errors_BS, bins=150, kde=True, color='blue', label='Black-Scholes  Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

#
# Independent variables (T and Moneyness)
X = X_BS
X = sm.add_constant(X)  # Add constant term for the intercept

# Dependent variable (option price)
y = y_BS

# Perform the regression
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# Save the regression summary as an image
summary = model.summary()
fig = plt.figure(figsize=(12, 8))
plt.text(0.01, 0.05, str(summary), {'fontsize': 10}, fontproperties='monospace')  # use monospaced font
plt.axis('off')
plt.savefig('regression_summary.png', dpi=300, bbox_inches='tight')

# Plot the regression results for 'Moneyness (S/K)'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['Moneyness'], y=y, label='Actual Prices')
sns.lineplot(x=X['Moneyness'], y=model.fittedvalues, color='red', label='Fitted Line')
plt.title('Regression of Option Price on Moneyness (S/K)')
plt.xlabel('Moneyness (S/K)')
plt.ylabel('Option Price')
plt.legend()
plt.show()

