# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\ML PROJECT\\WineQT.csv")

# Select all 11 relevant features
features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'ph', 'sulphates', 'alcohol']

X = df[features]
y = df['quality']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize XGBoost regressor
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define parameter grid for tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}

# Perform RandomizedSearchCV
tuner = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, 
                           n_iter=20, cv=5, scoring='neg_mean_squared_error', 
                           verbose=1, random_state=42, n_jobs=-1)

tuner.fit(X_train, y_train)

# Get best model
best_model = tuner.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {tuner.best_params_}")
print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2:.2f}")

# Save model and scaler
with open('red_wine_regression.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
