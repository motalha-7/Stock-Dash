import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pickle
from sklearn.preprocessing import StandardScaler


def train_svr_model(stock_code, days=60):
    # Fetch stock data for a longer period
    df = yf.download(stock_code, period='1y')
    
    # Prepare additional features
    df['Date'] = df.index
    df['Day'] = df['Date'].dt.dayofyear
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Momentum'] = df['Close'].diff(3)
    
    # Drop rows with NaN values created by rolling operations
    df = df.dropna()
    
    # Prepare the features (X) and target (y)
    X = df[['Day', 'SMA_20', 'Volatility', 'Momentum']].values
    y = df['Close'].values
    
    # Debugging: Print features and target
    print("X (Features):\n", X[:5])
    print("y (Target):\n", y[:5])
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1e-2, 1e-3, 1e-4], 'epsilon': [0.1, 0.01, 0.001]}
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Debugging: Check best parameters and cross-validation score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)
    
    # Train the SVR model with the best parameters
    best_svr = grid_search.best_estimator_
    best_svr.fit(X_train, y_train)
    
    # Debugging: Check model predictions
    y_train_pred = best_svr.predict(X_train)
    y_test_pred = best_svr.predict(X_test)
    
    print("Training Predictions:\n", y_train_pred[:5])
    print("Testing Predictions:\n", y_test_pred[:5])
    
    # Test the model
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    
    # Save the model
    with open('svr_model.pkl', 'wb') as file:
        pickle.dump(best_svr, file)
    
    return mse, mae



def load_svr_model():
    return joblib.load('svr_model.pkl')
