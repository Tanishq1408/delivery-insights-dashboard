"""
ETA Prediction Model - Trains XGBoost to predict delivery times
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

class ETAPredictionModel:
    def __init__(self):
        self.features_path = 'data/features'
        self.models_path = 'models'
        os.makedirs(self.models_path, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        return pd.read_csv(f'{self.features_path}/ml_dataset.csv')
    
    def split_data(self, df, target_col='target_duration'):
        X = df.drop(['target_duration', 'target_is_late'], axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def train_model(self, X_train, y_train):
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        within_5min = np.mean(np.abs(y_test - y_pred) <= 5) * 100
        within_15min = np.mean(np.abs(y_test - y_pred) <= 15) * 100
        within_30min = np.mean(np.abs(y_test - y_pred) <= 30) * 100
        return {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'within_5min': within_5min, 'within_15min': within_15min, 'within_30min': within_30min
        }
    
    def get_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names, 'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self):
        joblib.dump(self.model, f'{self.models_path}/eta_model.pkl')
        joblib.dump(self.scaler, f'{self.models_path}/scaler.pkl')
    
    def run(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test, feature_names = self.split_data(df)
        self.train_model(X_train, y_train)
        metrics = self.evaluate_model(X_test, y_test)
        feat_imp = self.get_feature_importance(feature_names)
        self.save_model()
        return metrics, feat_imp
