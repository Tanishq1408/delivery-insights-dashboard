"""
Feature Engineering - Creates smart features for ML model
"""

import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self):
        self.processed_path = 'data/processed'
        self.features_path = 'data/features'
        os.makedirs(self.features_path, exist_ok=True)
    
    def create_features(self, df):
        features = df.copy()
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month'] = features['date'].dt.month
        features['is_month_start'] = features['date'].dt.is_month_start
        features['is_month_end'] = features['date'].dt.is_month_end
        exp_map = {'New': 1, 'Intermediate': 2, 'Experienced': 3, 'Expert': 4}
        features['experience_numeric'] = features['experience_level'].map(exp_map)
        features['driver_avg_satisfaction'] = features.groupby('driver_id')['customer_satisfaction'].transform('mean')
        features['driver_success_rate'] = features.groupby('driver_id')['is_delivered'].transform('mean')
        features['stops_per_route'] = features.groupby(['date', 'driver_id'])['stop_number'].transform('max')
        features['route_density'] = features['stops_per_route'] / features['distance_from_depot_km']
        weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}
        features['weather_numeric'] = features['weather'].map(weather_map)
        traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
        features['traffic_numeric'] = features['traffic_level'].map(traffic_map)
        features['lat_diff'] = abs(features['delivery_lat'] - features['latitude'])
        features['lon_diff'] = abs(features['delivery_lon'] - features['longitude'])
        features['euclidean_distance'] = np.sqrt(features['lat_diff']**2 + features['lon_diff']**2)
        features['is_heavy'] = features['parcel_weight_kg'] > 10
        features['depot_avg_daily_volume'] = features.groupby('depot_id')['delivery_id'].transform('count')
        features['target_duration'] = features['delivery_duration_min']
        features['target_is_late'] = features['delivery_duration_min'] > 15
        return features
    
    def prepare_ml_dataset(self, features_df):
        feature_cols = [
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month', 'is_month_start', 'is_month_end', 'is_weekend', 'is_peak_season',
            'experience_numeric', 'base_stops_per_day', 'delivery_accuracy',
            'driver_avg_satisfaction', 'driver_success_rate',
            'stops_per_route', 'route_density',
            'weather_numeric', 'traffic_numeric',
            'distance_from_depot_km', 'lat_diff', 'lon_diff', 'euclidean_distance',
            'parcel_weight_kg', 'is_heavy', 'is_express',
            'depot_avg_daily_volume'
        ]
        ml_df = features_df[feature_cols + ['target_duration', 'target_is_late']].copy()
        ml_df = ml_df.fillna(ml_df.median())
        return ml_df
    
    def run(self):
        df = pd.read_csv(f'{self.processed_path}/deliveries_merged.csv')
        df['date'] = pd.to_datetime(df['date'])
        features = self.create_features(df)
        ml_df = self.prepare_ml_dataset(features)
        features.to_csv(f'{self.features_path}/features_all.csv', index=False)
        ml_df.to_csv(f'{self.features_path}/ml_dataset.csv', index=False)
        return ml_df
