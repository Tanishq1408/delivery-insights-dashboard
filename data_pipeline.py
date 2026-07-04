"""
Data Pipeline - Cleans and prepares raw delivery data
"""

import pandas as pd
import numpy as np
import os

class DataPipeline:
    def __init__(self):
        self.raw_path = 'data/raw'
        self.processed_path = 'data/processed'
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_raw_data(self):
        depots = pd.read_csv(f'{self.raw_path}/depots.csv')
        drivers = pd.read_csv(f'{self.raw_path}/drivers.csv')
        deliveries = pd.read_csv(f'{self.raw_path}/deliveries.csv')
        return depots, drivers, deliveries
    
    def clean_deliveries(self, deliveries_df):
        df = deliveries_df.copy()
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        df['actual_time'] = pd.to_datetime(df['actual_time'])
        df['date'] = pd.to_datetime(df['date'])
        df['delivery_duration_min'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
        df['is_on_time'] = df['delivery_duration_min'] <= 15
        df['hour'] = df['scheduled_time'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['is_peak_season'] = df['date'].dt.day.between(10, 20)
        df['is_delivered'] = df['outcome'] == 'Delivered'
        def time_category(hour):
            if hour <= 10: return 'Morning'
            elif hour <= 14: return 'Midday'
            elif hour <= 17: return 'Afternoon'
            else: return 'Evening'
        df['time_category'] = df['hour'].apply(time_category)
        return df
    
    def merge_datasets(self, deliveries_clean, drivers_df, depots_df):
        merged = deliveries_clean.merge(drivers_df, on='driver_id', how='left')
        merged = merged.merge(depots_df, on='depot_id', how='left', suffixes=('', '_depot'))
        return merged
    
    def run_pipeline(self):
        depots, drivers, deliveries = self.load_raw_data()
        deliveries_clean = self.clean_deliveries(deliveries)
        merged = self.merge_datasets(deliveries_clean, drivers, depots)
        os.makedirs(self.processed_path, exist_ok=True)
        deliveries_clean.to_csv(f'{self.processed_path}/deliveries_clean.csv', index=False)
        merged.to_csv(f'{self.processed_path}/deliveries_merged.csv', index=False)
        return merged
