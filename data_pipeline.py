"""
Data Pipeline - Cleans and prepares raw delivery data for analysis
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
        """Load raw CSV files"""
        depots = pd.read_csv(f'{self.raw_path}/depots.csv')
        drivers = pd.read_csv(f'{self.raw_path}/drivers.csv')
        deliveries = pd.read_csv(f'{self.raw_path}/deliveries.csv')
        return depots, drivers, deliveries
    
    def clean_deliveries(self, deliveries_df):
        """Clean and enrich delivery data"""
        df = deliveries_df.copy()
        
        # Convert datetime columns
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        df['actual_time'] = pd.to_datetime(df['actual_time'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate delivery duration (minutes)
        df['delivery_duration_min'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
        
        # Calculate on-time flag (within 15 minutes of scheduled time)
        df['is_on_time'] = df['delivery_duration_min'] <= 15
        
        # Calculate hour of day
        df['hour'] = df['scheduled_time'].dt.hour
        
        # Calculate day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        
        # Is peak season (middle of month)
        df['is_peak_season'] = df['date'].dt.day.between(10, 20)
        
        # Binary outcome
        df['is_delivered'] = df['outcome'] == 'Delivered'
        
        # Delivery time category
        def time_category(hour):
            if hour <= 10:
                return 'Morning'
            elif hour <= 14:
                return 'Midday'
            elif hour <= 17:
                return 'Afternoon'
            else:
                return 'Evening'
        
        df['time_category'] = df['hour'].apply(time_category)
        
        return df
    
    def merge_datasets(self, deliveries_clean, drivers_df, depots_df):
        """Merge all datasets for analysis"""
        # First merge deliveries with drivers on driver_id
        merged = deliveries_clean.merge(drivers_df, on='driver_id', how='left')
        
        # Check columns in depots_df
        print(f"Depots columns: {depots_df.columns.tolist()}")
        print(f"Merged columns before depot merge: {merged.columns.tolist()}")
        
        # Ensure depot_id exists in both dataframes
        if 'depot_id' not in depots_df.columns:
            raise KeyError(f"depot_id not found in depots_df. Available columns: {depots_df.columns.tolist()}")
        
        if 'depot_id' not in merged.columns:
            raise KeyError(f"depot_id not found in merged dataframe. Available columns: {merged.columns.tolist()}")
        
        # Merge with depots on depot_id
        merged = merged.merge(depots_df, on='depot_id', how='left', suffixes=('', '_depot'))
        
        return merged
    
    def run_pipeline(self):
        """Execute full pipeline"""
        print("Loading raw data...")
        depots, drivers, deliveries = self.load_raw_data()
        
        print("Cleaning deliveries...")
        deliveries_clean = self.clean_deliveries(deliveries)
        
        print("Merging datasets...")
        merged = self.merge_datasets(deliveries_clean, drivers, depots)
        
        # Save processed data
        deliveries_clean.to_csv(f'{self.processed_path}/deliveries_clean.csv', index=False)
        merged.to_csv(f'{self.processed_path}/deliveries_merged.csv', index=False)
        
        print(f"Pipeline complete! Processed {len(merged)} records")
        print(f"Saved to {self.processed_path}/")
        
        return merged

if __name__ == '__main__':
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()
