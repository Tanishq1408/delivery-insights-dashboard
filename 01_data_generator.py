"""
Delivery Data Generator
Creates realistic simulated delivery data for last-mile logistics analysis.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker('de_DE')  # German addresses since Bettermile is Berlin-based

class DeliveryDataGenerator:
    def __init__(self, n_drivers=50, n_depots=10, n_days=30, seed=42):
        self.n_drivers = n_drivers
        self.n_depots = n_depots
        self.n_days = n_days
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_depots(self):
        """Create depot locations around Berlin"""
        depots = []
        berlin_center = (52.5200, 13.4050)  # Berlin coordinates

        for i in range(self.n_depots):
            # Random location within ~30km of Berlin
            lat = berlin_center[0] + random.uniform(-0.25, 0.25)
            lon = berlin_center[1] + random.uniform(-0.25, 0.25)

            depots.append({
                'depot_id': f'DEP_{i:03d}',
                'depot_name': f'Berlin Depot {i+1}',
                'latitude': lat,
                'longitude': lon,
                'capacity': random.randint(100, 500),
                'region': random.choice(['North', 'South', 'East', 'West', 'Central'])
            })

        return pd.DataFrame(depots)

    def generate_drivers(self, depots_df):
        """Create driver profiles"""
        drivers = []
        experience_levels = ['New', 'Intermediate', 'Experienced', 'Expert']
        experience_weights = [0.2, 0.3, 0.35, 0.15]

        for i in range(self.n_drivers):
            experience = random.choices(experience_levels, weights=experience_weights)[0]

            if experience == 'New':
                base_stops = random.randint(25, 35)
                accuracy = random.uniform(0.75, 0.85)
            elif experience == 'Intermediate':
                base_stops = random.randint(35, 45)
                accuracy = random.uniform(0.85, 0.90)
            elif experience == 'Experienced':
                base_stops = random.randint(45, 55)
                accuracy = random.uniform(0.90, 0.95)
            else:
                base_stops = random.randint(55, 65)
                accuracy = random.uniform(0.95, 0.98)

            drivers.append({
                'driver_id': f'DRV_{i:03d}',
                'driver_name': fake.name(),
                'depot_id': random.choice(depots_df['depot_id'].tolist()),
                'experience_level': experience,
                'base_stops_per_day': base_stops,
                'delivery_accuracy': accuracy,
                'vehicle_type': random.choice(['Van', 'Truck', 'E-Bike']),
                'hire_date': fake.date_between(start_date='-3y', end_date='today')
            })

        return pd.DataFrame(drivers)

    def generate_deliveries(self, depots_df, drivers_df):
        """Create delivery records"""
        deliveries = []
        start_date = datetime(2025, 6, 1)

        for day in range(self.n_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            weekend_multiplier = 0.6 if is_weekend else 1.0
            peak_multiplier = 1.3 if 10 <= current_date.day <= 20 else 1.0

            for _, depot in depots_df.iterrows():
                depot_drivers = drivers_df[drivers_df['depot_id'] == depot['depot_id']]

                for _, driver in depot_drivers.iterrows():
                    n_deliveries = int(
                        driver['base_stops_per_day'] * 
                        weekend_multiplier * 
                        peak_multiplier * 
                        random.uniform(0.8, 1.2)
                    )

                    for stop_num in range(n_deliveries):
                        delivery_lat = depot['latitude'] + random.uniform(-0.1, 0.1)
                        delivery_lon = depot['longitude'] + random.uniform(-0.1, 0.1)

                        hour = random.randint(8, 17)
                        minute = random.randint(0, 59)
                        scheduled_time = current_date.replace(hour=hour, minute=minute)

                        weather = random.choice(['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
                        weather_delay = {
                            'Sunny': 0,
                            'Cloudy': random.uniform(0, 5),
                            'Rainy': random.uniform(5, 15),
                            'Snowy': random.uniform(15, 30)
                        }[weather]

                        if hour in [8, 9, 17, 18]:
                            traffic_delay = random.uniform(5, 20)
                        else:
                            traffic_delay = random.uniform(0, 10)

                        base_service_time = random.uniform(3, 8)
                        total_delay = weather_delay + traffic_delay
                        actual_time = scheduled_time + timedelta(minutes=total_delay + base_service_time)

                        success_prob = driver['delivery_accuracy']
                        if weather == 'Rainy':
                            success_prob -= 0.05
                        if is_weekend:
                            success_prob += 0.05

                        outcome = random.choices(
                            ['Delivered', 'Failed - Not Home', 'Failed - Wrong Address', 'Rescheduled'],
                            weights=[success_prob, (1-success_prob)*0.5, (1-success_prob)*0.3, (1-success_prob)*0.2]
                        )[0]

                        if actual_time <= scheduled_time + timedelta(minutes=15):
                            satisfaction = random.uniform(4.0, 5.0)
                        elif actual_time <= scheduled_time + timedelta(minutes=30):
                            satisfaction = random.uniform(3.0, 4.5)
                        else:
                            satisfaction = random.uniform(1.0, 3.5)

                        if outcome != 'Delivered':
                            satisfaction = random.uniform(1.0, 2.5)

                        deliveries.append({
                            'delivery_id': f'DEL_{len(deliveries):06d}',
                            'date': current_date.date(),
                            'depot_id': depot['depot_id'],
                            'driver_id': driver['driver_id'],
                            'stop_number': stop_num + 1,
                            'scheduled_time': scheduled_time,
                            'actual_time': actual_time,
                            'delivery_lat': delivery_lat,
                            'delivery_lon': delivery_lon,
                            'weather': weather,
                            'traffic_level': 'High' if hour in [8, 9, 17, 18] else 'Medium' if hour in [10, 11, 14, 15, 16] else 'Low',
                            'outcome': outcome,
                            'customer_satisfaction': round(satisfaction, 1),
                            'distance_from_depot_km': random.uniform(1, 15),
                            'parcel_weight_kg': random.uniform(0.5, 20),
                            'is_express': random.choice([True, False]),
                            'is_weekend': is_weekend
                        })

        return pd.DataFrame(deliveries)

    def generate_all_data(self):
        """Generate complete dataset"""
        print("🏭 Generating depots...")
        depots = self.generate_depots()

        print("👨‍💼 Generating drivers...")
        drivers = self.generate_drivers(depots)

        print("📦 Generating deliveries...")
        deliveries = self.generate_deliveries(depots, drivers)

        os.makedirs('data/raw', exist_ok=True)
        depots.to_csv('data/raw/depots.csv', index=False)
        drivers.to_csv('data/raw/drivers.csv', index=False)
        deliveries.to_csv('data/raw/deliveries.csv', index=False)

        print(f"✅ Generated {len(depots)} depots, {len(drivers)} drivers, {len(deliveries)} deliveries")
        print(f"📁 Data saved to data/raw/")

        return depots, drivers, deliveries

if __name__ == '__main__':
    generator = DeliveryDataGenerator(n_drivers=50, n_depots=10, n_days=30)
    depots, drivers, deliveries = generator.generate_all_data()