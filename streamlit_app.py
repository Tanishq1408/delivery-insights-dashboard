"""
🚚 Bettermile Delivery Insights Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import os

# Page config
st.set_page_config(page_title="Bettermile Delivery Insights", page_icon="🚚", layout="wide")

# ==================== DARK THEME CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1E1E2E !important;
        border: 1px solid #2D2D44 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    div[data-testid="stMetric"] label { color: #A0A0B0 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div { color: #FFFFFF !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: #4CAF50 !important; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; }
    section[data-testid="stSidebar"] * { color: #E0E0E0 !important; }
    .stButton>button {
        background-color: #2D2D44 !important; color: #FFFFFF !important;
        border: 1px solid #4A4A6A !important; border-radius: 8px !important;
    }
    .stButton>button:hover { background-color: #3D3D5C !important; }
    h1, h2, h3 { color: #FFFFFF !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1E1E2E !important; }
    .stTabs [data-baseweb="tab"] { color: #A0A0B0 !important; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; background-color: #2D2D44 !important; }
    .stDataFrame { background-color: #1E1E2E !important; }
    .stSelectbox, .stSlider, .stDateInput { background-color: #1E1E2E !important; }
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FFFFFF; }
    .subtitle { color: #A0A0B0; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE FOR PERSISTENCE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.depots = None
    st.session_state.drivers = None
    st.session_state.deliveries = None

# ==================== REALISTIC ETA PREDICTION FUNCTION ====================
def predict_delivery_time(hour, experience, weather, traffic, distance, weight):
    """
    Realistic ETA prediction based on European last-mile delivery logic.
    Returns: predicted_minutes, lower_bound, upper_bound, confidence_pct
    """
    # Load model and scaler
    try:
        model = joblib.load('models/eta_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        use_ml = True
    except:
        use_ml = False  # Fallback to rule-based if model not available
    
    # Base delivery time (realistic European context)
    # Average stop time: parking, walk, delivery, scan, back to van
    base_time = 12  # minutes base per stop
    
    # Experience factor (new drivers take longer)
    exp_multiplier = {
        'New': 1.6,        # 60% slower (learning routes, parking, buildings)
        'Intermediate': 1.2,  # 20% slower
        'Experienced': 1.0,     # Baseline
        'Expert': 0.8       # 20% faster (knows shortcuts, parking spots)
    }[experience]
    
    # Weather impact (European weather is unpredictable)
    weather_delay = {
        'Sunny': 0,
        'Cloudy': 2,       # Slight delay, harder to find house numbers
        'Rainy': 8,        # Slower walking, wet packages need care
        'Snowy': 18        # Slippery, traffic chaos, parking nightmare
    }[weather]
    
    # Traffic impact (Berlin rush hour is brutal)
    traffic_multiplier = {
        'Low': 1.0,
        'Medium': 1.25,    # 25% slower
        'High': 1.6        # 60% slower (stuck in traffic, circling for parking)
    }[traffic]
    
    # Distance factor (between stops, not from depot)
    # In Europe, stops are close but parking is hard
    distance_factor = 1.0 + (distance / 15) * 0.5  # Up to 50% more for spread-out routes
    
    # Weight factor (heavy packages slow you down)
    weight_factor = 1.0 + (weight / 20) * 0.3  # Up to 30% more for heavy stuff
    
    # Time of day factor (lunch breaks, school zones, etc.)
    hour_factor = 1.0
    if 8 <= hour <= 10:    # Morning rush + school zones
        hour_factor = 1.15
    elif 12 <= hour <= 14:  # Lunch break chaos, people not home
        hour_factor = 1.1
    elif 17 <= hour <= 18:  # Evening rush, people coming home
        hour_factor = 1.2
    
    # Calculate realistic prediction
    predicted = (base_time * exp_multiplier * traffic_multiplier * 
                 distance_factor * weight_factor *
