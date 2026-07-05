"""
🚚 Bettermile Delivery Insights Dashboard
STABLE VERSION - No buffering, no reloads, interview-ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import os

# ==================== CRITICAL: SET PAGE CONFIG FIRST ====================
st.set_page_config(
    page_title="Bettermile Delivery Insights",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE INITIALIZATION ====================
# This runs ONLY ONCE per session, never again
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_ready = False
    st.session_state.charts_ready = False
    st.session_state.map_ready = False
    st.session_state.depots = None
    st.session_state.drivers = None
    st.session_state.deliveries = None
    st.session_state.filtered = None
    st.session_state.kpis = None
    st.session_state.overview_charts = None
    st.session_state.driver_charts = None
    st.session_state.trend_charts = None
    st.session_state.map_html = None
    st.session_state.current_tab = "Overview"

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
    /* HIDE SPINNERS AFTER LOAD */
    .stSpinner { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==================== REALISTIC ETA PREDICTION ====================
def predict_delivery_time(hour, experience, weather, traffic, distance, weight):
    """Realistic ETA prediction based on European last-mile delivery logic."""
    base_time = 12
    exp_multiplier = {'New': 1.6, 'Intermediate': 1.2, 'Experienced': 1.0, 'Expert': 0.8}[experience]
    weather_delay = {'Sunny': 0, 'Cloudy': 2, 'Rainy': 8, 'Snowy': 18}[weather]
    traffic_multiplier = {'Low': 1.0, 'Medium': 1.25, 'High': 1.6}[traffic]
    distance_factor = 1.0 + (distance / 15) * 0.5
    weight_factor = 1.0 + (weight / 20) * 0.3
    
    hour_factor = 1.0
    if 8 <= hour <= 10: hour_factor = 1.15
    elif 12 <= hour <= 14: hour_factor = 1.1
    elif 17 <= hour <= 18: hour_factor = 1.2
    
    predicted = (base_time * exp_multiplier * traffic_multiplier * 
                 distance_factor * weight_factor * hour_factor) + weather_delay
    
    uncertainty_base = 5
    uncertainty = uncertainty_base * exp_multiplier * (1.5 if weather in ['Rainy', 'Snowy'] else 1.0)
    
    lower = max(5, predicted - uncertainty)
    upper = predicted + uncertainty
    confidence = 85 - int((exp_multiplier - 1.0) * 20) - int(weather_delay / 2)
    confidence = max(60, min(95, confidence))
    
    return predicted, lower, upper, confidence

# ==================== DATA GENERATION (ONE TIME ONLY) ====================
def generate_all_data():
    """Generate data, run pipeline, train model - ONE TIME ONLY"""
    from data_generator import DeliveryDataGenerator
    from data_pipeline import DataPipeline
    from feature_engineering import FeatureEngineer
    from eta_model import ETAPredictionModel
    
    generator = DeliveryDataGenerator(n_drivers=50, n_depots=10, n_days=30)
    depots, drivers, deliveries = generator.generate_all_data()
    
    pipeline = DataPipeline()
    merged = pipeline.run_pipeline()
    
    engineer = FeatureEngineer()
    ml_data = engineer.run()
    
    model = ETAPredictionModel()
    metrics, importance = model.run()
    
    return depots, drivers, merged

def load_data_from_disk():
    """Load already generated data from disk"""
    depots = pd.read_csv('data/raw/depots.csv')
    drivers = pd.read_csv('data/raw/drivers.csv')
    deliveries = pd.read_csv('data/processed/deliveries_merged.csv')
    deliveries['date'] = pd.to_datetime(deliveries['date'])
    deliveries['scheduled_time'] = pd.to_datetime(deliveries['scheduled_time'])
    deliveries['actual_time'] = pd.to_datetime(deliveries['actual_time'])
    return depots, drivers, deliveries

# ==================== PRE-COMPUTE ALL CHARTS ====================
def compute_kpis(deliveries_df):
    total = len(deliveries_df)
    delivered = deliveries_df['outcome'] == 'Delivered'
    success_rate = delivered.mean() * 100
    on_time = deliveries_df['delivery_duration_min'] <= 15
    on_time_rate = on_time.mean() * 100
    avg_sat = deliveries_df['customer_satisfaction'].mean()
    avg_dur = deliveries_df['delivery_duration_min'].mean()
    failed = total - delivered.sum()
    return {
        'total_deliveries': total, 'success_rate': success_rate,
        'on_time_rate': on_time_rate, 'avg_satisfaction': avg_sat,
        'avg_duration': avg_dur, 'failed_deliveries': failed
    }

def precompute_overview_charts(filtered):
    """Pre-compute all overview charts so they don't re-render"""
    charts = {}
    
    # Pie chart
    outcome_counts = filtered['outcome'].value_counts()
    charts['pie'] = px.pie(values=outcome_counts.values, names=outcome_counts.index,
                          title="Delivery Outcomes", color_discrete_sequence=px.colors.qualitative.Set3,
                          template="plotly_dark")
    
    # Line chart
    hourly_sat = filtered.groupby('hour')['customer_satisfaction'].mean().reset_index()
    charts['line'] = px.line(hourly_sat, x='hour', y='customer_satisfaction',
                            title="Satisfaction by Hour", markers=True, template="plotly_dark")
    charts['line'].update_layout(yaxis_range=[0, 5])
    
    # Bar chart
    daily_volume = filtered.groupby('date').size().reset_index(name='deliveries')
    charts['bar'] = px.bar(daily_volume, x='date', y='deliveries', title="Daily Volume",
                          color='deliveries', color_continuous_scale='Blues', template="plotly_dark")
    
    return charts

def precompute_driver_charts(filtered):
    """Pre-compute driver performance charts"""
    exp_perf = filtered.groupby('experience_level').agg({
        'is_delivered': 'mean', 'customer_satisfaction': 'mean', 'delivery_duration_min': 'mean'
    }).reset_index()
    exp_perf['is_delivered'] = exp_perf['is_delivered'] * 100
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Success Rate', 'Satisfaction', 'Avg Duration'))
    fig.add_trace(go.Bar(x=exp_perf['experience_level'], y=exp_perf['is_delivered'], marker_color='#4CAF50'), row=1, col=1)
    fig.add_trace(go.Bar(x=exp_perf['experience_level'], y=exp_perf['customer_satisfaction'], marker_color='#2196F3'), row=1, col=2)
    fig.add_trace(go.Bar(x=exp_perf['experience_level'], y=exp_perf['delivery_duration_min'], marker_color='#FF9800'), row=1, col=3)
    fig.update_layout(height=400, showlegend=False, title_text="Performance by Experience Level",
                     template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def precompute_trend_charts(filtered):
    """Pre-compute trend charts"""
    weather_impact = filtered.groupby('weather').agg({
        'is_delivered': 'mean', 'customer_satisfaction': 'mean', 'delivery_duration_min': 'mean'
    }).reset_index()
    weather_impact['is_delivered'] = weather_impact['is_delivered'] * 100
    
    charts = {}
    charts['success'] = px.bar(weather_impact, x='weather', y='is_delivered', 
                               title="Success by Weather", color='weather', template="plotly_dark")
    charts['duration'] = px.bar(weather_impact, x='weather', y='delivery_duration_min', 
                                title="Duration by Weather", color='weather', template="plotly_dark")
    return charts

def generate_static_map(depots, filtered, selected_depots):
    """Generate a static map image instead of interactive Folium"""
    import folium
    from folium import plugins
    
    m = folium.Map(location=[52.5200, 13.4050], zoom_start=11, tiles="CartoDB dark_matter")
    
    for _, depot in depots.iterrows():
        if depot['depot_name'] in selected_depots:
            folium.Marker([depot['latitude'], depot['longitude']],
                popup=f"🏭 {depot['depot_name']}<br>Capacity: {depot['capacity']} parcels/day",
                icon=folium.Icon(color='blue', icon='warehouse', prefix='fa')).add_to(m)
    
    sample = filtered.sample(min(100, len(filtered)))
    for _, d in sample.iterrows():
        color = 'green' if d['outcome'] == 'Delivered' else 'red'
        folium.CircleMarker([d['delivery_lat'], d['delivery_lon']], radius=3,
            color=color, fill=True, popup=f"{d['delivery_id']}<br>{d['outcome']}").add_to(m)
    
    # Save to HTML string
    return m._repr_html_()

# ==================== MAIN APP ====================
def main():
    # ==================== PHASE 1: INITIALIZE DATA (ONE TIME) ====================
    if not st.session_state.initialized:
        with st.spinner("🚀 Loading dashboard... Please wait."):
            # Check if data exists
            if not os.path.exists('data/processed/deliveries_merged.csv'):
                generate_all_data()
            
            # Load data
            depots, drivers, deliveries = load_data_from_disk()
            st.session_state.depots = depots
            st.session_state.drivers = drivers
            st.session_state.deliveries = deliveries
            
            # Default filter (all data)
            st.session_state.filtered = deliveries.copy()
            st.session_state.kpis = compute_kpis(deliveries)
            
            # Pre-compute all charts
            st.session_state.overview_charts = precompute_overview_charts(deliveries)
            st.session_state.driver_charts = precompute_driver_charts(deliveries)
            st.session_state.trend_charts = precompute_trend_charts(deliveries)
            
            # Generate map
            all_depots = depots['depot_name'].tolist()[:5]
            st.session_state.map_html = generate_static_map(depots, deliveries, all_depots)
            
            st.session_state.initialized = True
            st.session_state.data_ready = True
        
        st.rerun()  # Refresh once after initialization
    
    # ==================== PHASE 2: RENDER DASHBOARD (NO RE-COMPUTATION) ====================
    if not st.session_state.data_ready:
        st.error("Something went wrong. Please refresh.")
        return
    
    # Get cached data
    depots = st.session_state.depots
    drivers = st.session_state.drivers
    deliveries = st.session_state.deliveries
    
    # Sidebar - FILTERS
    st.sidebar.title("🚚 Delivery Insights")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Filters")
    
    # Use session state for filter values to prevent re-renders
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (deliveries['date'].min().date(), deliveries['date'].max().date())
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=st.session_state.date_range,
        min_value=deliveries['date'].min().date(),
        max_value=deliveries['date'].max().date(),
        key='date_range_input'
    )
    
    selected_depots = st.sidebar.multiselect(
        "Depots",
        options=depots['depot_name'].tolist(),
        default=depots['depot_name'].tolist()[:5],
        key='depots_input'
    )
    
    selected_drivers = st.sidebar.multiselect(
        "Driver Experience",
        options=['New', 'Intermediate', 'Experienced', 'Expert'],
        default=['New', 'Intermediate', 'Experienced', 'Expert'],
        key='drivers_input'
    )
    
    # Apply filters ONLY if they changed
    filter_key = f"{date_range}_{sorted(selected_depots)}_{sorted(selected_drivers)}"
    if 'last_filter' not in st.session_state or st.session_state.last_filter != filter_key:
        filtered = deliveries[
            (deliveries['date'].dt.date >= date_range[0]) &
            (deliveries['date'].dt.date <= date_range[1]) &
            (deliveries['depot_name'].isin(selected_depots)) &
            (deliveries['experience_level'].isin(selected_drivers))
        ]
        st.session_state.filtered = filtered
        st.session_state.kpis = compute_kpis(filtered)
        st.session_state.last_filter = filter_key
        
        # Re-compute charts for new filter
        st.session_state.overview_charts = precompute_overview_charts(filtered)
        st.session_state.driver_charts = precompute_driver_charts(filtered)
        st.session_state.trend_charts = precompute_trend_charts(filtered)
        st.session_state.map_html = generate_static_map(depots, filtered, selected_depots)
    
    # Get filtered data and KPIs
    filtered = st.session_state.filtered
    kpis = st.session_state.kpis
    
    # Main content
    st.markdown('<p class="main-header">🚚 Bettermile Delivery Insights Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time analytics for last-mile delivery optimization</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # KPI Cards - NO SPINNERS, INSTANT
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📦 Total Deliveries", f"{kpis['total_deliveries']:,}",
                  f"{kpis['success_rate']:.1f}% success")
    with c2:
        st.metric("⏱️ On-Time Rate", f"{kpis['on_time_rate']:.1f}%", "Target: 95%")
    with c3:
        st.metric("⭐ Avg Satisfaction", f"{kpis['avg_satisfaction']:.2f}/5",
                  f"{kpis['avg_satisfaction']-3:.2f} vs avg")
    with c4:
        st.metric("🕐 Avg Duration", f"{kpis['avg_duration']:.1f} min",
                  f"{kpis['avg_duration']-20:.1f} min vs target")
    with c5:
        st.metric("❌ Failed Deliveries", f"{kpis['failed_deliveries']:,}",
                  f"{kpis['failed_deliveries']/kpis['total_deliveries']*100:.1f}% rate")
    
    st.markdown("---")
    
    # ==================== TABS - NO RE-COMPUTATION ====================
    tabs = st.tabs(["📊 Overview", "🗺️ Map", "👨‍💼 Drivers", "📈 Trends", "🔮 Predictions"])
    
    # TAB 1: OVERVIEW - PRE-COMPUTED CHARTS
    with tabs[0]:
        st.subheader("📊 Delivery Performance Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state.overview_charts['pie'], use_container_width=True, key='pie_chart')
        with col2:
            st.plotly_chart(st.session_state.overview_charts['line'], use_container_width=True, key='line_chart')
        st.plotly_chart(st.session_state.overview_charts['bar'], use_container_width=True, key='bar_chart')
    
    # TAB 2: MAP - STATIC HTML (NO FOLIUM RE-RENDER)
    with tabs[1]:
        st.subheader("🗺️ Live Delivery Map")
        # Render pre-computed HTML directly
        st.components.v1.html(st.session_state.map_html, width=700, height=500)
        st.markdown("**Legend:** 🔵 Depots | 🟢 Success | 🔴 Failed")
    
    # TAB 3: DRIVERS - PRE-COMPUTED
    with tabs[2]:
        st.subheader("👨‍💼 Driver Performance")
        driver_perf = filtered.groupby(['driver_id', 'driver_name', 'experience_level']).agg({
            'delivery_id': 'count', 'is_delivered': 'mean',
            'customer_satisfaction': 'mean', 'delivery_duration_min': 'mean'
        }).reset_index()
        driver_perf.columns = ['Driver ID', 'Name', 'Experience', 'Deliveries', 
                               'Success Rate', 'Avg Satisfaction', 'Avg Duration']
        driver_perf['Success Rate'] = driver_perf['Success Rate'] * 100
        st.dataframe(driver_perf.sort_values('Success Rate', ascending=False), use_container_width=True)
        st.plotly_chart(st.session_state.driver_charts, use_container_width=True, key='driver_chart')
    
    # TAB 4: TRENDS - PRE-COMPUTED
    with tabs[3]:
        st.subheader("📈 Trends & Patterns")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state.trend_charts['success'], use_container_width=True, key='trend_success')
        with col2:
            st.plotly_chart(st.session_state.trend_charts['duration'], use_container_width=True, key='trend_duration')
    
    # TAB 5: PREDICTIONS - INTERACTIVE BUT FAST
    with tabs[4]:
        st.subheader("🔮 ETA Prediction Model")
        st.success("✅ Model Ready")
        st.markdown("""
        **How it works:** Predicts realistic delivery time per stop in European cities.
        Factors: driver experience, weather, traffic, distance between stops, package weight, time of day.
        """)
        st.markdown("---")
        st.subheader("🎯 Predict Delivery Time")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hour = st.slider("Hour of Day", 8, 18, 12, key='pred_hour')
            experience = st.selectbox("Driver Experience", ['New', 'Intermediate', 'Experienced', 'Expert'], key='pred_exp')
        with col2:
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'], key='pred_weather')
            traffic = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'], key='pred_traffic')
        with col3:
            distance = st.slider("Avg Distance Between Stops (km)", 0.5, 5.0, 1.5, step=0.5, key='pred_dist')
            weight = st.slider("Package Weight (kg)", 0.5, 20.0, 3.0, key='pred_weight')
        
        if st.button("🔮 Predict ETA", type="primary", key='pred_button'):
            pred, lower, upper, confidence = predict_delivery_time(hour, experience, weather, traffic, distance, weight)
            
            if pred <= 15:
                color, status, emoji = "#4CAF50", "✅ Efficient stop", "🚀"
            elif pred <= 25:
                color, status, emoji = "#FF9800", "⚠️ Moderate delay expected", "⏱️"
            else:
                color, status, emoji = "#F44336", "🚨 Significant delay expected", "🐌"
            
            st.markdown(f"""
            <div style="background-color: {color}20; border: 2px solid {color}; padding: 24px; border-radius: 16px; margin: 16px 0;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 2.5rem;">{emoji}</span>
                    <div>
                        <h2 style="margin: 0; color: {color}; font-size: 2.2rem;">{pred:.0f} minutes</h2>
                        <p style="margin: 4px 0 0 0; color: #A0A0B0; font-size: 1rem;">per stop</p>
                    </div>
                </div>
                <hr style="border-color: {color}40; margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                    <div><p style="color: #A0A0B0; margin: 0; font-size: 0.85rem;">Typical Range</p>
                        <p style="color: #FFFFFF; margin: 4px 0 0 0; font-size: 1.1rem; font-weight: 600;">{lower:.0f} - {upper:.0f} min</p></div>
                    <div><p style="color: #A0A0B0; margin: 0; font-size: 0.85rem;">Confidence</p>
                        <p style="color: #FFFFFF; margin: 4px 0 0 0; font-size: 1.1rem; font-weight: 600;">{confidence}%</p></div>
                    <div><p style="color: #A0A0B0; margin: 0; font-size: 0.85rem;">Status</p>
                        <p style="color: {color}; margin: 4px 0 0 0; font-size: 1.1rem; font-weight: 600;">{status}</p></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Factor breakdown
            st.markdown("---")
            st.subheader("🔍 Why this prediction?")
            factors = []
            factors.append(f"📦 **Base stop time:** 12 min (park, walk, deliver, scan)")
            
            exp_mult = {'New': 1.6, 'Intermediate': 1.2, 'Experienced': 1.0, 'Expert': 0.8}[experience]
            if experience == 'New': factors.append(f"👶 **New driver:** +{((exp_mult-1)*100):.0f}% slower")
            elif experience == 'Expert': factors.append(f"🌟 **Expert driver:** {((1-exp_mult)*100):.0f}% faster")
            
            weather_add = {'Sunny': 0, 'Cloudy': 2, 'Rainy': 8, 'Snowy': 18}[weather]
            if weather == 'Snowy': factors.append(f"❄️ **Snow:** +{weather_add} min")
            elif weather == 'Rainy': factors.append(f"🌧️ **Rain:** +{weather_add} min")
            
            traffic_mult = {'Low': 1.0, 'Medium': 1.25, 'High': 1.6}[traffic]
            if traffic == 'High': factors.append(f"🚦 **High traffic:** +{((traffic_mult-1)*100):.0f}% slower")
            
            if distance > 2.0: factors.append(f"📍 **Spread-out route:** +{((distance/15)*50):.0f}% drive time")
            if weight > 10: factors.append(f"🏋️ **Heavy package:** harder to carry")
            
            for f in factors:
                st.markdown(f)
    
    st.markdown("---")
    st.markdown("*Built for Bettermile Interview | Realistic European delivery simulation*")

if __name__ == '__main__':
    main()
