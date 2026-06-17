import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add current directory to path so we can import utils
sys.path.append(os.getcwd())
from utils.helpers import load_and_clean_data
from utils.three_js_template import get_three_js_html
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="AI Cosmic Intelligence Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a dark, space-like theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at top right, #0d1b2a, #0e1117);
    }
    .stSidebar {
        background-color: #1b263b;
    }
    .stMetric {
        background-color: #415a77;
        padding: 15px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #050a14;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Data
@st.cache_data
def get_data():
    return load_and_clean_data()

# Load Models
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('models/rf_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return rf, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

df = get_data()
rf, scaler = load_models()

# Sidebar Navigation
st.sidebar.title("🚀 AI Cosmic Explorer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Explore Planets", "Habitability Predictor", "AI Planet Generator", "3D Space Simulation"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Mission Control:** Inspired by NASA Eyes, enhanced with AI and physics simulations.")

# --- Explore Planets Page ---
if page == "Explore Planets":
    st.title("🔭 Explore Planets")
    st.markdown("Search and filter the NASA Exoplanet dataset to find interesting worlds.")
    
    if df is not None:
        # Search Functionality
        search_query = st.text_input("🔍 Search for a Planet or Star System", "")
        
        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            min_rad = st.slider("Min Planet Radius (R_earth)", 0.0, 20.0, 0.0)
        with c2:
            max_dist = st.slider("Max Orbital Distance (AU)", 0.0, 100.0, 50.0)
        with c3:
            min_temp = st.slider("Min Star Temp (K)", 1000, 10000, 2000)
            
        filtered_df = df[
            (df['planet_name'].str.contains(search_query, case=False) | df['hostname'].str.contains(search_query, case=False)) &
            (df['planet_radius'] >= min_rad) &
            (df['orbit_semi_major_axis'] <= max_dist) &
            (df['star_temp_k'] >= min_temp)
        ]
        
        # Display Table
        st.dataframe(filtered_df[['planet_name', 'hostname', 'planet_radius', 'orbital_period', 'orbit_semi_major_axis', 'star_temp_k', 'habitability_label']],
                     use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Dataset Overview")
        c4, c5 = st.columns(2)
        with c4:
            fig_hist = px.histogram(df, x='habitability_label', title="Habitability Distribution",
                                   color_discrete_sequence=['#00d4ff'], labels={'habitability_label': 'Is Habitable?'})
            st.plotly_chart(fig_hist, width='stretch')
        with c5:
            fig_scatter = px.scatter(df, x='star_temp_k', y='planet_radius', color='habitability_label',
                                   title="Star Temp vs Planet Radius", template="plotly_dark",
                                   labels={'star_temp_k': 'Star Temp (K)', 'planet_radius': 'Planet Radius (R_earth)'},
                                   color_continuous_scale='Bluered')
            st.plotly_chart(fig_scatter, width='stretch')

# --- Habitability Predictor Page ---
elif page == "Habitability Predictor":
    st.title("🧠 Habitability Predictor")
    st.markdown("Enter planetary and stellar features to predict if a world could support life.")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            p_rad = st.slider("Planet Radius (Earth Radii)", 0.1, 20.0, 1.0)
            o_per = st.number_input("Orbital Period (Days)", 0.1, 10000.0, 365.0)
            s_temp = st.slider("Star Temperature (K)", 1000.0, 30000.0, 5778.0)
        with c2:
            o_axis = st.slider("Orbit Semi-Major Axis (AU)", 0.01, 50.0, 1.0)
            s_rad = st.slider("Star Radius (Solar Radii)", 0.1, 100.0, 1.0)
            s_lum = st.number_input("Star Luminosity (Solar Luminosity)", 0.01, 10000.0, 1.0)
            
    if rf is not None and scaler is not None:
        # Prepare Input
        input_data = np.array([[p_rad, o_per, s_temp, o_axis, s_rad, s_lum]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = rf.predict(input_scaled)[0]
        confidence = rf.predict_proba(input_scaled)[0][prediction]
        
        st.markdown("---")
        st.subheader("Results")
        res_c1, res_c2 = st.columns(2)
        
        if prediction == 1:
            res_c1.success("✅ POTENTIALLY HABITABLE")
            st.balloons()
        else:
            res_c1.error("❌ UNLIKELY HABITABLE")
            
        res_c2.metric("AI Confidence Score", f"{confidence:.1%}")
        
        # Explanation
        st.markdown("""
        **How the AI decides:**
        The Random Forest model analyzes multiple features simultaneously, but it prioritizes **Stellar Flux** (the amount of light the planet receives) and the **Planet's Size**. 
        If a planet is too large, it might be a gas giant; if it's too close to its star, it's too hot for liquid water.
        """)

# --- AI Planet Generator Page ---
elif page == "AI Planet Generator":
    st.title("✨ AI Planet Generator")
    st.markdown("Generate a procedurally created exoplanet and test its habitability.")
    
    if st.button("🚀 Generate New Planet"):
        # Randomly generate features
        p_rad = np.random.uniform(0.5, 10.0)
        o_axis = np.random.uniform(0.05, 5.0)
        s_temp = np.random.uniform(2500, 10000)
        s_rad = np.random.uniform(0.5, 2.0)
        s_lum = (s_rad**2) * ((s_temp / 5778)**4)
        # Kepler's 3rd law (simplified): P^2 = a^3 / M
        o_per = np.sqrt(o_axis**3) * 365.25
        
        st.markdown("### 🪐 New Planet Detected!")
        
        # Display in Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Radius", f"{p_rad:.2f} R_e")
        col2.metric("Orbit Distance", f"{o_axis:.2f} AU")
        col3.metric("Star Temp", f"{s_temp:.0f} K")
        
        # Predict Habitability
        if rf is not None and scaler is not None:
            input_data = np.array([[p_rad, o_per, s_temp, o_axis, s_rad, s_lum]])
            input_scaled = scaler.transform(input_data)
            prediction = rf.predict(input_scaled)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success("✅ This planet is predicted to be **HABITABLE**!")
            else:
                st.warning("❌ This planet is likely **UNINHABITABLE**.")
            
            # Interactive 3D Orbit View for this planet
            fig = go.Figure()
            # Star
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=20, color='orange'), name='Host Star'))
            # Orbit
            theta = np.linspace(0, 2*np.pi, 100)
            ox = o_axis * np.cos(theta)
            oy = o_axis * np.sin(theta)
            oz = np.zeros_like(ox)
            fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode='lines', line=dict(color='white', width=1), showlegend=False))
            # Planet
            fig.add_trace(go.Scatter3d(x=[o_axis], y=[0], z=[0], mode='markers', marker=dict(size=8, color='#00d4ff'), name='Planet'))
            
            fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='black'),
                             margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='black')
            st.plotly_chart(fig, width='stretch')

# --- 3D Space Simulation Page ---
elif page == "3D Space Simulation":
    st.title("🌌 3D Space Simulation")
    st.markdown("""
    A high-performance **Three.js** simulation inspired by NASA Eyes. 
    Use your mouse to rotate, zoom, and explore the system in real-time.
    """)
    
    # Selection for system simulation
    system_options = ["Random Generated System"] + sorted(df['hostname'].unique().tolist())[:20] # Top 20 for performance
    selected_system = st.selectbox("Select Star System to Visualize", system_options)
    
    planets_to_sim = []
    
    if selected_system == "Random Generated System":
        n_planets = st.slider("Number of Planets to Generate", 1, 8, 5)
        for i in range(n_planets):
            orbit_radius = (i + 1) * 2 + np.random.uniform(-0.5, 0.5)
            planets_to_sim.append({
                "name": f"Planet {i+1}",
                "radius": np.random.uniform(0.5, 2.5),
                "orbit_radius": orbit_radius,
                "is_habitable": 1.5 < orbit_radius < 4.5 # Simple HZ for random system
            })
    else:
        system_df = df[df['hostname'] == selected_system].sort_values('orbit_semi_major_axis')
        for _, row in system_df.iterrows():
            planets_to_sim.append({
                "name": row['planet_name'],
                "radius": row['planet_radius'],
                "orbit_radius": row['orbit_semi_major_axis'],
                "is_habitable": bool(row['habitability_label'])
            })
            
    # Render Three.js Component
    html_code = get_three_js_html(planets_to_sim)
    components.html(html_code, height=600, scrolling=False)
    
    st.info("✨ **NASA Eyes Insight:** This simulation uses **Kepler's Third Law** for orbital speeds—inner planets move faster than outer ones. Blue/Cyan planets are in the Habitable Zone.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("v1.0 AI Cosmic Explorer | Powered by NASA & AI")
