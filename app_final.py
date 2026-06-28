
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# --- Imports ---
try:
    from utils.helpers import load_and_clean_data
    from utils.three_js_template import get_three_js_html
    from src.train_model import create_habitability_label, get_trained_models
    from src.ai_pipeline import predict_with_all_models, rank_planets_by_habitability, export_ranked_results
    from src.xai_explanations import generate_habitability_explanation
    import streamlit.components.v1 as components
except ImportError as e:
    print(f"Import warning: {e}")


# --- Page Config ---
st.set_page_config(
    page_title="AI Cosmic Intelligence - Research Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NASA-inspired Dark Theme CSS ---
st.markdown("""
    &lt;style&gt;
    .main { background-color: #0a0e17; color: #e0e6ed; }
    .stApp { background: radial-gradient(circle at 10% 0%, #0f172a 0%, #0a0e17 40%, #050810 100%); }
    .stSidebar { background: linear-gradient(180deg, #0f172a 0%, #0a0e17 100%); border-right: 1px solid #1e293b; }
    .stMetric { background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border-left: 3px solid #00d4ff; }
    h1, h2, h3, h4 { color: #00d4ff; font-family: 'Segoe UI', sans-serif; }
    .stButton&gt;button { 
        background: linear-gradient(90deg, #00d4ff, #0ea5e9); 
        color: #0a0e17; 
        font-weight: 700; 
        border-radius: 8px; 
        border: none;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    .stButton&gt;button:hover {
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
        transform: translateY(-1px);
    }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .css-1d391kg { padding-top: 1rem; }
    &lt;/style&gt;
    """, unsafe_allow_html=True)


# --- Data/Model Loading (Cached) ---
@st.cache_data
def get_data():
    try:
        df = load_and_clean_data()
        return df
    except Exception as e:
        st.error(f"Data Load Error: {str(e)}")
        return None


@st.cache_resource
def get_trained_models_cached(df):
    try:
        rf, dnn, pinn, scaler, X_test, y_test = get_trained_models(df)
        return rf, dnn, pinn, scaler, X_test, y_test
    except Exception as e:
        st.error(f"Model Training/Load Error: {str(e)}")
        try:
            rf = joblib.load('models/rf_model.joblib')
            scaler = joblib.load('models/scaler.joblib')
            return rf, None, None, scaler, None, None
        except Exception as e2:
            st.error(f"Fallback Model Error: {e2}")
            return None, None, None, None, None, None


# --- Reports Generators ---
def save_report(report_content, filename):
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    return filepath


def generate_markdown_report(df, title):
    report = f"# {title}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"## Dataset Summary\n- Total Planets: {len(df)}\n"
    report += f"- Habitable Labeled: {df['habitability_label'].sum()}\n"
    if 'overall_habitability_score' in df.columns:
        top5 = df.nlargest(5, 'overall_habitability_score')
        report += f"\n## Top 5 Habitable Planets\n"
        for _, row in top5.iterrows():
            report += f"- {row['planet_name']} ({row['hostname']}): {row['overall_habitability_score']:.2%}\n"
    return report


# --- Main App ---
def main():
    # Load data/models
    df = get_data()
    if df is None:
        st.stop()
    rf, dnn, pinn, scaler, X_test, y_test = get_trained_models_cached(df)

    # Sidebar Navigation
    st.sidebar.title("🚀 AI Cosmic Intelligence")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "ℹ️ About Project", "📊 Dataset Explorer", "🧠 AI Prediction",
         "🏆 Planet Ranking", "🌌 3D Spatial Explorer", "⚡ Simulation",
         "📈 Research Analytics", "🔬 Model Comparison", "⚙️ Settings", "📤 Export Results"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **AI Research Platform**: for exoplanet habitability analysis.")


    # --- 1. HOME ---
    if page == "🏠 Home":
        st.title("🌌 AI-Driven Spatial Analysis of Exoplanetary Systems")
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Mission Overview")
            st.markdown("""
            Welcome to a research platform for exploring exoplanetary habitability using AI and physics-based simulations!
            
            **Key Features**:
            - AI model ensemble for predictions
            - Interactive 3D visualization
            - Automated research reports
            - Model benchmarking & comparison
            """)
        with col2:
            st.image("plots/space_simulation.gif", use_container_width=True)
        
        st.divider()
        st.subheader("Quick Statistics")
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Planets Analyzed", f"{len(df)}")
        qc2.metric("Star Systems", f"{df['hostname'].nunique()}")
        qc3.metric("Habitable Candidates", f"{df['habitability_label'].sum()}")
        avg_rad = df['planet_radius'].mean()
        qc4.metric("Avg Planet Radius", f"{avg_rad:.2f} R_earth")


    # --- 2. ABOUT PROJECT ---
    elif page == "ℹ️ About Project":
        st.title("ℹ️ About the Project")
        st.divider()
        st.markdown("""
        ## Project Goal
        To develop an AI-powered research platform for analyzing exoplanetary systems and predicting habitability.
        
        ## AI Models Used
        - **Random Forest Classifier**: For baseline predictions
        - **Bayesian DNN**: With uncertainty estimation
        - **Physics-Informed NN**: Incorporating orbital mechanics
        
        ## Data Source
        NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/)
        
        ## Technologies
        - Python, Streamlit, Plotly
        - PyTorch, Scikit-learn
        - Three.js (for 3D visualization)
        """)


    # --- 3. DATASET EXPLORER ---
    elif page == "📊 Dataset Explorer":
        st.title("📊 Dataset Explorer")
        st.divider()
        
        with st.expander("🔍 Search & Filter"):
            search = st.text_input("Search by planet or star system:", "")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_rad = st.slider("Min Radius (R_earth)", 0.0, 20.0, 0.0)
                max_rad = st.slider("Max Radius (R_earth)", 0.0, 20.0, 20.0)
            with col2:
                min_per = st.number_input("Min Period (days)", 0.0, 100000.0, 0.0)
                max_per = st.number_input("Max Period (days)", 0.0, 100000.0, 100000.0)
            with col3:
                min_temp = st.slider("Min Star Temp (K)", 1000, 40000, 1000)
                max_temp = st.slider("Max Star Temp (K)", 1000, 40000, 40000)
        
        filtered = df[
            (df['planet_name'].str.contains(search, case=False, na=False) | 
             df['hostname'].str.contains(search, case=False, na=False)) &
            (df['planet_radius'].between(min_rad, max_rad)) &
            (df['orbital_period'].between(min_per, max_per)) &
            (df['star_temp_k'].between(min_temp, max_temp))
        ]
        
        st.subheader(f"Results ({len(filtered)} planets)")
        st.dataframe(filtered, use_container_width=True)
        
        st.divider()
        st.subheader("Visualizations")
        vc1, vc2 = st.columns(2)
        with vc1:
            fig1 = px.histogram(filtered, x='planet_radius', title="Planet Radius Distribution",
                                color='habitability_label', template='plotly_dark')
            st.plotly_chart(fig1, use_container_width=True)
        with vc2:
            fig2 = px.scatter(filtered, x='orbit_semi_major_axis', y='star_temp_k',
                              color='habitability_score' if 'habitability_score' in filtered.columns else 'habitability_label',
                              title="Orbital Distance vs Star Temperature",
                              template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)


    # --- 4. AI PREDICTION ---
    elif page == "🧠 AI Prediction":
        st.title("🧠 AI Habitability Prediction")
        st.divider()
        
        # Planet selector
        planet = st.selectbox("Choose a planet:", sorted(df['planet_name'].tolist()))
        row = df[df['planet_name'] == planet].iloc[0]
        
        st.subheader(f"🪐 {planet} ({row['hostname']})")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.write("**Planet Properties**")
            st.metric("Radius", f"{row['planet_radius']:.2f} R_earth")
            st.metric("Orbit Period", f"{row['orbital_period']:.1f} days")
            st.metric("Orbit Distance", f"{row['orbit_semi_major_axis']:.2f} AU")
        with pc2:
            st.write("**Star Properties**")
            st.metric("Temperature", f"{row['star_temp_k']:.0f} K")
            st.metric("Luminosity", f"{row['star_luminosity']:.2f} L_sun")
            st.metric("Radius", f"{row['star_radius_solar']:.2f} R_sun")
        
        # Run prediction
        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Running models..."):
                if rf and scaler:
                    X = np.array([[
                        row['planet_radius'], row['orbital_period'], row['star_temp_k'],
                        row['orbit_semi_major_axis'], row['star_radius_solar'], row['star_luminosity']
                    ]])
                    results = predict_with_all_models(rf, dnn, pinn, scaler, X)
                    
                    st.divider()
                    st.subheader("🤖 AI Model Predictions")
                    mc1, mc2, mc3 = st.columns(3)
                    
                    with mc1:
                        st.markdown("#### 🌲 Random Forest")
                        rf_pred = "✅ Habitable" if results['rf_prediction'][0] == 1 else "❌ Not Habitable"
                        st.write(rf_pred)
                        st.metric("Confidence", f"{results['rf_confidence'][0]:.1%}")
                    
                    with mc2:
                        st.markdown("#### 🧱 Bayesian DNN")
                        dnn_pred = "✅ Habitable" if results['dnn_prediction'][0] == 1 else "❌ Not Habitable"
                        st.write(dnn_pred)
                        st.metric("Confidence", f"{results['dnn_confidence'][0]:.1%}")
                        st.metric("Uncertainty", f"{results['dnn_uncertainty'][0]:.1%}")
                    
                    with mc3:
                        st.markdown("#### ⚡ Physics-Informed NN")
                        pinn_pred = "✅ Habitable" if results['pinn_prediction'][0] == 1 else "❌ Not Habitable"
                        st.write(pinn_pred)
                        st.metric("Confidence", f"{results['pinn_confidence'][0]:.1%}")
                    
                    st.divider()
                    st.subheader("🎯 Overall Habitability")
                    overall_col, conf_col = st.columns(2)
                    overall_col.metric("Score", f"{results['overall_habitability_score'][0]:.1%}")
                    conf_col.metric("AI Confidence", f"{results['prediction_confidence'][0]:.1%}")
                    st.progress(results['overall_habitability_score'][0])
                    
                    st.divider()
                    st.subheader("📜 Explanation")
                    explanation, _, _ = generate_habitability_explanation(row)
                    st.markdown(explanation)


    # --- 5. PLANET RANKING ---
    elif page == "🏆 Planet Ranking":
        st.title("🏆 Planet Ranking")
        st.divider()
        
        if st.button("🔄 Run Full Ranking"):
            with st.spinner("Ranking planets..."):
                features = ['planet_radius', 'orbital_period', 'star_temp_k', 
                           'orbit_semi_major_axis', 'star_radius_solar', 'star_luminosity']
                X = df[features].values
                results = predict_with_all_models(rf, dnn, pinn, scaler, X)
                df_ranked = rank_planets_by_habitability(df, results)
                
                # Show top planets
                top_20 = df_ranked.nlargest(20, 'overall_habitability_score')
                st.dataframe(top_20, use_container_width=True)
                
                # Save results
                output_file = export_ranked_results(df_ranked)
                st.success(f"✅ Results saved to {output_file}")
                
                # Download button
                csv = df_ranked.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Ranked CSV", csv, "ranked_planets.csv", "text/csv")
        else:
            st.info("Click 'Run Full Ranking' to generate habitability scores.")


    # --- 6. 3D SPATIAL EXPLORER ---
    elif page == "🌌 3D Spatial Explorer":
        st.title("🌌 3D Spatial Explorer")
        st.divider()
        
        system_options = ["Random System"] + sorted(df['hostname'].unique().tolist())[:30]
        selected_system = st.selectbox("Select System:", system_options)
        
        planets_to_sim = []
        star_info = {"name": selected_system, "temp": 5778, "luminosity": 1.0}
        
        if selected_system == "Random System":
            n = st.slider("Planets to Generate", 1, 10, 5)
            for i in range(n):
                r = (i+1)*2 + np.random.randn()*0.3
                planets_to_sim.append({
                    "name": f"Planet {i+1}",
                    "radius": np.random.uniform(0.8, 3.0),
                    "orbit_radius": r,
                    "is_habitable": 1.2 < r < 3.0
                })
        else:
            system_df = df[df['hostname'] == selected_system].sort_values('orbit_semi_major_axis')
            if len(system_df):
                star_info = {
                    "name": selected_system,
                    "temp": system_df['star_temp_k'].iloc[0],
                    "luminosity": system_df['star_luminosity'].iloc[0]
                }
                for _, p in system_df.iterrows():
                    planets_to_sim.append({
                        "name": p['planet_name'],
                        "radius": p['planet_radius'],
                        "orbit_radius": p['orbit_semi_major_axis'],
                        "is_habitable": bool(p['habitability_label'])
                    })
        
        html_code = get_three_js_html(planets_to_sim, star_info)
        components.html(html_code, height=700, scrolling=False)


    # --- 7. SIMULATION ---
    elif page == "⚡ Simulation":
        st.title("⚡ Orbital Simulation")
        st.divider()
        st.info("This tab simulates orbital dynamics using simplified Kepler's laws.")
        
        num_planets = st.slider("Number of planets:", 1, 8, 3)
        planet_data = []
        for i in range(num_planets):
            st.markdown(f"**Planet {i+1}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                r = st.number_input(f"Orbit (AU)", value=float(i+1), key=f"p{i+1}r")
            with c2:
                rad = st.number_input(f"Radius (R_e)", value=1.0, key=f"p{i+1}rad")
            with c3:
                habitable = st.checkbox("Habitable?", value=1.2 < r < 3.0, key=f"p{i+1}h")
            planet_data.append({
                "name": f"Planet {i+1}", "radius": rad, "orbit_radius": r, "is_habitable": habitable
            })
        
        html_sim = get_three_js_html(planet_data, {"name": "Simulated Star", "temp": 5778, "luminosity": 1.0})
        components.html(html_sim, height=650, scrolling=False)


    # --- 8. RESEARCH ANALYTICS ---
    elif page == "📈 Research Analytics":
        st.title("📈 Research Analytics")
        st.divider()
        
        st.subheader("Dataset Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_a = px.scatter(df, x='orbit_semi_major_axis', y='planet_radius',
                             color='habitability_label', title='Orbit vs Size',
                             template='plotly_dark')
            st.plotly_chart(fig_a, use_container_width=True)
        with col_b:
            fig_b = px.histogram(df, x='star_temp_k', title='Star Temperature',
                                 template='plotly_dark', color_discrete_sequence=['#00d4ff'])
            st.plotly_chart(fig_b, use_container_width=True)
        
        st.divider()
        st.subheader("Quick Reports")
        if st.button("📄 Generate Dataset Summary Report"):
            report = generate_markdown_report(df, "Exoplanet Dataset Summary")
            fpath = save_report(report, "dataset_summary.md")
            st.success(f"✅ Report saved to {fpath}")
            st.download_button("📥 Download Report", report, "dataset_summary.md", "text/markdown")


    # --- 9. MODEL COMPARISON ---
    elif page == "🔬 Model Comparison":
        st.title("🔬 Model Comparison")
        st.divider()
        
        if X_test is not None and y_test is not None and rf:
            from src.ai_pipeline import get_model_comparison_metrics
            metrics_df, rf_prob, dnn_prob = get_model_comparison_metrics(rf, dnn, X_test, y_test)
            st.dataframe(metrics_df, use_container_width=True)
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                metrics_melt = metrics_df.reset_index().melt(id_vars='index', var_name='Metric')
                fig = px.bar(metrics_melt, x='Metric', y='value', color='index', barmode='group',
                            title="Performance Comparison", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model comparison requires full models.")


    # --- 10. SETTINGS ---
    elif page == "⚙️ Settings":
        st.title("⚙️ Settings")
        st.divider()
        st.write("Configuration options coming soon!")
        st.checkbox("Enable dark theme (default: on)", value=True, disabled=True)
        st.slider("Animation speed", 0.1, 2.0, 1.0)


    # --- 11. EXPORT RESULTS ---
    elif page == "📤 Export Results":
        st.title("📤 Export Results")
        st.divider()
        
        # Download full dataset
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Dataset", full_csv, "exoplanet_data.csv", "text/csv")
        
        st.divider()
        
        # Generate and download reports
        st.subheader("Generate Reports")
        if st.button("📄 Generate Project Summary Report"):
            report = generate_markdown_report(df, "AI Cosmic Intelligence - Project Summary")
            fpath = save_report(report, "project_summary.md")
            st.success(f"✅ Report saved to {fpath}")
            st.download_button("📥 Download Report", report, "project_summary.md", "text/markdown")


if __name__ == "__main__":
    main()

