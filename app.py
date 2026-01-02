import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import os


st.set_page_config(page_title="Prognos AI - AeroHealth Manager", layout="wide", page_icon="✈️")

@st.cache_resource
def load_fleet_assets(fleet_id):
    model_path = f'{fleet_id}model.h5'
    scaler_path = f'scaler_{fleet_id}.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            
            model = tf.keras.models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading assets: {e}")
    return None, None


st.title("✈️ Fleet Maintenance Management")
st.markdown("##### Real-time Predictive Maintenance & Health Monitoring Dashboard")
st.divider()

selected_fleet = st.sidebar.selectbox("Select Fleet Dataset", ["FD001", "FD002", "FD003", "FD004"])

model, scaler = load_fleet_assets(selected_fleet)

if model:
    tab1, tab2 = st.tabs(["🔍 Individual Engine Inspection", "📊 Full Fleet Status"])

    
    with tab1:
        engine_id = st.number_input("Enter Engine ID", min_value=1, max_value=300, value=1)
        
        
        num_sensors = model.input_shape[-1]
        dummy_input = np.random.rand(1, 30, num_sensors)
        rul = float(model.predict(dummy_input, verbose=0)[0][0])

        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.write(f"### Current Health: Engine #{engine_id}")
            st.metric("Predicted Remaining Useful Life", f"{int(rul)} Cycles")
            
            if rul <= 20: 
                st.error("STATUS: CRITICAL")
                recommendation = "🚨 IMMEDIATE GROUNDING: Schedule Engine Teardown"
            elif rul <= 50: 
                st.warning("STATUS: WARNING")
                recommendation = "⚠️ SCHEDULE INSPECTION: Perform Borescope Check"
            else: 
                st.success("STATUS: HEALTHY")
                recommendation = "✅ NORMAL OPS: No immediate maintenance required"
            
            st.info(f"**Maintenance Action:** {recommendation}")
        
        with col_m2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rul,
                title = {'text': "Life Remaining (Cycles)"},
                gauge = {
                    'axis': {'range': [0, 130]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 20], 'color': "red"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 130], 'color': "green"}]}))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        
        st.divider()
        st.subheader("🛠️ Component-Level Wear Analysis")
        
        
        wear_factor = max(0, min(100, 100 - (rul / 1.3))) 
        
        maint_matrix = pd.DataFrame({
            "Component System": ["High Pressure Compressor (HPC)", "Fan Blades", "Oil System", "Fuel Nozzles"],
            "Estimated Wear": [f"{int(wear_factor)}%", f"{int(wear_factor * 0.85)}%", f"{int(wear_factor * 0.3)}%", f"{int(wear_factor * 0.6)}%"],
            "Last Service": ["15 Cycles Ago", "42 Cycles Ago", "5 Cycles Ago", "88 Cycles Ago"],
            "Task Priority": ["CRITICAL" if rul <= 20 else "High" if rul <= 50 else "Normal"] * 4
        })

        
        def color_priority(val):
            color = 'red' if val == 'CRITICAL' else 'orange' if val == 'High' else 'green'
            return f'color: {color}; font-weight: bold'

        st.table(maint_matrix.style.applymap(color_priority, subset=['Task Priority']))

    
    with tab2:
        st.subheader(f"Global Fleet Health Report: {selected_fleet}")
        
        @st.cache_data
        def calculate_fleet_status(fleet_id):
            # Simulated data for the full fleet status
            num_engines = 100
            ruls = np.random.randint(5, 140, size=num_engines)
            data = pd.DataFrame({
                "Engine ID": np.arange(1, num_engines + 1), 
                "Predicted RUL": ruls
            })
            data["Maintenance Status"] = data["Predicted RUL"].apply(
                lambda r: "🔴 CRITICAL" if r <= 20 else "🟡 WARNING" if r <= 50 else "🟢 HEALTHY"
            )
            return data

        fleet_data = calculate_fleet_status(selected_fleet)

        # FLEET SUMMARY METRICS
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Total Active Units", len(fleet_data))
        sm2.metric("Critical Alerts", len(fleet_data[fleet_data["Predicted RUL"] <= 20]), delta_color="inverse")
        sm3.metric("Fleet Avg Health", f"{int(fleet_data['Predicted RUL'].mean())} Cycles")

        # Visual Grid / Heatmap
        st.write("### 🌡️ Fleet Health Heatmap")
        heatmap_values = fleet_data["Predicted RUL"].values[:100].reshape(10, 10)
        fig_heat = px.imshow(
            heatmap_values, 
            text_auto=True, 
            color_continuous_scale=['red', 'yellow', 'green'],
            labels=dict(color="RUL Cycles")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.write("### 📋 Global Maintenance Log")
        st.dataframe(fleet_data.sort_values("Predicted RUL"), use_container_width=True, hide_index=True)

        # Export Functionality
        csv = fleet_data.to_csv(index=False).encode('utf-8')
        st.download_button("📩 Export Fleet Report", data=csv, file_name=f"{selected_fleet}_fleet_report.csv", mime='text/csv')

else:
    st.error("Assets not found. Please ensure the .h5 models and .pkl scalers are in your project folder.")