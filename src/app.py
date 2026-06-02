import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Zürich Air Quality Monitor", layout="wide")

st.title("Air Quality Zürich - Live Monitor & Forecast")
st.write("This dashboard displays data collected hourly by an MLOps pipeline and served via Hopsworks.")

@st.cache_data(ttl=3600)
def load_data():
    df_live = None
    possible_live_paths = [
        "data/processed/features_latest.parquet", 
        "./data/processed/features_latest.parquet"
    ]
    for path in possible_live_paths:
        if os.path.exists(path):
            df_live = pd.read_parquet(path)
            break
            
    if df_live is None:
        try:
            import hopsworks
            # Nutze st.secrets für Hugging Face Cloud-Kompatibilität, falls .env fehlt
            api_key = os.getenv("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
            if api_key:
                project = hopsworks.login(api_key_value=api_key, project="AeroPredict")
                fs = project.get_feature_store()
                air_quality_fg = fs.get_feature_group(name="air_quality_features_1", version=1)
                df_live = air_quality_fg.read()
        except Exception as e:
            st.error(f"Error loading live data from Hopsworks: {e}")

    # --- GEÄNDERT: Die 2-Jahres-Historie wird nun direkt aus Hopsworks geladen ---
    df_history = None
    try:
        import hopsworks
        api_key = os.getenv("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
        if api_key:
            # Separater, langlebigerer Cache für die historischen Daten via innerer Funktion
            @st.cache_data(ttl=86400)
            def fetch_history_from_cloud(key):
                proj = hopsworks.login(api_key_value=key, project="AeroPredict")
                store = proj.get_feature_store()
                fg = store.get_feature_group(name="air_quality_features_1", version=1)
                return fg.read()
                
            df_history = fetch_history_from_cloud(api_key)
    except Exception as e:
        st.warning(f"Could not load historical data from Hopsworks Feature Store: {e}")

    return df_live, df_history

df_live_raw, df_history_raw = load_data()

def prepare_dataframe(df_input):
    if df_input is None or df_input.empty:
        return None
    df = df_input.copy()
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

df_live = prepare_dataframe(df_live_raw)
df_history = prepare_dataframe(df_history_raw)

def fill_missing_features(df):
    if df is None: return None
    df.columns = [c.lower() for c in df.columns]
    
    # Robuste Imputation für Live-Daten-Wackler vor der Berechnung
    cols_to_impute = ['pm25', 'temperature', 'relativehumidity', 'wind_speed', 'wind_direction', 'surface_pressure']
    for col in cols_to_impute:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    if 'pm25' in df.columns:
        if 'pm25_rolling_24h_mean' not in df.columns:
            df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        if 'pm25_rolling_24h_var' not in df.columns:
            df['pm25_rolling_24h_var'] = df['pm25'].rolling(window=24, min_periods=1).var().fillna(0)
        if 'pm25_lag_1h' not in df.columns:
            df['pm25_lag_1h'] = df['pm25'].shift(1).bfill()
        if 'pm25_lag_6h' not in df.columns:
            df['pm25_lag_6h'] = df['pm25'].shift(6).bfill()
        if 'pm25_lag_24h' not in df.columns:
            df['pm25_lag_24h'] = df['pm25'].shift(24).bfill()
        
        # Interaktionen absichern
        if 'wind_speed' in df.columns:
            df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
        if 'temperature' in df.columns and 'relativehumidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']
            
    return df

df_live = fill_missing_features(df_live)

if df_live is not None and not df_live.empty:
    time_col = 'timestamp'
    df_live['hour'] = df_live[time_col].dt.hour
    df_live['day_of_week'] = df_live[time_col].dt.dayofweek

    latest = df_live.iloc[-1]
    prev = df_live.iloc[-2] if len(df_live) > 1 else latest
    
    st.write(f"**Last station update:** {latest[time_col]}")
    
    # --- SECTION 1: CURRENT MEASUREMENTS ---
    st.subheader("Current Measurements")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    if 'pm25' in latest:
        diff = latest['pm25'] - prev['pm25'] if 'pm25' in prev else 0.0
        m_col1.metric("Current PM2.5", f"{latest['pm25']:.2f} µg/m³", f"{diff:.2f}")
    if 'temperature' in latest:
        m_col2.metric("Temperature", f"{latest['temperature']:.1f} °C")
    if 'relativehumidity' in latest:
        m_col3.metric("Humidity", f"{latest['relativehumidity']:.1f} %")
    if 'wind_speed' in latest:
        m_col4.metric("Wind Speed", f"{latest['wind_speed']:.1f} km/h")
    if 'surface_pressure' in latest:
        m_col5.metric("Air Pressure", f"{latest['surface_pressure']:.0f} hPa")

    # --- SECTION 2: 14-DAY TREND ---
    st.divider()
    st.subheader("Current Trend (Last 14 Days)")
    if 'pm25' in df_live.columns:
        fig_live = px.line(df_live, x=time_col, y='pm25', title='Particulate Matter – Recent History')
        fig_live.update_traces(line_color='#2ca02c')
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.warning("No live data available for trend chart.")

    # --- SECTION 3: AI FORECAST ---
    st.divider()
    st.subheader("Predictive Health Management: 24h Forecast Comparison")
    
    model_path = "models/air_quality_model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            # FIX: wind_direction hinzugefügt, damit das Set exakt mit den 14 Trainings-Features übereinstimmt
            feature_cols = [
                'pm25_rolling_24h_mean', 'pm25_rolling_24h_var', 
                'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',    
                'hour', 'day_of_week', 'temperature', 'relativehumidity', 
                'surface_pressure', 'wind_speed', 'wind_direction',
                'pm25_wind_interaction', 'temp_humidity_interaction'
            ]
            
            if all(col in df_live.columns for col in feature_cols):
                X_input = df_live[feature_cols].tail(1)
                prediction = model.predict(X_input)[0]
                
                p_col1, p_col2 = st.columns(2)
                
                if 'pm25_rolling_24h_mean' in latest:
                    p_col1.metric(
                        label="Current State: PM2.5 Average (Last 24h)", 
                        value=f"{latest['pm25_rolling_24h_mean']:.2f} µg/m³",
                        delta="Historical Baseline"
                    )
                
                delta_zukunft = prediction - latest['pm25_rolling_24h_mean'] if 'pm25_rolling_24h_mean' in latest else 0.0
                p_col2.metric(
                    label="Model Forecast: Expected PM2.5 Average (Next 24h)", 
                    value=f"{prediction:.2f} µg/m³",
                    delta=f"{delta_zukunft:+.2f} µg/m³ vs. today",
                    delta_color="inverse"
                )
                
                st.markdown("##### **Preventive Recommendation for Asthma Patients:**")
                if prediction < 10.0:
                    st.success("🟢 **Safe:** Forecasted particulate levels are low. No restrictions on outdoor activities.")
                elif prediction < 25.0:
                    st.warning("🟡 **Moderate:** Levels are slightly elevated. Sensitive individuals should monitor prolonged physical exertion outdoors.")
                else:
                    st.error("🔴 **High Risk:** Elevated levels forecasted. Asthma patients and vulnerable groups are advised to reduce intense outdoor activities over the next 24 hours.")
                    
            else:
                st.error("Required features for the forecast are missing from the live dataset.")
                missing = [c for c in feature_cols if c not in df_live.columns]
                st.write("Missing columns:", missing)
        except Exception as e:
            st.error(f"Error during model forecast: {e}")
    else:
        st.warning("No trained model found at `models/air_quality_model.pkl`.")

    # --- SECTION 4: HISTORY ---
    st.divider()
    with st.expander("Show 2-Year History"):
        st.write("The complete historical dataset used for model training.")
        if df_history is not None and 'pm25' in df_history.columns:
            fig_hist = px.line(df_history, x=time_col, y='pm25', title='Measured Particulate Matter since Project Start (2024 – Today)')
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Historical data could not be loaded from Hopsworks Cloud.")
else:
    st.error("No live data could be loaded. Please check the data structure.")