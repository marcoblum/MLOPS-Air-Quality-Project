import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Bern Air Quality Monitor", layout="wide")

st.title("🌬️ Luftqualität Bern - Live Monitor & Vorhersage")
st.write("Dieses Dashboard zeigt Daten, die stündlich von einer MLOps-Pipeline gesammelt werden.")

# 1. Daten laden
@st.cache_data
def load_data():
    if os.path.exists("data/processed/features_latest.parquet"):
        df = pd.read_parquet("data/processed/features_latest.parquet")
        return df
    return None

df = load_data()

if df is not None:
    # Metriken anzeigen (Letzter Wert)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Aktueller PM2.5", f"{latest['pm25']:.2f} µg/m³", f"{latest['pm25']-prev['pm25']:.2f}")
    col2.metric("24h Durchschnitt", f"{latest['pm25_rolling_24h']:.2f} µg/m³")
    
    # 2. Grafik: Verlauf
    st.subheader("Historischer Verlauf (PM2.5)")
    fig = px.line(df, x=df.index, y='pm25', title='Gemessene Feinstaubwerte')
    st.plotly_chart(fig, use_container_width=True)

    # 3. Vorhersage-Bereich
    st.divider()
    st.subheader("🔮 KI-Vorhersage")
    
    if os.path.exists("models/air_quality_model.pkl"):
        model = joblib.load("models/air_quality_model.pkl")
        feature_cols = ['pm25', 'pm25_rolling_24h', 'pm25_lag_1h']
        prediction = model.predict(df[feature_cols].tail(1))[0]
        
        st.info(f"Die KI sagt für die nächste Stunde einen Wert von **{prediction:.2f} µg/m³** voraus.")
    else:
        st.warning("Kein trainiertes Modell gefunden.")

else:
    st.error("Noch keine Daten vorhanden. Bitte lass die Feature Pipeline laufen.")