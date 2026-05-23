import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Zürich Air Quality Monitor", layout="wide")

st.title("🌬️ Luftqualität Zürich - Live Monitor & Vorhersage")
st.write("Dieses Dashboard zeigt Daten, die stündlich von einer MLOps-Pipeline gesammelt und über Hopsworks bereitgestellt werden.")

# 1. Daten laden (angepasst an das Hopsworks-Format)
@st.cache_data
def load_data():
    if os.path.exists("data/processed/features_latest.parquet"):
        df = pd.read_parquet("data/processed/features_latest.parquet")
        return df
    return None

df = load_data()

# FALLBACK: Wenn lokal nichts liegt, direkt von Hopsworks ziehen
if df is None:
    st.warning("Lokale Cachedatendatei nicht gefunden. Versuche direkt von Hopsworks zu laden...")
    try:
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AeroPredict")
        fs = project.get_feature_store()
        air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
        df = air_quality_fg.read()
    except Exception as e:
        st.error(f"Fehler beim Laden der Live-Daten: {e}")

if df is not None and not df.empty:
    # --- ZEIT-KONVERTIERUNG & FEATURE-SICHERUNG ---
    # Wir stellen sicher, dass timestamp ein echtes Datum ist
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')
        # Zeit-Features für das Modell explizit neu berechnen, falls sie fehlen
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        # Falls timestamp der Index ist
        df.index = pd.to_datetime(df.index)
        df = df.sort_values(by=df.index.name if df.index.name else df.index)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek

    # Letzte Zeilen für Metriken extrahieren
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Zeitstempel bestimmen
    time_info = latest['timestamp'] if 'timestamp' in latest else "Unbekannt"
    st.write(f"**Letztes Update der Messstation:** {time_info}")
    
    # Metriken anzeigen
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aktueller PM2.5", f"{latest['pm25']:.2f} µg/m³", f"{latest['pm25']-prev['pm25']:.2f}")
    col2.metric("24h Durchschnitt", f"{latest['pm25_rolling_24h_mean']:.2f} µg/m³")
    
    if 'temperature' in latest:
        col3.metric("Temperatur", f"{latest['temperature']:.1f} °C")
    if 'relativehumidity' in latest:
        col4.metric("Luftfeuchtigkeit", f"{latest['relativehumidity']:.1f} %")
    
    # 2. Grafik: Verlauf
    st.subheader("Historischer Verlauf (PM2.5 in Zürich)")
    x_axis = 'timestamp' if 'timestamp' in df.columns else df.index
    fig = px.line(df, x=x_axis, y='pm25', title='Gemessene Feinstaubwerte über Zeit')
    st.plotly_chart(fig, use_container_width=True)

    # 3. Vorhersage-Bereich
    st.divider()
    st.subheader("🔮 KI-Vorhersage (Nächste 24 Stunden im Schnitt)")
    
    if os.path.exists("models/air_quality_model.pkl"):
        model = joblib.load("models/air_quality_model.pkl")
        
        # EXAKT die Features, auf die das Modell trainiert wurde
        feature_cols = [
            'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
            'hour', 'day_of_week', 'temperature', 'relativehumidity'
        ]
        
        # Sicherstellen, dass alle Spalten im aktuellen Datensatz existieren
        if all(col in df.columns for col in feature_cols):
            # Wir übergeben die Daten exakt in der richtigen Reihenfolge an das Modell
            X_input = df[feature_cols].tail(1)
            prediction = model.predict(X_input)[0]
            st.info(f"Das trainierte Modell prognostiziert für den kommenden Zeitraum einen PM2.5-Schnitt von **{prediction:.2f} µg/m³**.")
        else:
            st.error("Die benötigten Features für die Vorhersage fehlen im Datensatz.")
            st.write("Vorhandene Spalten:", list(df.columns))
    else:
        st.warning("Kein trainiertes Modell unter `models/air_quality_model.pkl` gefunden.")

else:
    st.error("Noch keine Daten vorhanden.")