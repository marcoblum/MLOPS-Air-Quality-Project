import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Zürich Air Quality Monitor", layout="wide")

st.title("🌬️ Luftqualität Zürich - Live Monitor & Vorhersage")
st.write("Dieses Dashboard zeigt Daten, die stündlich von einer MLOps-Pipeline gesammelt und über Hopsworks bereitgestellt werden.")

@st.cache_data(ttl=3600)  # Maximal 1 Stunde cachen
def load_data():
    # 1. Versuch: Lokaler Cache
    if os.path.exists("data/processed/features_latest.parquet"):
        return pd.read_parquet("data/processed/features_latest.parquet")
        
    # 2. Versuch: Live-Abfrage aus Hopsworks
    try:
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()
        
        # Hol den Key aus den Hugging Face Secrets oder der lokalen .env
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            st.error("Kritischer Fehler: Kein HOPSWORKS_API_KEY in den Umgebungsvariablen gefunden!")
            return None
            
        project = hopsworks.login(api_key_value=api_key, project="AeroPredict")
        fs = project.get_feature_store()
        air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
        df = air_quality_fg.read()
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Live-Daten von Hopsworks: {e}")
        return None

df = load_data()

if df is not None and not df.empty:
    # Spaltennamen zur Sicherheit komplett in Kleinschreibung umwandeln
    df.columns = [c.lower() for c in df.columns]
    
    # Zeitstempel-Spalte finden und konvertieren
    time_col = None
    for col in ['timestamp', 'date', 'zeit']:
        if col in df.columns:
            time_col = col
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col)
            break
            
    # Falls kein Zeitstempel gefunden wurde, nutzen wir den Index
    if time_col is None:
        df.index = pd.to_datetime(df.index)
        df = df.sort_values(by=df.index)
        df['derived_time'] = df.index
        time_col = 'derived_time'

    # Dynamische Features berechnen, falls sie direkt aus Hopsworks fehlen
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    
    # Wichtig: Falls die rollierenden Spalten fehlen, berechnen wir sie fliegend!
    if 'pm25_rolling_24h_mean' not in df.columns and 'pm25' in df.columns:
        df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
    if 'pm25_lag_1h' not in df.columns and 'pm25' in df.columns:
        df['pm25_lag_1h'] = df['pm25'].shift(1).bfill()
    if 'pm25_lag_24h' not in df.columns and 'pm25' in df.columns:
        df['pm25_lag_24h'] = df['pm25'].shift(24).bfill()

    # Letzte Zeilen extrahieren
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Zeit-Info anzeigen
    st.write(f"**Letztes Update der Messstation:** {latest[time_col]}")
    
    # --- METRIKEN ---
    col1, col2, col3, col4 = st.columns(4)
    
    if 'pm25' in latest:
        diff = latest['pm25'] - prev['pm25'] if 'pm25' in prev else 0.0
        col1.metric("Aktueller PM2.5", f"{latest['pm25']:.2f} µg/m³", f"{diff:.2f}")
        
    if 'pm25_rolling_24h_mean' in latest:
        col2.metric("24h Durchschnitt", f"{latest['pm25_rolling_24h_mean']:.2f} µg/m³")
    elif 'pm25_rolling_24h' in latest:
        col2.metric("24h Durchschnitt", f"{latest['pm25_rolling_24h']:.2f} µg/m³")
        
    if 'temperature' in latest:
        col3.metric("Temperatur", f"{latest['temperature']:.1f} °C")
    if 'relativehumidity' in latest:
        col4.metric("Luftfeuchtigkeit", f"{latest['relativehumidity']:.1f} %")
    
    # --- VERLAUFSGRAFIK ---
    st.subheader("Historischer Verlauf (PM2.5 in Zürich)")
    if 'pm25' in df.columns:
        fig = px.line(df, x=time_col, y='pm25', title='Gemessene Feinstaubwerte über Zeit')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Keine PM2.5-Spalte für die Grafik vorhanden.")

    # --- KI-VORHERSAGE ---
    st.divider()
    st.subheader("🔮 KI-Vorhersage (Nächste 24 Stunden im Schnitt)")
    
    model_path = "models/air_quality_model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            feature_cols = [
                'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
                'hour', 'day_of_week', 'temperature', 'relativehumidity'
            ]
            
            # Überprüfen, ob alle Features existieren
            if all(col in df.columns for col in feature_cols):
                X_input = df[feature_cols].tail(1)
                prediction = model.predict(X_input)[0]
                st.info(f"Das trainierte Modell prognostiziert für den kommenden Zeitraum einen PM2.5-Schnitt von **{prediction:.2f} µg/m³**.")
            else:
                st.error("Die benötigten Features für die Vorhersage fehlen im Hopsworks-Datensatz.")
                st.write("Vorhandene Spalten:", list(df.columns))
        except Exception as e:
            st.error(f"Fehler bei der Modellvorhersage: {e}")
    else:
        st.warning("Kein trainiertes Modell unter `models/air_quality_model.pkl` gefunden.")
else:
    st.error("Es konnten keine Daten geladen werden. Bitte überprüfe die Hopsworks-Verbindung.")