import requests
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

SENSOR_IDS = {"pm25": 11463116, "temperature": 11463147, "relativehumidity": 11463139}

def get_timestamp(entry):
    dt_to = entry.get('period', {}).get('datetimeTo', {})
    return dt_to.get('utc') if isinstance(dt_to, dict) else None


def fetch_weather_data(start_date, end_date):
    """Holt historische stündliche Wind- und Wetterdaten von Open-Meteo."""
    lat, lon = 47.3769, 8.5417  # Region Zentralschweiz / Zürich
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "wind_speed_10m,wind_direction_10m,surface_pressure",
        "timezone": "UTC"
    }
    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code == 200:
            hourly = response.json().get("hourly", {})
            weather_df = pd.DataFrame({
                "timestamp_match": pd.to_datetime(hourly.get("time")),
                "wind_speed": hourly.get("wind_speed_10m"),
                "wind_direction": hourly.get("wind_direction_10m"),
                "surface_pressure": hourly.get("surface_pressure")
            })
            return weather_df
    except Exception:
        return None
    return None


def transform_batch(raw_data):
    """Transformiert rohe API-Daten in ein erweitertes Feature-DataFrame."""
    if not raw_data:
        return None

    raw_df = pd.DataFrame(raw_data)
    df = raw_df.pivot_table(index="timestamp", columns="parameter", values="value")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    if 'pm25' in df.columns:
        df['pm25'] = df['pm25'].ffill(limit=3).bfill(limit=3)

    # --- ADVANCED FEATURE ENGINEERING (Volle Proposal- & Wind-Compliance) ---
    if 'pm25' in df.columns:
        df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        df['pm25_rolling_24h_var'] = df['pm25'].rolling(window=24, min_periods=1).var()  
        df['pm25_lag_1h'] = df['pm25'].shift(1)
        df['pm25_lag_6h'] = df['pm25'].shift(6)                                          
        df['pm25_lag_24h'] = df['pm25'].shift(24)
        df['target_24h_mean'] = df['pm25'].shift(-24).rolling(window=24, min_periods=1).mean()

    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].ffill(limit=3)
    if 'relativehumidity' in df.columns:
        df['relativehumidity'] = df['relativehumidity'].ffill(limit=3)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- LIVE-METEOROLOGIE DAZUMISCHEN ---
    if len(df) > 0:
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        weather_df = fetch_weather_data(min_ts, max_ts)
        
        if weather_df is not None:
            df['timestamp_match'] = df['timestamp'].dt.floor('h').dt.tz_localize(None)
            weather_df['timestamp_match'] = weather_df['timestamp_match'].dt.floor('h')
            df = pd.merge(df, weather_df, on='timestamp_match', how='left')
            df = df.drop(columns=['timestamp_match'])
            
            df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
            df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']

    cols_to_check = ['pm25', 'pm25_rolling_24h_mean', 'target_24h_mean']
    df = df.dropna(subset=[c for c in cols_to_check if c in df.columns])

    return df if len(df) > 0 else None


def run_temp_backfill():
    headers = {"X-API-Key": API_KEY}
    now = datetime.utcnow()

    # Wir jagen die vollen 36 Schritte à 20 Tage hoch (~720 Tage Historie)
    steps = 36
    step_days = 20
    
    print(f"!!! STARTING LOCAL-ONLY TEMPORARY BACKFILL (Total ca. 720 Tage) !!!")
    print("Sammle alle historischen Daten inklusive Wind im RAM...\n")

    all_batches = []

    # --- HAUPT-LOOP ---
    for i in range(steps):
        date_to   = (now - timedelta(days=i * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_from = (now - timedelta(days=(i + 1) * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[Schritt {i+1}/{steps}] Lade Zeitraum: {date_from} bis {date_to}")

        batch_data = []
        for label, s_id in SENSOR_IDS.items():
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1000}
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for entry in results:
                        ts = get_timestamp(entry)
                        if ts:
                            batch_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
            except Exception:
                pass

        df_batch = transform_batch(batch_data)

        if df_batch is not None and len(df_batch) > 0:
            print(f"  -> ✅ {len(df_batch)} Zeilen verarbeitet.")
            all_batches.append(df_batch)
            
        time.sleep(0.25) # Schutz vor OpenAQ Rate-Limits

    # --- DATEN-SICHERUNG ---
    if all_batches:
        df_final = pd.concat(all_batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Erstellt den Ordner vollautomatisch, falls er noch nicht existiert!
        backup_dir = "data/local_backup_2_years"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_path = f"{backup_dir}/features_latest.parquet"
        df_final.to_parquet(backup_path)
        print(f"\n🎉 FERTIG! Riesiges Historien-File erfolgreich lokal erstellt: {backup_path}")
        print(f"Gesamtzeilen für dein Rekord-Training: {len(df_final)}")
    else:
        print("❌ Fehler: Es konnten keine Daten gesammelt werden.")

if __name__ == "__main__":
    run_temp_backfill()