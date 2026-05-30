import requests
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Lädt deine echten API-Keys aus der vorhandenen .env-Datei
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

SENSOR_IDS = {"pm25": 11463116, "temperature": 11463147, "relativehumidity": 11463139}

def get_timestamp(entry):
    dt_to = entry.get('period', {}).get('datetimeTo', {})
    return dt_to.get('utc') if isinstance(dt_to, dict) else None

def transform_batch(raw_data):
    if not raw_data: return None
    raw_df = pd.DataFrame(raw_data)
    df = raw_df.pivot_table(index="timestamp", columns="parameter", values="value")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    if 'pm25' in df.columns:
        df['pm25'] = df['pm25'].ffill(limit=3).bfill(limit=3)

    # --- FEATURE ENGINEERING (Exakt wie in deinem Original) ---
    if 'pm25' in df.columns:
        df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        df['pm25_lag_1h'] = df['pm25'].shift(1)
        df['pm25_lag_24h'] = df['pm25'].shift(24)
        df['target_24h_mean'] = df['pm25'].shift(-24).rolling(window=24, min_periods=1).mean()

    if 'temperature' in df.columns: df['temperature'] = df['temperature'].ffill(limit=3)
    if 'relativehumidity' in df.columns: df['relativehumidity'] = df['relativehumidity'].ffill(limit=3)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cols_to_check = ['pm25', 'pm25_rolling_24h_mean', 'target_24h_mean']
    df = df.dropna(subset=[c for c in cols_to_check if c in df.columns])
    return df if len(df) > 0 else None

def run_one_time_pipeline():
    headers = {"X-API-Key": API_KEY}
    now = datetime.utcnow()

    # 37 Schritte à 20 Tage = ~740 Tage (Die vollen 2 Jahre Historie PLUS der lückenlose Mai)
    steps = 37 
    step_days = 20
    
    print(f"!!! STARTE ONE-TIME PIPELINE (Total: {steps * step_days} Tage) !!!")
    print("Sammle alle historischen Daten lokal im RAM ohne Hopsworks-Upload...\n")

    os.makedirs("data/processed", exist_ok=True)
    all_historical_batches = []

    for i in range(steps):
        date_to   = (now - timedelta(days=i * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_from = (now - timedelta(days=(i + 1) * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"[Batch {i+1}/{steps}] Lade Zeitraum: {date_from} bis {date_to}")
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
                    print(f"   -> {label}: {len(results)} Zeilen geladen")
                else:
                    print(f"   -> Fehler {label}: HTTP {response.status_code}")
            except Exception as e:
                print(f"   -> API Fehler bei {label}: {e}")

        df_batch = transform_batch(batch_data)

        if df_batch is not None:
            print(f"   -> ✅ {len(df_batch)} Zeilen verarbeitet und im RAM gesichert.")
            all_historical_batches.append(df_batch)
        else:
            print(f"   -> ⚠️ Keine Daten in diesem Zeitraum.")
        
        time.sleep(0.2) # Schutz vor OpenAQ Rate-Limits

    if all_historical_batches:
        print("\nVerschmleze alle Zeiträume zu einer großen lokalen Master-Datei...")
        df_total = pd.concat(all_historical_batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        # Exakt der Speicherort, den dein train_model.py erwartet
        target_path = "data/processed/features_latest.parquet"
        df_total.to_parquet(target_path)
        
        print(f"\n🎉 FERTIG! Lokale Master-Datei erfolgreich erstellt: {target_path}")
        print(f"Gesamtzeilen (2 Jahre Vergangenheit + lückenloser Mai): {len(df_total)}")
    else:
        print("❌ Kritischer Fehler: Es konnten keine Daten gesammelt werden.")

if __name__ == "__main__":
    run_one_time_pipeline()