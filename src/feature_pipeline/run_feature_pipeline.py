import os
import requests
import pandas as pd
import numpy as np
import sys
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()
print(f"DEBUG IN RUNNER - Projekt: '{os.getenv('HOPSWORKS_PROJECT')}'")
print(f"DEBUG IN RUNNER - Key vorhanden?: {os.getenv('HOPSWORKS_API_KEY') is not None}")
API_KEY = os.getenv("OPENAQ_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID") 

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

SENSOR_IDS = {"pm25": 11463116, "temperature": 11463147, "relativehumidity": 11463139}

def get_timestamp(entry):
    dt_to = entry.get('period', {}).get('datetimeTo', {})
    return dt_to.get('utc') if isinstance(dt_to, dict) else None


def fetch_weather_data(start_date, end_date):
    """Holt stündliche Wind- und Wetterdaten von Open-Meteo passend zum Batch-Zeitraum."""
    lat, lon = 47.3769, 8.5417  # Region Zürich
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


def transform_batch(raw_data, is_live_mode=True):
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

    if 'pm25' in df.columns:
        df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        df['pm25_rolling_24h_var'] = df['pm25'].rolling(window=24, min_periods=1).var().fillna(0)
        df['pm25_lag_1h'] = df['pm25'].shift(1).bfill()
        df['pm25_lag_6h'] = df['pm25'].shift(6).bfill()
        df['pm25_lag_24h'] = df['pm25'].shift(24).bfill()
        df['target_24h_mean'] = df['pm25'].shift(-24).rolling(window=24, min_periods=1).mean()

    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].ffill(limit=3)
    if 'relativehumidity' in df.columns:
        df['relativehumidity'] = df['relativehumidity'].ffill(limit=3)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if len(df) > 0:
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        weather_df = fetch_weather_data(min_ts, max_ts)
        
        if weather_df is not None:
            df['timestamp_match'] = df['timestamp'].dt.floor('h').dt.tz_localize(None)
            weather_df['timestamp_match'] = weather_df['timestamp_match'].dt.floor('h')
            df = pd.merge(df, weather_df, on='timestamp_match', how='left')
            df = df.drop(columns=['timestamp_match'])
            
            if 'wind_direction' in df.columns:
                df['wind_direction'] = df['wind_direction'].astype('float64')
            if 'wind_speed' in df.columns:
                df['wind_speed'] = df['wind_speed'].astype('float64')
            
            df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'].fillna(0) + 1.0)
            df['temp_humidity_interaction'] = df['temperature'].fillna(0) * df['relativehumidity'].fillna(0)

    if is_live_mode:
        cols_to_check = ['pm25', 'pm25_rolling_24h_mean']
    else:
        cols_to_check = ['pm25', 'pm25_rolling_24h_mean', 'target_24h_mean']
        
    df = df.dropna(subset=[c for c in cols_to_check if c in df.columns])

    return df if len(df) > 0 else None


def upload_to_hopsworks(df, air_quality_fg, batch_label=""):
    """Lädt einen DataFrame direkt via Python API hoch ohne Spark Queue."""
    try:
        print(f"  -> Sende {len(df)} Zeilen an Hopsworks (ohne Spark)... {batch_label}")
        air_quality_fg.insert(
            df,
            write_options={
                "wait_for_job": True,
                "use_spark": False
            }
        )
        print(f"  -> ✅ Upload zu Hopsworks erfolgreich! {batch_label}")
        return True
    except Exception as e:
        print(f"  -> ❌ Upload zu Hopsworks fehlgeschlagen ({batch_label}): {e}")
        return False


def run_pipeline():
    headers = {"X-API-Key": API_KEY}
    now = datetime.utcnow()

    is_backfill = len(sys.argv) > 1 and sys.argv[1] == "--backfill"

    if is_backfill:
        steps = 36
        step_days = 20
        print(f"!!! STARTING HOPSWORKS STEPPED BACKFILL (Total ca. 720 Tage) !!!")
    else:
        steps = 1
        step_days = 14  
        print(f"Hole Daten der letzten {step_days} Tage für stündliche Feature-Berechnung (inkl. Wind)...")

    print(f"Verbinde mit Hopsworks...")
    try:
        if not HOPSWORKS_PROJECT:
            raise ValueError("HOPSWORKS_PROJECT fehlt in der .env Datei!")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        air_quality_fg = fs.get_or_create_feature_group(
            name="air_quality_features_1",
            version=1,
            primary_key=["timestamp"],
            description="Air Quality data with wind features and computed interactions",
            online_enabled=False,
            event_time="timestamp"
        )
        print("Hopsworks Verbindung OK ✅\n")
    except Exception as e:
        print(f"❌ Kritischer Fehler: Verbindung zu Hopsworks fehlgeschlagen: {e}")
        return

    os.makedirs("data/processed", exist_ok=True)
    all_batches = []

    for i in range(steps):
        if not is_backfill:
            date_to   = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            date_from = (now - timedelta(days=step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            date_to   = (now - timedelta(days=i * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            date_from = (now - timedelta(days=(i + 1) * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"\n[Schritt {i+1}/{steps}] Lade Zeitraum: {date_from} bis {date_to}")

        batch_data = []

        for label, s_id in SENSOR_IDS.items():
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1000}

            success = False
            retries = 3
            backoff = 2.0

            for attempt in range(retries):
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=15)
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        for entry in results:
                            ts = get_timestamp(entry)
                            if ts:
                                batch_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
                        print(f"  -> {label}: {len(results)} Datensätze geladen.")
                        success = True
                        break
                    elif response.status_code == 429:
                        print(f"  -> ⚠️ Rate Limit (429) bei {label}. Warte {backoff}s... (Versuch {attempt+1}/{retries})")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        print(f"  -> ⚠️ HTTP {response.status_code} bei {label}. Versuche erneut in 2s...")
                        time.sleep(2)
                except Exception as e:
                    print(f"  -> ⚠️ Netzwerk-/API-Fehler bei {label}: {e}. Versuche erneut in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2

            if not success:
                print(f"  -> ❌ Schritt final fehlgeschlagen für Sensor {label} nach {retries} Versuchen.")

        df_batch = transform_batch(batch_data, is_live_mode=(not is_backfill))

        if df_batch is None or len(df_batch) == 0:
            print(f"  -> Keine verwertbaren Daten in diesem Batch, übersprungen.")
            continue

        print(f"  -> {len(df_batch)} Zeilen nach Feature Engineering")
        all_batches.append(df_batch)
        time.sleep(0.5)

    if all_batches:
        print("\nAggregiere und bereinige alle gesammelten Batches...")
        df_final = pd.concat(all_batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f" -> Bereinigter Datensatz enthält {len(df_final)} eindeutige Zeilen.")

        if is_backfill:
            print(f"\nStarte finalen Gesamt-Upload des 2-Jahres-Backfills an Hopsworks...")
            upload_to_hopsworks(df_final, air_quality_fg, "[Kompletter Backfill]")
        
        else:
            local_parquet_path = "data/processed/features_latest.parquet"
            df_final.to_parquet(local_parquet_path)
            print(f"\n✅ Live-Daten lokal gesichert: {local_parquet_path} ({len(df_final)} Zeilen)")

            upload_to_hopsworks(df_final, air_quality_fg, "[stündlicher Lauf]")

            try:
                print("Pushe aktuellen Parquet-Cache direkt zu Hugging Face Spaces...")
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=local_parquet_path,
                    path_in_repo="data/processed/features_latest.parquet",
                    repo_id=HF_REPO_ID,
                    repo_type="space"
                )
                print("✅ Direkt-Upload zu Hugging Face erfolgreich!")
            except Exception as hf_err:
                print(f"⚠️ Hugging Face Direkt-Upload fehlgeschlagen: {hf_err}")

    print("\n--- Pipeline abgeschlossen ---")

if __name__ == "__main__":
    run_pipeline()