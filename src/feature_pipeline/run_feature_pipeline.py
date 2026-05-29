import requests
import pandas as pd
import numpy as np
import os
import sys
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") # Optionale Absicherung für Hugging Face API

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"

SENSOR_IDS = {"pm25": 11463116, "temperature": 11463147, "relativehumidity": 11463139}

def get_timestamp(entry):
    dt_to = entry.get('period', {}).get('datetimeTo', {})
    return dt_to.get('utc') if isinstance(dt_to, dict) else None


def transform_batch(raw_data):
    """Transformiert rohe API-Daten in ein Feature-DataFrame."""
    if not raw_data:
        return None

    raw_df = pd.DataFrame(raw_data)
    df = raw_df.pivot_table(index="timestamp", columns="parameter", values="value")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    # Imputation gegen kurze API-Wackler
    if 'pm25' in df.columns:
        df['pm25'] = df['pm25'].ffill(limit=3).bfill(limit=3)

    # --- FEATURE ENGINEERING ---
    if 'pm25' in df.columns:
        df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        df['pm25_lag_1h'] = df['pm25'].shift(1)
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
                "wait_for_job": False,
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

    # --- MODUS BESTIMMEN ---
    is_backfill = len(sys.argv) > 1 and sys.argv[1] == "--backfill"

    if is_backfill:
        steps = 36
        step_days = 20
        print(f"!!! STARTING STEPPED BACKFILL (Total ca. 720 Tage) !!!")
    else:
        steps = 1
        step_days = 14  # Für stündlichen Lauf: Letzte 14 Tage abrufen
        print(f"Hole Daten der letzten {step_days} Tage für stündliche Feature-Berechnung...")

    # --- HOPSWORKS VERBINDUNG ---
    print(f"Verbinde mit Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="AeroPredict")
        fs = project.get_feature_store()
        air_quality_fg = fs.get_or_create_feature_group(
            name="air_quality_features",
            version=1,
            primary_key=["timestamp"],
            description="Air Quality data with computed features and 24h targets",
            online_enabled=False,
            event_time="timestamp"
        )
        print("Hopsworks Verbindung OK ✅\n")
        hopsworks_available = True
    except Exception as e:
        print(f"⚠️  Hopsworks nicht erreichbar: {e}")
        print("Fahre fort – Daten werden lokal und für Hugging Face aufbereitet.\n")
        hopsworks_available = False
        air_quality_fg = None

    os.makedirs("data/processed", exist_ok=True)
    all_batches = []
    successful_uploads = 0
    failed_uploads = 0

    # --- HAUPT-LOOP ---
    for i in range(steps):
        # KRISENSICHERE DATUMS-BERECHNUNG FÜR DEN LIVE-RUNNER
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
            # Limit hochgeschraubt auf 2000, um sicher alle Stunden der 14 Tage zu fangen!
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 2000}

            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for entry in results:
                        ts = get_timestamp(entry)
                        if ts:
                            batch_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
                    print(f"  -> {label}: {len(results)} Datensätze")
                else:
                    print(f"  -> Fehler {label}: HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"  -> Timeout bei {label} – übersprungen")
            except Exception as e:
                print(f"  -> API Fehler bei {label}: {e}")

        df_batch = transform_batch(batch_data)

        if df_batch is None or len(df_batch) == 0:
            print(f"  -> Keine verwertbaren Daten in diesem Batch, übersprungen.")
            continue

        print(f"  -> {len(df_batch)} Zeilen nach Feature Engineering")

        if is_backfill and hopsworks_available:
            batch_label = f"[{date_from[:10]} – {date_to[:10]}]"
            success = upload_to_hopsworks(df_batch, air_quality_fg, batch_label)
            if success:
                successful_uploads += 1
            else:
                failed_uploads += 1
                fallback_path = f"data/processed/backfill_batch_{i+1}.parquet"
                df_batch.to_parquet(fallback_path)
            time.sleep(0.5)
        else:
            all_batches.append(df_batch)

    # --- STÜNDLICHER LAUF: Absicherung für das Live-Dashboard ---
    if not is_backfill and all_batches:
        df_final = pd.concat(all_batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # 1. Lokal im GitHub-Runner speichern
        local_parquet_path = "data/processed/features_latest.parquet"
        df_final.to_parquet(local_parquet_path)
        print(f"\n✅ Daten lokal im Runner gesichert: {local_parquet_path} ({len(df_final)} Zeilen)")

        # 2. Upload zu Hopsworks versuchen
        if hopsworks_available:
            upload_to_hopsworks(df_final, air_quality_fg, "[stündlicher Lauf]")

        # 3. DIREKT-SYNCHRONISATION MIT HUGGING FACE (Dashboard absolut krisensicher machen!)
        try:
            print("Pushe aktuellen Parquet-Cache direkt zu Hugging Face Spaces...")
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=local_parquet_path,
                path_in_repo="data/processed/features_latest.parquet",
                repo_id="Balumi13/Air-Quality",
                repo_type="space"
            )
            print("✅ Direkt-Upload zu Hugging Face erfolgreich! Dashboard hat frische Daten.")
        except Exception as hf_err:
            print(f"⚠️ Hugging Face Direkt-Upload übersprungen oder fehlgeschlagen: {hf_err}")

    print("\n--- Pipeline abgeschlossen ---")
    if is_backfill:
        print(f"Backfill Ergebnis: {successful_uploads} Batches erfolgreich, {failed_uploads} fehlgeschlagen")

if __name__ == "__main__":
    run_pipeline()