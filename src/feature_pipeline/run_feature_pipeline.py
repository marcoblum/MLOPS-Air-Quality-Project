import requests
import pandas as pd
import os
import sys
import hopsworks
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"

SENSOR_IDS = {"pm25": 11463116, "temperature": 11463147, "relativehumidity": 11463139}

def get_timestamp(entry):
    dt_to = entry.get('period', {}).get('datetimeTo', {})
    return dt_to.get('utc') if isinstance(dt_to, dict) else None

def run_pipeline():
    headers = {"X-API-Key": API_KEY}
    all_data = []
    now = datetime.utcnow()

    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        steps = 1
        step_days = 14
        print(f"!!! TEST-BACKFILL: Lade nur {step_days} Tage !!!")
    else:
        steps = 1
        step_days = 1

    for i in range(steps):
        date_to   = (now - timedelta(days=i * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_from = (now - timedelta(days=(i + 1) * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[Schritt {i+1}/{steps}] Lade Zeitraum: {date_from} bis {date_to}")

        for label, s_id in SENSOR_IDS.items():
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1000}

            try:
                # FIX 1: Timeout hinzugefügt – hängt nie länger als 15s
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for entry in results:
                        ts = get_timestamp(entry)
                        if ts:
                            all_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
                    print(f"  -> {label}: {len(results)} Datensätze")
                else:
                    print(f"  -> Fehler {label}: HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"  -> Timeout bei {label} – übersprungen")
            except Exception as e:
                print(f"  -> API Fehler bei {label}: {e}")

        if steps > 1:
            time.sleep(0.5)

    if not all_data:
        print("\nKeine Daten gefunden.")
        return

    raw_df = pd.DataFrame(all_data)
    df = raw_df.pivot_table(index="timestamp", columns="parameter", values="value").reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.columns = [c.lower() for c in df.columns]

    # FIX 2: NaN-Zeilen entfernen – verhindert Fehler beim Insert
    df = df.dropna()
    print(f"\nNach dropna(): {len(df)} Zeilen übrig")

    print(f"\nVerbinde mit Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="AeroPredict")
    fs = project.get_feature_store()

    air_quality_fg = fs.get_or_create_feature_group(
        name="air_quality_features",
        version=1,
        primary_key=["timestamp"],
        description="Air Quality data from OpenAQ Kaserne",
        online_enabled=False,   # FIX 3: Schneller ohne Online-Store
        event_time="timestamp"
    )

    print(f"Sende {len(df)} Zeilen an Hopsworks...")

    # FIX 4: wait=True entfernt – Job im Hopsworks-UI verfolgen
    job, _ = air_quality_fg.insert(df, write_options={"wait_for_job": False})
    print(f"\nJob gestartet! ID: {job.id if job else 'unbekannt'}")
    print("Status prüfen unter: https://c.app.hopsworks.ai/p/<deine-project-id>/jobs")
    print("--- Pipeline abgeschlossen ---")

if __name__ == "__main__":
    run_pipeline()