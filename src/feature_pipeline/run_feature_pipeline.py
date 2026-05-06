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
    if not os.path.exists("C:\\tmp"): os.makedirs("C:\\tmp")
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
        steps = 36
        step_days = 20
        print(f"!!! STARTING STEPPED BACKFILL (Total ca. 720 Tage) !!!")
    else:
        steps = 1
        step_days = 2
    
    for i in range(steps):
        date_to = (now - timedelta(days=i * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_from = (now - timedelta(days=(i + 1) * step_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"\n[Schritt {i+1}/{steps}] Lade Zeitraum: {date_from} bis {date_to}")

        for label, s_id in SENSOR_IDS.items():
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1000}
            
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for entry in results:
                        ts = get_timestamp(entry)
                        if ts: all_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
                    print(f"  -> {label}: {len(results)} Datensätze")
                else:
                    print(f"  -> Fehler {label}: {response.status_code}")
            except Exception as e:
                print(f"  -> API Fehler bei {label}: {e}")
        
        if steps > 1: time.sleep(0.5)

    if all_data:
        raw_df = pd.DataFrame(all_data)
        df = raw_df.pivot_table(index="timestamp", columns="parameter", values="value").reset_index()
        
        # FIX 1: Timestamp als echtes Datetime-Objekt konvertieren (statt BigInt)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.columns = [c.lower() for c in df.columns]

        print(f"\nVerbinde mit Hopsworks... (Gesamtzeilen: {len(df)})")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="AeroPredict")
        fs = project.get_feature_store()

        # FIX 2: Feature Group mit explizitem event_time definieren
        air_quality_fg = fs.get_or_create_feature_group(
            name="air_quality_features",
            version=1,
            primary_key=["timestamp"],
            description="Stepped Air Quality data from OpenAQ Kaserne",
            online_enabled=True,
            event_time="timestamp" # Verhindert Schema-Konflikte bei Zeitreihen
        )

        print(f"Sende {len(df)} Zeilen an die Feature Group...")
        air_quality_fg.insert(df)
        print("--- Upload zu Hopsworks erfolgreich! ---")
    else:
        print("\nKeine Daten gefunden. Bitte API-Key und Sensor-Verfügbarkeit prüfen.")

if __name__ == "__main__":
    run_pipeline()