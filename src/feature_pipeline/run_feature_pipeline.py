import requests
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

# Diese IDs kommen direkt aus DEINEM funktionierenden find_location-Screenshot
SENSOR_IDS = {
    "pm25": 11463116,
    "temperature": 11463147,
    "relativehumidity": 11463139
}

def get_timestamp(entry):
    period = entry.get('period', {})
    dt_to = period.get('datetimeTo', {})
    if isinstance(dt_to, dict):
        return dt_to.get('utc')
    return None

def run_pipeline():
    headers = {"X-API-Key": API_KEY}
    all_data = []

    # Zeitraum Logik
    date_to = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    date_from = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        # Beim Backfill gehen wir weit zurück
        date_from = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"!!! BACKFILLING: Lade Sensoren der Kaserne seit {date_from} !!!")

    for label, s_id in SENSOR_IDS.items():
        # WICHTIG: In v3 ist der Pfad für SENSOR-Messwerte: /sensors/{id}/measurements
        url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
        
        params = {
            "datetime_from": date_from,
            "datetime_to": date_to,
            "limit": 1000,
            "order_by": "datetime",
            "sort": "desc"
        }
        
        print(f"Fetching {label} (ID: {s_id})...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
            print(f"-> {len(results)} Datensätze für {label} gefunden.")
            for entry in results:
                ts = get_timestamp(entry)
                if ts:
                    all_data.append({
                        "timestamp": ts,
                        "parameter": label,
                        "value": entry['value']
                    })
        else:
            print(f"Fehler bei {label}: {response.status_code} - {response.text}")

    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs("data/raw", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_path = f"data/raw/air_quality_kaserne_{timestamp_str}.parquet"
        df.to_parquet(file_path, index=False)
        print(f"\n--- Pipeline Erfolgreich: {len(df)} Zeilen gespeichert ---")
    else:
        print("Keine Daten gefunden. Prüfe API Key oder Sensor IDs.")

if __name__ == "__main__":
    run_pipeline()