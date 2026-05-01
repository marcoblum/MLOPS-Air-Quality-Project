import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

# NEU: IDs für Zürich-Kaserne (Location 9589)
SENSOR_IDS = {
    "pm25": 15168993,
    "temperature": 15168995,
    "relativehumidity": 15168994
}

def get_timestamp(entry):
    period = entry.get('period', {})
    dt_to = period.get('datetimeTo', {})  # kein Snake_case!
    if isinstance(dt_to, dict):
        return dt_to.get('utc')
    return None

def run_pipeline():
    headers = {"X-API-Key": API_KEY}
    all_data = []

    print("=== Sensor-Status Check ===")
    for label, s_id in SENSOR_IDS.items():
        url = f"https://api.openaq.org/v3/sensors/{s_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get('results', [{}])[0]
            last = data.get('datetimeLast', {})
            print(f"{label} (ID {s_id}): letztes Datum = {last.get('utc', 'unbekannt')}")
        else:
            print(f"{label}: Fehler {response.status_code}")
    print("===========================\n")

    # Zeitraum definieren (letzte 48 Stunden)
    date_to = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    date_from = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for label, s_id in SENSOR_IDS.items():
        # WICHTIG: Die URL bleibt sauber, die Parameter kommen in das params-Dictionary
        url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
        
        # HIER sind die entscheidenden Parameter für die v3 API
        params = {
            "datetime_from": date_from,
            "datetime_to": date_to,
            "limit": 100,
            "order_by": "datetime", # Sortiere nach Zeit
            "sort": "desc"          # Neueste Werte zuerst (2026 statt 2017)
        }
        
        print(f"Fetching {label} (ID: {s_id}) for Zurich...")
        
        # params=params sorgt dafür, dass requests die URL korrekt zusammenbaut
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            results = response.json().get('results', [])
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
        
        # NEU: Dateiname auf 'zurich' geändert
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_path = f"data/raw/air_quality_zurich_{timestamp_str}.parquet"
        
        df.to_parquet(file_path, index=False)
        
        print(f"\n--- Feature Pipeline Erfolgreich ---")
        print(f"Gespeichert unter: {file_path}")
        print(f"Zeitraum: {date_from} bis {date_to}")
        print(df.sort_values("timestamp", ascending=False).head()) # Hier sollten nun 2026-Zeitstempel stehen
    else:
        print("Keine aktuellen Daten gefunden. Prüfe die API-Verfügbarkeit.")

if __name__ == "__main__":
    run_pipeline()