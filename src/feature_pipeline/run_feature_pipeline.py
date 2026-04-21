import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

# Sensoren für Bern-Bollwerk (Location 9582)
SENSOR_IDS = {
    "pm25": 29493, 
    "no2": 29491   
}

def get_timestamp(entry):
    """Sicherer Extraktor für v3 Zeitstempel."""
    period = entry.get('period', {})
    # Wir probieren die gängigsten v3 Formate durch
    dt_to = period.get('datetime_to') or period.get('datetimeTo')
    if dt_to and isinstance(dt_to, dict):
        return dt_to.get('utc')
    # Backup: Falls es direkt unter datetimeFrom liegt
    dt_from = entry.get('datetimeFrom')
    if dt_from and isinstance(dt_from, dict):
        return dt_from.get('utc')
    return None

def run_pipeline():
    headers = {"X-API-Key": API_KEY}
    all_data = []

    for label, s_id in SENSOR_IDS.items():
        url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements?limit=100"
        print(f"Fetching {label} (ID: {s_id})...")
        
        response = requests.get(url, headers=headers)
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
            print(f"Fehler bei {label}: {response.status_code}")

    if all_data:
        df = pd.DataFrame(all_data)
        
        # Sicherstellen, dass der lokale Storage existiert [cite: 32]
        os.makedirs("data/raw", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_path = f"data/raw/air_quality_bern_{timestamp_str}.parquet"
        
        # Speichern als versionierte Parquet-Datei [cite: 25, 30]
        df.to_parquet(file_path, index=False)
        
        print(f"\n--- Feature Pipeline Erfolgreich ---")
        print(f"Gespeichert unter: {file_path}")
        print(df.head())
    else:
        print("Keine validen Daten mit Zeitstempel gefunden.")

if __name__ == "__main__":
    run_pipeline()