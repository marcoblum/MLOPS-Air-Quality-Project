import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

# Deine verifizierten Sensor-IDs aus der Kaserne
SENSOR_IDS = {
    "pm25": 11463116,
    "temperature": 11463147,
    "relativehumidity": 11463139
}

def fetch_history():
    headers = {"X-API-Key": API_KEY}
    
    # Start- und Endpunkt für die 2 Jahre
    current_start = datetime(2022, 5, 1)
    final_end = datetime.now()
    step = timedelta(days=10) # 10 Tage entsprechen ca. 240 Messpunkten pro Sensor

    while current_start < final_end:
        current_end = current_start + step
        
        date_from = current_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to = current_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"Lade Zeitraum: {date_from} bis {date_to}...")
        
        all_data = []
        for label, s_id in SENSOR_IDS.items():
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1000}
            
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for entry in results:
                        ts = entry.get('period', {}).get('datetimeTo', {}).get('utc')
                        if ts:
                            all_data.append({"timestamp": ts, "parameter": label, "value": entry['value']})
                
                # Kleine Pause, um die API nicht zu stressen (Rate Limiting)
                time.sleep(0.5)
            except Exception as e:
                print(f"Fehler bei {label}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            os.makedirs("data/raw", exist_ok=True)
            filename = f"data/raw/history_{current_start.strftime('%Y%m%d')}.parquet"
            df.to_parquet(filename, index=False)
            print(f"-> Gespeichert: {filename} ({len(df)} Zeilen)")

        # Zeitfenster weiterschieben
        current_start = current_end

if __name__ == "__main__":
    fetch_history()