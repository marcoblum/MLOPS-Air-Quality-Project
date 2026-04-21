import requests
import os
from dotenv import load_dotenv

load_dotenv()
headers = {"X-API-Key": os.getenv("OPENAQ_API_KEY")}

def find_best_station():
    # Suche nach Locations in der Schweiz (ID 92), die PM2.5 (Parameter ID 2) messen
    url = "https://api.openaq.org/v3/locations?countries_id=92&parameters_id=2"
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get('results', [])
        print(f"Gefunden: {len(results)} Stationen mit PM2.5 in der Schweiz.\n")
        
        for loc in results:
            # Wir listen Name, ID und alle verfügbaren Sensoren auf
            sensors = [s['parameter']['name'] for s in loc.get('sensors', [])]
            print(f"ID: {loc['id']} | Stadt: {loc['name']} | Sensoren: {sensors}")
    else:
        print(f"Fehler: {response.status_code} - {response.text}")

if __name__ == "__main__":
    find_best_station()