import requests
import os
from dotenv import load_dotenv

load_dotenv()
headers = {"X-API-Key": os.getenv("OPENAQ_API_KEY")}

def find_best_station():
    # Suche nach Locations in der Schweiz (ID 92), die PM2.5 (Parameter ID 2) messen
    url = "https://api.openaq.org/v3/locations"
    
    # Die Parameter müssen in ein Dictionary, damit requests sie an die URL anhängt
    params = {
        "countries_id": 92,
        "parameters_id": 2,
        "order_by": "id",  # Geändert von last_updated auf id
        "sort": "desc",    # Höchste IDs zuerst (meistens die neuesten Sensoren)
        "limit": 100       # Wir laden mehr, um manuell nach dem Datum zu schauen
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        results = response.json().get('results', [])
        print(f"Top {len(results)} aktuellste Stationen mit PM2.5 in der Schweiz:\n")
        
        for loc in results:
            # Korrektur: 'last_updated' statt 'lastUpdated'
            last_upd = loc.get('last_updated', 'Unbekannt')
            print(f"Stadt: {loc['name']} (ID: {loc['id']}) | Letztes Update: {last_upd}")
            
            for s in loc.get('sensors', []):
                print(f"  - Sensor: {s['parameter']['name']} | ID: {s['id']}")
            print("-" * 30)
    else:
        print(f"Fehler: {response.status_code} - {response.text}")

if __name__ == "__main__":
    find_best_station()