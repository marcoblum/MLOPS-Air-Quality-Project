import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

def test_connection():
    # Wir rufen die Liste der Länder ab - das ist der stabilste v3 Endpunkt
    url = "https://api.openaq.org/v3/countries"
    headers = {"X-API-Key": API_KEY}
    
    print("Testing OpenAQ v3 connection...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        countries = data.get('results', [])
        print(f"Erfolg! {len(countries)} Länder gefunden.")
        # Zeige die ersten 3 Länder als Beweis
        for c in countries[:3]:
            print(f"Land: {c['name']} (ID: {c['id']})")
    else:
        print(f"Fehler: {response.status_code}")
        print(f"Details: {response.text}")

if __name__ == "__main__":
    test_connection()