import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

# 1. Umgebung laden & Zertifikat-Pfad setzen (exakt wie in deiner Pipeline)
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def read_data_from_hopsworks():
    print("🔄 Verbinde mit Hopsworks (Projekt: AeroPredict)...")
    
    if not HOPSWORKS_API_KEY:
        raise ValueError("❌ HOPSWORKS_API_KEY fehlt in der .env Datei!")

    try:
        # 2. Login ins spezifische Projekt
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="AeroPredict")
        fs = project.get_feature_store()
        print("✅ Verbindung zum Feature Store erfolgreich.")

        # 3. Feature Group referenzieren
        print("📦 Hole Feature Group 'air_quality_features_1'...")
        air_quality_fg = fs.get_feature_group(
            name="air_quality_features_1",
            version=1
        )

        # 4. Gesamten 2-Jahres-Backfill auslesen
        print("📥 Lese Daten aus Hopsworks (ohne Spark, direkt via REST)...")
        # .read() holt die Daten ohne Spark-Cluster-Infrastruktur direkt als Pandas DataFrame
        df = air_quality_fg.read()
        
        print(f"\n🎉 Erfolg! Daten erfolgreich geladen.")
        print(f" -> Anzahl Zeilen: {len(df)}")
        print(f" -> Vorhandene Spalten: {df.columns.tolist()}")
        
        # Zeige die neuesten Zeilen an
        print("\n--- Letzte 5 Zeilen (Auszug) ---")
        print(df.tail())
        
        return df

    except Exception as e:
        print(f"❌ Fehler beim Lesen aus Hopsworks: {e}")
        return None

if __name__ == "__main__":
    read_data_from_hopsworks()