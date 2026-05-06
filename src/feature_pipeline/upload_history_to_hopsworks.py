import os
import glob
import pandas as pd
from dotenv import load_dotenv

# 1. WINDOWS PFAD-HACK (Muss ganz oben stehen!)
# Erstellt C:\tmp falls nicht vorhanden und setzt den Zertifikats-Pfad
if not os.path.exists("C:\\tmp"):
    os.makedirs("C:\\tmp")
os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"

# Erst jetzt Hopsworks importieren
import hopsworks

def run_upload():
    # 2. Umgebung laden & Login
    load_dotenv()
    
    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="AeroPredict"
        )
        fs = project.get_feature_store()
        print(f"ERFOLG! Verbunden mit Projekt: {project.name}")
    except Exception as e:
        print(f"Login fehlgeschlagen: {e}")
        return

    # 3. Lokale Daten laden
    file_pattern = os.path.join("data", "raw", "history_*.parquet")
    files = glob.glob(file_pattern)

    if not files:
        print("Keine Historiendaten unter data/raw/ gefunden!")
        return

    print(f"{len(files)} Dateien gefunden. Lade Daten in den Speicher...")
    df_list = [pd.read_parquet(f) for f in files]
    df_history = pd.concat(df_list, ignore_index=True)

    # 4. Spaltennamen-Check (Verhindert KeyError)
    # Wir schauen nach, wie deine Zeit-Spalte wirklich heißt
    cols = df_history.columns.tolist()
    print(f"Verfügbare Spalten: {cols}")
    
    # Automatisches Mapping: Wir suchen nach 'date' oder 'datetime'
    target_col = None
    for col in ['date', 'datetime', 'timestamp']:
        if col in cols:
            target_col = col
            break
    
    if not target_col:
        print("FEHLER: Keine Zeit-Spalte (date/datetime) gefunden!")
        return

    print(f"Nutze '{target_col}' als Zeitindex...")
    df_history[target_col] = pd.to_datetime(df_history[target_col])

    # 5. Feature Group in Hopsworks erstellen/holen
    # Das ist der zentrale Speicherort deiner FTI-Architektur
    air_quality_fg = fs.get_or_create_feature_group(
        name="air_quality_features",
        version=1,
        primary_key=[target_col], 
        description="Historische Luftqualitäts- und Wetterdaten",
        online_enabled=True, # Wichtig für die spätere Live-Inference
        event_time=target_col
    )

    # 6. Upload starten
    print("Starte Upload zu Hopsworks... Das kann einen Moment dauern.")
    air_quality_fg.insert(df_history)
    print("--- UPLOAD ERFOLGREICH! ---")

if __name__ == "__main__":
    run_upload()