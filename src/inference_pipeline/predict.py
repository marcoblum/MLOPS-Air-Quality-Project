import joblib
import pandas as pd
import os
from datetime import datetime

def make_prediction():
    # 1. Modell laden
    model_path = "models/air_quality_model.pkl"
    if not os.path.exists(model_path):
        print("Fehler: Kein trainiertes Modell gefunden!")
        return
    
    model = joblib.load(model_path)
    
    # 2. Aktuellste Daten aus dem Feature Store laden
    # Wir brauchen die berechneten Features (Lags & Rolling Mean)
    feature_store_path = "data/processed/features_latest.parquet"
    if not os.path.exists(feature_store_path):
        print("Fehler: Feature Store ist leer. Lass erst compute_features.py laufen.")
        return
    
    df = pd.read_parquet(feature_store_path)

    # Füge das nach dem Laden des Dataframes (df = pd.read_parquet...) ein:
    print(f"DEBUG: Der absolut neueste Zeitstempel im System: {df.index.max()}")
    print(f"DEBUG: Anzahl der Zeilen insgesamt: {len(df)}")
    df = df.sort_index()
    
    # Den allerletzten verfügbaren Datenpunkt nehmen
    latest_data = df.tail(1)
    current_time = latest_data.index[0]
    
    # 3. Features auswählen (Reihenfolge muss EXAKT wie im Training sein)
    # Wichtig: Wir nutzen die Namen aus deinem Training-Skript!
    feature_cols = [
        'pm25_rolling_24h_mean', 
        'pm25_lag_1h', 
        'temp_lag_1h', 
        'hum_lag_1h'
    ]
    
    # Überprüfen, ob alle Features da sind
    missing = [c for c in feature_cols if c not in latest_data.columns]
    if missing:
        print(f"Fehler: Folgende Features fehlen im Feature Store: {missing}")
        return

    X_latest = latest_data[feature_cols]
    
    # 4. Vorhersage treffen
    prediction = model.predict(X_latest)[0]
    
    # 5. Ausgabe verschönern
    print(f"\n--- Luftqualitäts-Vorhersage für Zürich Kaserne ---")
    print(f"Letzter Messzeitpunkt (UTC): {current_time}")
    print(f"Aktueller PM2.5 Wert: {latest_data['pm25'].values[0]:.2f} µg/m³")
    print(f"Aktuelle Temperatur: {latest_data['temperature'].values[0]:.1f} °C")
    print(f"---")
    # Da wir auf 'target_24h_mean' trainiert haben, ist das die Vorhersage für den 24h-Schnitt
    print(f"PROGNOSE für PM2.5 (Durchschnitt nächste 24h): {prediction:.2f} µg/m³")

    

if __name__ == "__main__":
    make_prediction()