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
    
    # 2. Aktuellste Daten laden
    feature_store_path = "data/processed/features_latest.parquet"
    if not os.path.exists(feature_store_path):
        print("Fehler: Feature Store ist leer. Lass erst compute_features.py laufen.")
        return
    
    df = pd.read_parquet(feature_store_path)

    print(f"DEBUG: Der absolut neueste Zeitstempel im System: {df.index.max() if df.index.name else 'Kein Indexname'}")
    print(f"DEBUG: Anzahl der Zeilen insgesamt: {len(df)}")
    
    # Sortieren nach Zeitstempel
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_values(by=df.index)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
    # Den allerletzten verfügbaren Datenpunkt nehmen
    latest_data = df.tail(1)
    current_time = latest_data['timestamp'].values[0] if 'timestamp' in latest_data.columns else latest_data.index[0]
    
    # 3. Features auswählen (EXAKT wie in der neuen train_model.py definiert!)
    feature_cols = [
        'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
        'hour', 'day_of_week', 'temperature', 'relativehumidity'
    ]
    
    # Überprüfen, ob alle Features da sind
    missing = [c for c in feature_cols if c not in latest_data.columns]
    if missing:
        print(f"Fehler: Folgende Features fehlen im Datensatz: {missing}")
        return

    X_latest = latest_data[feature_cols]
    
    # 4. Vorhersage treffen
    prediction = model.predict(X_latest)[0]
    
    # 5. Ausgabe
    print(f"\n--- Luftqualitäts-Vorhersage für Zürich Kaserne ---")
    print(f"Letzter Messzeitpunkt: {current_time}")
    print(f"Aktueller PM2.5 Wert: {latest_data['pm25'].values[0]:.2f} µg/m³")
    if 'temperature' in latest_data.columns:
        print(f"Aktuelle Temperatur: {latest_data['temperature'].values[0]:.1f} °C")
    print(f"---")
    print(f"PROGNOSE für PM2.5 (Durchschnitt nächste 24h): {prediction:.2f} µg/m³")

if __name__ == "__main__":
    make_prediction()