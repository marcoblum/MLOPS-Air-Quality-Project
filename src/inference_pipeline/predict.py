import joblib
import pandas as pd
import os

def make_prediction():
    # 1. Das aktuellste Modell laden
    model_path = "models/air_quality_model.pkl"
    if not os.path.exists(model_path):
        print("Fehler: Kein trainiertes Modell gefunden! Lass erst das Training laufen.")
        return
    
    model = joblib.load(model_path)
    
    # ... (Modell und Features laden wie bisher)
    
    # 2. Die Features auswählen, die das Modell nun erwartet
    # WICHTIG: Die Reihenfolge und Namen müssen exakt wie im Training sein!
    feature_cols = [
        'pm25_rolling_24h', 
        'pm25_lag_1h', 
        'temp_lag_1h', 
        'hum_lag_1h'
    ]
    
    X_latest = latest_data[feature_cols]
    
    # 3. Vorhersage treffen
    prediction = model.predict(X_latest)[0]
    
    # Ausgabe verschönern
    print(f"\n--- Luftqualitäts-Vorhersage für Zürich Kaserne ---")
    print(f"Aktueller Zeitpunkt: {current_time}")
    print(f"Aktueller PM2.5 Wert: {latest_data['pm25'].values[0]:.2f} µg/m³")
    print(f"Aktuelle Temperatur: {latest_data['temperature'].values[0]:.1f} °C")
    print(f"---")
    print(f"VORHERSAGE für PM2.5 in 1 Stunde: {prediction:.2f} µg/m³")

if __name__ == "__main__":
    make_prediction()