import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_model():
    # 1. Daten laden
    df = pd.read_parquet("data/processed/features_latest.parquet")
    
    # --- SCHRITT 2: Target definieren & Daten vorbereiten ---
    
    # Das Modell soll lernen, den PM2.5 Wert der NÄCHSTEN Stunde vorherzusagen
    # .shift(-1) zieht den Wert aus der Zukunft eine Zeile hoch in die aktuelle Zeile
    df['target'] = df['pm25'].shift(-1)
    
    # Durch das Shifting hat die allerletzte Zeile kein Target mehr (NaN)
    # Zudem entfernen wir Zeilen, die durch Lags in compute_features entstanden sind
    df = df.dropna()

    # --- SCHRITT 3: Features (X) und Target (y) definieren ---

    # Hier listest du alle "Hinweise" auf, die das Modell nutzen darf.
    # Wir nehmen die neuen Wetter-Features aus Zürich mit dazu!
    feature_cols = [
        'pm25_rolling_24h', 
        'pm25_lag_1h', 
        'temp_lag_1h', 
        'hum_lag_1h'
    ]
    
    X = df[feature_cols]  # Die Eingabedaten (Features)
    y = df['target']      # Die Zielvariable (was vorhergesagt werden soll)
    
    # 3. Features (X) und Target (y) trennen
    # Wir nutzen den aktuellen Wert, den 24h Schnitt und den 1h-Lag
    feature_cols = ['pm25', 'pm25_rolling_24h', 'pm25_lag_1h']
    X = df[feature_cols]
    y = df['target']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. Modell trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training startet...")
    model.fit(X_train, y_train)
    
    # 6. Evaluieren
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Modell fertig! Mittlerer Fehler (MAE): {mae:.2f} µg/m³")
    
    # 7. Modell speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/air_quality_model.pkl")
    print("Modell unter 'models/air_quality_model.pkl' gespeichert.")

if __name__ == "__main__":
    train_model()