import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_model():
    # 1. Daten laden
    df = pd.read_parquet("data/processed/features_latest.parquet")
    
    # 2. Target definieren: Wir wollen den PM25 Wert von der NÄCHSTEN Stunde vorhersagen
    # Dafür shiften wir die pm25 Spalte um -1
    df['target'] = df['pm25'].shift(-1)
    
    # Letzte Zeile löschen (da wir dort kein Target für die Zukunft haben)
    df = df.dropna()
    
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