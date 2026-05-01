import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model():
    # 1. Daten aus dem Feature Store laden
    df = pd.read_parquet("data/processed/features_latest.parquet")
    
    # Sicherstellen, dass die Zeit sortiert ist
    df = df.sort_index()

    # 2. Features (X) und Target (y) definieren
    # Wir fügen hour, day_of_week und is_weekend hinzu:
    features = [
    'pm25_rolling_24h_mean', 
    'pm25_lag_1h', 
    'pm25_lag_24h', 
    'pm25_lag_7d',
    'temp_lag_1h', 
    'hum_lag_1h', 
    'hour', 
    'day_of_week', 
    'is_weekend'
    ]
    target = 'target_24h_mean'

    # Falls 'no2' in deinen Daten stabil ist, kannst du es auch hinzufügen:
    if 'no2' in df.columns:
        df['no2_lag_1h'] = df['no2'].shift(1)
        features.append('no2_lag_1h')
    
    # Wichtig: Erst NaNs droppen, dann X und y zuweisen!
    df = df.dropna(subset=features + [target])

    # --- HIER WAR DER FIX: X und y definieren ---
    X = df[features]
    y = df[target]
    # --------------------------------------------

    # 3. TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"Starte Training mit TimeSeriesSplit auf {len(df)} Zeitpunkten...")

    # Wir durchlaufen die Splits, um am Ende das Modell auf dem aktuellsten Stand zu haben
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 4. Modell trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluieren
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Training Erfolgreich ---")
    print(f"MAE (Mittlerer Fehler): {mae:.2f} µg/m³")
    print(f"R² Score (Erklärte Varianz): {r2:.2f}")

    # 6. Modell speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/air_quality_model.pkl")
    print("Modell unter 'models/air_quality_model.pkl' gespeichert.")

if __name__ == "__main__":
    train_model()