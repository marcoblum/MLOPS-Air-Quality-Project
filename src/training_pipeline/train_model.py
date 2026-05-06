import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import hopsworks
from dotenv import load_dotenv

# 1. Windows-Fix für Zertifikate (falls lokal ausgeführt)
if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"

def train_model():
    load_dotenv()
    
    # 2. Verbindung zu Hopsworks herstellen
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="AeroPredict"
    )
    fs = project.get_feature_store()
    
    # 3. Daten aus dem Feature Store laden (statt lokaler Parquet-Datei)
    # Das ist das Herzstück deiner FTI-Architektur
    print("Lade Daten von Hopsworks...")
    air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
    df = air_quality_fg.read() 
    
    # Sicherstellen, dass die Zeit sortiert ist
    # WICHTIG: Hopsworks gibt 'timestamp' zurück (siehe dein Terminal-Log)
    time_col = 'timestamp' 
    df = df.sort_values(by=time_col)

    # 4. Feature Engineering (Lags berechnen)
    # Da wir nun die 4 Jahre Historie haben, funktionieren die Lags!
    df['pm25_rolling_24h_mean'] = df['value'].rolling(window=24).mean()
    df['pm25_lag_1h'] = df['value'].shift(1)
    df['pm25_lag_24h'] = df['value'].shift(24)
    df['pm25_lag_7d'] = df['value'].shift(24 * 7)
    
    # Zeit-Features
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Zielwert: Durchschnitt der nächsten 24 Stunden
    df['target_24h_mean'] = df['value'].rolling(window=24).mean().shift(-24)

    # 5. Features (X) und Target (y) definieren
    features = [
        'pm25_rolling_24h_mean', 
        'pm25_lag_1h', 
        'pm25_lag_24h', 
        'pm25_lag_7d',
        'hour', 
        'day_of_week', 
        'is_weekend'
    ]
    target = 'target_24h_mean'
    
    # NaNs droppen (entstehen durch Lags/Rolling Windows)
    df = df.dropna(subset=features + [target])

    if len(df) < 100:
        print(f"FEHLER: Zu wenig Daten für Training ({len(df)} Zeilen).")
        return

    X = df[features]
    y = df[target]

    # 6. TimeSeriesSplit & Training
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"Starte Training auf {len(df)} Zeitpunkten...")

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 7. Evaluieren
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Training Erfolgreich ---")
    print(f"MAE: {mae:.2f} µg/m³")
    print(f"R² Score: {r2:.2f}")

    # 8. Modell speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/air_quality_model.pkl")
    print("Modell lokal gespeichert.")

if __name__ == "__main__":
    train_model()