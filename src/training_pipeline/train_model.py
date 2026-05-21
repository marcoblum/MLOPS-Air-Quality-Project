import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import hopsworks
from dotenv import load_dotenv

# Zertifikats-Fix für Windows (lokal)
# Zertifikats-Fix für Windows (lokal) und Linux (GitHub)
if os.name == 'nt':
    if not os.path.exists("C:\\tmp"): os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    # Für Linux / GitHub Action
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def train_model():
    print("--- START TRAINING PIPELINE ---")
    load_dotenv()
    
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="AeroPredict"
    )
    fs = project.get_feature_store()
    
    print("Lade Daten von Hopsworks...")
    air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
    
    # Wir lesen die Daten. Da wir in der Feature Pipeline 'wait=True' nutzen,
    # wird dieses Skript hier erst funktionieren, wenn der Job fertig ist.
    try:
        df = air_quality_fg.read()
        print(f"{len(df)} Zeilen erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # WICHTIG: Spaltennamen sortieren und Zeit konvertieren
    time_col = 'timestamp' 
    df = df.sort_values(by=time_col)
    df[time_col] = pd.to_datetime(df[time_col])

    # FEATURE ENGINEERING (angepasst an Pivot-Format)
    # Wir nutzen jetzt 'pm25' statt 'value'
    df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24).mean()
    df['pm25_lag_1h'] = df['pm25'].shift(1)
    df['pm25_lag_24h'] = df['pm25'].shift(24)
    
    # Zeit-Features
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    
    # Zielvariable: PM25 Durchschnitt der NÄCHSTEN 24 Stunden (Vorhersage-Target)
    df['target_24h_mean'] = df['pm25'].rolling(window=24).mean().shift(-24)

    # Wir nehmen Temperatur und Feuchtigkeit als zusätzliche Features mit rein!
    features = [
        'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
        'hour', 'day_of_week', 'temperature', 'relativehumidity'
    ]
    target = 'target_24h_mean'
    
    # Zeilen mit NaN (durch rolling/shift) entfernen
    df = df.dropna(subset=features + [target])

    if len(df) < 50: # Ein bisschen mehr Daten sollten es für RF schon sein
        print(f"Zu wenig Daten nach Vorbereitung: {len(df)} Zeilen.")
        return

    X = df[features]
    y = df[target]

    # TimeSeriesSplit für ehrliche Validierung
    tscv = TimeSeriesSplit(n_splits=3)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Starte Training auf {len(X_train)} Zeilen...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Metriken berechnen
    y_pred = model.predict(X_test)
    print(f"Training erfolgreich! R2-Score: {r2_score(y_test, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    
    # Modell lokal speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/air_quality_model.pkl")
    print("Modell unter models/air_quality_model.pkl gespeichert.")

if __name__ == "__main__":
    train_model()