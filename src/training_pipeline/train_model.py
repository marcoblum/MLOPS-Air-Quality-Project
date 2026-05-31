import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import requests
from dotenv import load_dotenv

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"): 
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def fetch_historical_weather(start_date, end_date):
    """Holt historische Wind- und Wetterdaten von Open-Meteo."""
    print("🌍 Rufe zusätzliche meteorologische Daten (Wind, Luftdruck) via Open-Meteo API ab...")
    lat, lon = 47.3769, 8.5417 
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_10m,wind_direction_10m,surface_pressure",
        "timezone": "UTC"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            hourly_data = response.json().get("hourly", {})
            weather_df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly_data.get("time")),
                "wind_speed": hourly_data.get("wind_speed_10m"),
                "wind_direction": hourly_data.get("wind_direction_10m"),
                "surface_pressure": hourly_data.get("surface_pressure")
            })
            print(f"✅ Wetterdaten erfolgreich geladen: {len(weather_df)} Zeilen.")
            return weather_df
        else:
            print(f"⚠️ Wetter API Fehler: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Fehler beim Abruf der Wetterdaten: {e}")
        return None

def train_model():
    print("--- START TRAINING PIPELINE (FULL PROPOSAL COMPLIANCE MODE) ---")
    load_dotenv()
    
    # Direktes Laden deines grossen 2-Jahres Gold-Standard Backfill Files
    features_path = "data/processed/history/features_latest_backfill.parquet"
    
    if os.path.exists(features_path):
        print(f"Lade lokal prozessiertes Gold-Standard-File: {features_path}")
        df = pd.read_parquet(features_path)
    else:
        print(f"❌ Kritischer Fehler: Keine Datenbasis unter {features_path} gefunden!")
        return

    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # --- ERGÄNZUNG: REKONSTRUKTION & FEATURE ENGINEERING ---
    print("Berechne zusätzliche Proposal-Features (6h-Lag & 24h-Varianz)...")
    df['pm25_reconstructed'] = df['pm25_lag_1h'].shift(-1)
    df['pm25_reconstructed'] = df['pm25_reconstructed'].ffill()
    
    df['pm25_lag_6h'] = df['pm25_reconstructed'].shift(6)
    df['pm25_rolling_24h_var'] = df['pm25_reconstructed'].rolling(window=24).var()
    df = df.drop(columns=['pm25_reconstructed'])

    # --- 1. ERST METEOROLOGIE-DATEN HOHLEN & REINMERGEN ---
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    weather_df = fetch_historical_weather(min_date, max_date)
    
    if weather_df is not None:
        print("🔗 Führe Time-Series Join zwischen Luftqualität und Winddaten aus...")
        df['timestamp_match'] = df['timestamp'].dt.floor('h').dt.tz_localize(None)
        weather_df['timestamp_match'] = weather_df['timestamp'].dt.floor('h')
        
        # Bereinigung: Falls alte leere Wind-Spalten im df existieren, werfen wir sie vor dem Merge raus
        for col in ['wind_speed', 'wind_direction', 'surface_pressure']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        weather_cols = ['timestamp_match', 'wind_speed', 'wind_direction', 'surface_pressure']
        df = pd.merge(df, weather_df[weather_cols], on='timestamp_match', how='left')
        df = df.drop(columns=['timestamp_match'])

    # Spaltennamen nochmals radikal komplett kleinschreiben zur Sicherheit
    df.columns = [c.lower() for c in df.columns]
    
    # Diagnostik-Print: Zeigt uns im Terminal exakt an, was im DataFrame existiert
    print("Vorhandene Spalten im Datensatz:", df.columns.tolist())

    # --- 2. JETZT DIE FEATURE SELECTION (NACHDEM DIE SPALTEN EXISTIEREN) ---
    if 'wind_speed' in df.columns and not df['wind_speed'].isna().all():
        print("✨ Wind-Features erfolgreich erkannt! Berechne Interaktionsterme für das Champion-Modell...")
        
        # Fehlende Werte in den Wetterdaten absichern, damit dropna uns nicht den Datensatz leert
        df['wind_speed'] = df['wind_speed'].ffill().bfill().fillna(0)
        df['wind_direction'] = df['wind_direction'].ffill().bfill().fillna(0)
        df['surface_pressure'] = df['surface_pressure'].ffill().bfill().fillna(1013.25)
        
        df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
        df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']
        
        features = [
            'pm25_rolling_24h_mean', 'pm25_rolling_24h_var', 
            'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',    
            'hour', 'day_of_week', 'temperature', 'relativehumidity', 
            'surface_pressure', 'wind_speed', 'wind_direction',
            'pm25_wind_interaction', 'temp_humidity_interaction'
        ]
    else:
        print("⚠️ Keine Wind-Features gefunden. Wechsle zu Basis-Features.")
        features = [
            'pm25_rolling_24h_mean', 'pm25_rolling_24h_var',
            'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',
            'hour', 'day_of_week', 'temperature', 'relativehumidity'
        ]
    
    target = 'target_24h_mean'
    df = df.dropna(subset=features + [target])
    print(f"Daten erfolgreich vorbereitet. Datensatz enthält {len(df)} Zeilen.")

    X = df[features]
    y = df[target]

    # --- CROSS VALIDATION ---
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []
    
    print(f"Starte Cross-Validation auf {len(features)} Features...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = XGBRegressor(
            n_estimators=400, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))
        print(f"Fold {fold+1} -> R2: {r2s[-1]:.2f} | RMSE: {rmses[-1]:.2f}")

    print("\n--- FINALE EVALUATION (DURCHSCHNITT) ---")
    print(f"Mean R2-Score : {np.mean(r2s):.2f}")
    print(f"Mean RMSE      : {np.mean(rmses):.2f} µg/m³")
    print(f"Mean MAE       : {np.mean(maes):.2f} µg/m³")
    
    print("\nTrainiere finales Champion-Modell auf allen Daten...")
    final_model = XGBRegressor(
        n_estimators=400, 
        learning_rate=0.01, 
        max_depth=5, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    # Sicherstellen, dass das Modell immer im globalen Projekt-Ordner landet
    # Unabhängig davon, aus welchem Unterordner das Skript gestartet wird
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_output_path = os.path.join(model_dir, "air_quality_model.pkl")
    joblib.dump(final_model, model_output_path)
    print(f"🎯 Erfolg! Modell mit allen Spalten trainiert und gespeichert unter:\n -> {model_output_path}")

if __name__ == "__main__":
    train_model()