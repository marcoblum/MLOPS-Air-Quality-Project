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
    if not os.path.exists("C:\\tmp"): os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def fetch_historical_weather(start_date, end_date):
    """Holt historische Wind- und Wetterdaten von Open-Meteo."""
    print("🌍 Rufe zusätzliche meteorologische Daten (Wind, Luftdruck) via Open-Meteo API ab...")
    
    # Region Zentralschweiz / Zürich
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
    print("--- START TRAINING PIPELINE (WIND MODE - NO CYCLIC) ---")
    load_dotenv()
    
    df = None
    
    backup_path = "../feature_pipeline/data/processed/features_latest.parquet"
    if not os.path.exists(backup_path):
        backup_path = "data/processed/features_latest.parquet"
        
    if os.path.exists(backup_path):
        print(f"Lade lokal prozessiertes Gold-Standard-File: {backup_path}")
        df = pd.read_parquet(backup_path)
    else:
        print("❌ Kritischer Fehler: Keine Datenbasis gefunden!")
        return

    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # --- EXTERNE WETTERDATEN HOLLEN & MERGEN ---
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    weather_df = fetch_historical_weather(min_date, max_date)
    
    if weather_df is not None:
        print("🔗 Führe Time-Series Join zwischen Luftqualität und Winddaten aus...")
        df['timestamp_match'] = df['timestamp'].dt.floor('h').dt.tz_localize(None)
        weather_df['timestamp_match'] = weather_df['timestamp'].dt.floor('h')
        
        df = pd.merge(df, weather_df[['timestamp_match', 'wind_speed', 'wind_direction', 'surface_pressure']], 
                      on='timestamp_match', how='left')
        df = df.drop(columns=['timestamp_match'])
    else:
        print("⚠️ Fahre ohne zusätzliche Winddaten fort (Fallback).")

    target = 'target_24h_mean'
    if target not in df.columns:
        print(f"Kritischer Fehler: '{target}' fehlt!")
        return

    # --- ADVANCED FEATURE ENGINEERING (OHNE SIN/COS) ---
    print("Berechne physikalische Interaktionsterme...")
    
    if 'wind_speed' in df.columns and not df['wind_speed'].isna().all():
        # Interaktion: Feinstaub-Vorstunde dividiert durch Windgeschwindigkeit (Stau-Effekt bei Windstille)
        df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
        # Interaktion: Temperatur und Luftfeuchtigkeit
        df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']
        
        # Features definieren (Zurück zu normalen hour und day_of_week Variablen!)
        features = [
            'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
            'hour', 'day_of_week', 'temperature', 'relativehumidity', 
            'surface_pressure', 'wind_speed', 'wind_direction',
            'pm25_wind_interaction', 'temp_humidity_interaction'
        ]
    else:
        # Fallback falls Wetter-API ausfällt
        features = [
            'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
            'hour', 'day_of_week', 'temperature', 'relativehumidity'
        ]
    
    df = df.dropna(subset=features + [target])
    print(f"Daten erfolgreich vorbereitet. Datensatz enthält {len(df)} Zeilen.")

    X = df[features]
    y = df[target]

    # --- CROSS VALIDATION ---
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []
    
    print(f"Starte Cross-Validation mit wind-optimiertem Feature-Set...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = XGBRegressor(
            n_estimators=500, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"Fold {fold+1} -> R2: {r2:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    print("\n--- FINALE EVALUATION (DURCHSCHNITT) ---")
    print(f"Mean R2-Score : {np.mean(r2s):.2f}")
    print(f"Mean RMSE      : {np.mean(rmses):.2f} µg/m³")
    print(f"Mean MAE       : {np.mean(maes):.2f} µg/m³")
    
    print("\nTrainiere finales Champion-Modell auf allen Daten...")
    final_model = XGBRegressor(
        n_estimators=500, 
        learning_rate=0.01, 
        max_depth=5, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/air_quality_model.pkl")
    print("Modell erfolgreich unter models/air_quality_model.pkl überschrieben.")

if __name__ == "__main__":
    train_model()