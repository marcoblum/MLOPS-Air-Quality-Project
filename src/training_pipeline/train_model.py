import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import hopsworks
from dotenv import load_dotenv

# Zertifikats-Pfad für Windows/Linux absichern
if os.name == 'nt':
    if not os.path.exists("C:\\tmp"): os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def train_model():
    print("--- START TRAINING PIPELINE (FULL DATA MODE) ---")
    load_dotenv()
    
    df = None
    
    # 1. HAUPTWEG: Lade den KOMPLETTEN 2-Jahres-Backfill live aus dem Offline-Store
    try:
        print("Verbinde mit Hopsworks und lade gesamten historischen Datensatz...")
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AeroPredict")
        fs = project.get_feature_store()
        air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
        
        # FIX: Wir zwingen Hopsworks, direkt aus der Hive/Hudi-Infrastruktur zu lesen.
        # Das umgeht den stündlichen Lese-Cache und holt alle 9.500+ Zeilen!
        df = air_quality_fg.read(read_options={"use_api": False, "read_from_hive": True})
        print(f"✅ ERFOLG: {len(df)} Zeilen live aus dem Hopsworks Offline-Store geladen!")
    except Exception as e:
        print(f"⚠️ Hopsworks-Live-Abruf aktuell blockiert oder fehlgeschlagen ({e}).")
        print("-> Schalte um auf automatischen MLOps-Fallback auf lokales Parquet-File...")

    # 2. FALLBACK: Falls Hopsworks gestreikt hat, nutze das lokale Backup
    if df is None or df.empty:
        backup_path = "../feature_pipeline/data/processed/features_latest.parquet"
        if not os.path.exists(backup_path):
            backup_path = "data/processed/features_latest.parquet"
            
        if os.path.exists(backup_path):
            print(f"Lade lokal prozessiertes Gold-Standard-File: {backup_path}")
            df = pd.read_parquet(backup_path)
        else:
            print("❌ Kritischer Fehler: Keine Datenbasis (weder Hopsworks noch lokales Backup) gefunden!")
            return

    # Spaltennamen zur Sicherheit in Kleinschreibung vereinheitlichen
    df.columns = [c.lower() for c in df.columns]

    # Zeitstempel sortieren (Dozenten-Feedback umgesetzt)
    time_col = 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Zielvariable
    target = 'target_24h_mean'
    if target not in df.columns:
        print(f"Kritischer Fehler: '{target}' fehlt im Datensatz!")
        return

    # Features definieren, die wir beim Backfill berechnet haben
    features = [
        'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
        'hour', 'day_of_week', 'temperature', 'relativehumidity'
    ]
    
    # NaN-Werte entfernen
    df = df.dropna(subset=features + [target])
    print(f"Daten erfolgreich vorbereitet. Datensatz enthält {len(df)} saubere Zeilen für das Training.")

    X = df[features]
    y = df[target]

    # --- TIMERIES-SPLIT CROSS VALIDATION ---
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []
    
    print(f"Starte TimeSeriesSplit Cross-Validation mit XGBoost...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.03, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metriken für dieses Fold berechnen
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"Fold {fold+1} -> R2: {r2:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    print("\n--- FINALE EVALUATION (DURCHSCHNITT) ---")
    print(f"Mean R2-Score : {np.mean(r2s):.2f}")
    print(f"Mean RMSE      : {np.mean(rmses):.2f} µg/m³ (Geforderte Metrik!)")
    print(f"Mean MAE       : {np.mean(maes):.2f} µg/m³")
    
    # Trainiere das finale Modell auf ALLEN Zeilen für die Produktion (Hugging Face)
    print("\nTrainiere finales Champion-Modell auf kompletten historischen Daten...")
    final_model = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.03, 
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    # Modell abspeichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/air_quality_model.pkl")
    print("Modell erfolgreich unter models/air_quality_model.pkl überschrieben.")

if __name__ == "__main__":
    train_model()