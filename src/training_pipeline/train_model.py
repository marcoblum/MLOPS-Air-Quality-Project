import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from dotenv import load_dotenv

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"): os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

def train_model():
    print("--- START TRAINING PIPELINE (LOCAL TUNING MODE) ---")
    load_dotenv()
    
    # Pfad zur frisch generierten, 11'000 Zeilen starken Backup-Datei
    # Falls das Skript im Ordner 'src/training_pipeline' liegt, gehen wir zwei Ordner hoch und in den feature_pipeline-Ordner
    backup_path = "../feature_pipeline/data/processed/features_latest.parquet"
    
    # Fallback-Pfad, falls du das Skript aus dem Root-Verzeichnis startest
    if not os.path.exists(backup_path):
        backup_path = "data/processed/features_latest.parquet"

    if os.path.exists(backup_path):
        print(f"Lade lokal prozessiertes Gold-Standard-File: {backup_path}")
        df = pd.read_parquet(backup_path)
    else:
        print(f"Fehler: Die Datei '{backup_path}' wurde nicht gefunden. Bitte lasse zuerst die Feature-Pipeline laufen!")
        return

    # Spaltennamen zur Sicherheit in Kleinschreibung vereinheitlichen
    df.columns = [c.lower() for c in df.columns]

    # Zeitstempel sortieren (Dozenten-Feedback umgesetzt)
    time_col = 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Zielvariable (Target liegt nun fix in der Datei, keine Warnung mehr!)
    target = 'target_24h_mean'
    
    if target not in df.columns:
        print(f"Kritischer Fehler: '{target}' fehlt in der Parquet-Datei!")
        return

    # Features definieren, die wir beim Backfill berechnet haben
    features = [
        'pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 
        'hour', 'day_of_week', 'temperature', 'relativehumidity']
    
    # NaN-Werte entfernen
    df = df.dropna(subset=features + [target])

    print(f"Daten erfolgreich vorbereitet. Datensatz enthält {len(df)} saubere Zeilen für das Training.")

    X = df[features]
    y = df[target]

    # --- TIMERIES-SPLIT CROSS VALIDATION ---
    tscv = TimeSeriesSplit(n_splits=5)  # 5 Splits über die 11'000 Zeilen
    
    maes, rmses, r2s = [], [], []
    
    print(f"Starte TimeSeriesSplit Cross-Validation mit XGBoost...")
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # XGBoost Regressor mit optimalen Standard-Hyperparametern für Zeitreihen
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

    print("\n--- FINALE EVALUATION (DURCHSCHNITT ÜBER GANZES BACKFILL) ---")
    print(f"Mean R2-Score : {np.mean(r2s):.2f}")
    print(f"Mean RMSE      : {np.mean(rmses):.2f} µg/m³ (Geforderte Metrik!)")
    print(f"Mean MAE       : {np.mean(maes):.2f} µg/m³")
    
    # Trainiere das finale Modell auf ALLEN 11'000 Zeilen für die Produktion (Hugging Face)
    print("\nTrainiere finales XGBoost-Modell auf kompletten historischen Daten...")
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