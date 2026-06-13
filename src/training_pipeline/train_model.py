import os
import hopsworks
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

if os.name == 'nt':
    if not os.path.exists("C:\\tmp"):
        os.makedirs("C:\\tmp")
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"
else:
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "/tmp"

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def train_model():
    print("--- START TRAINING PIPELINE (HOPSWORKS LIVE RETRAIN MODE) ---")
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID")

    print("Verbinde mit Hopsworks Feature Store...")
    if not HOPSWORKS_API_KEY:
        raise ValueError("❌ HOPSWORKS_API_KEY fehlt in der .env Datei!")
    if not HOPSWORKS_PROJECT:
        raise ValueError("❌ HOPSWORKS_PROJECT fehlt in der .env Datei!")

    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()

        print("Lese aggregierten Gold-Datensatz aus 'air_quality_features_1'...")
        air_quality_fg = fs.get_feature_group(name="air_quality_features_1", version=1)
        df = air_quality_fg.read()
        print(f"✅ {len(df)} Zeilen erfolgreich direkt via REST geladen.")

    except Exception as e:
        print(f"❌ Kritischer Fehler beim Laden von Hopsworks: {e}")
        return

    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    print("Schliesse API-Lücken im Datensatz durch Vorwärts- und Rückwärts-Imputation...")
    cols_to_fill = [
        'pm25', 'temperature', 'relativehumidity', 'wind_speed',
        'wind_direction', 'surface_pressure', 'pm25_rolling_24h_mean', 'pm25_rolling_24h_var'
    ]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    if 'wind_speed' in df.columns and 'pm25_lag_1h' in df.columns:
        df['pm25_lag_1h'] = df['pm25_lag_1h'].ffill().bfill()
        df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
    if 'temperature' in df.columns and 'relativehumidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']

    print("Vorhandene Spalten im Datensatz:", df.columns.tolist())

    features = [
        'pm25_rolling_24h_mean', 'pm25_rolling_24h_var',
        'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',
        'hour', 'day_of_week', 'temperature', 'relativehumidity',
        'surface_pressure', 'wind_speed', 'wind_direction',
        'pm25_wind_interaction', 'temp_humidity_interaction'
    ]

    features = [f for f in features if f in df.columns]
    target = 'target_24h_mean'

    df = df.dropna(subset=features + [target])
    print(f"Daten erfolgreich vorbereitet. Trainings-Datensatz enthält {len(df)} Zeilen.")

    X = df[features]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []

    print(f"Starte Cross-Validation auf {len(features)} Features...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBRegressor(**XGB_PARAMS)
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

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "air_quality_model.pkl")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("air_quality_forecasting")

    with mlflow.start_run():
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_train_rows", len(df))

        mlflow.log_metrics({
            "cv_r2_mean": float(np.mean(r2s)),
            "cv_rmse_mean": float(np.mean(rmses)),
            "cv_mae_mean": float(np.mean(maes)),
        })

        print("\nTrainiere finales Modell auf allen verfügbaren Daten...")
        final_model = XGBRegressor(**XGB_PARAMS)
        final_model.fit(X, y)

        joblib.dump(final_model, model_output_path)
        print(f"Erfolg! Modell trainiert und gespeichert unter:\n -> {model_output_path}")

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="air_quality_xgboost",
            input_example=X.head(),
        )
        print("✅ Run-Metriken geloggt und neue Modellversion in MLflow registriert.")

    if HF_TOKEN and HF_REPO_ID:
        try:
            print("\nPushe aktuelles Champion-Modell (.pkl) direkt zu Hugging Face Spaces...")
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                token=HF_TOKEN,
                path_or_fileobj=model_output_path,
                path_in_repo="models/air_quality_model.pkl",
                repo_id=HF_REPO_ID,
                repo_type="space"
            )
            print("✅ Modell-Upload zu Hugging Face erfolgreich!")
        except Exception as hf_err:
            print(f"⚠️ Hugging Face Modell-Upload fehlgeschlagen: {hf_err}")
    else:
        print("\n⚠️ Kein HF_TOKEN oder HF_REPO_ID gefunden. Überspringe Upload zu Hugging Face.")


if __name__ == "__main__":
    train_model()