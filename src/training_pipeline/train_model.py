import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import hopsworks
from dotenv import load_dotenv

# Zertifikats-Fix für Windows (lokal)
if os.name == 'nt':
    os.environ["HOPSWORKS_CLIENT_CERT_PATH"] = "C:\\tmp"

def train_model():
    print("--- VERSION 2 START (AUTOMATIC LOGIN) ---")
    load_dotenv()
    
    # LOGIN FIX: Wir erzwingen die Nutzung des API-Keys aus der Umgebung
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="AeroPredict"
    )
    fs = project.get_feature_store()
    
    print("Lade Daten von Hopsworks...")
    air_quality_fg = fs.get_feature_group(name="air_quality_features", version=1)
    df = air_quality_fg.read() 
    
    time_col = 'timestamp' 
    df = df.sort_values(by=time_col)

    # Einfaches Feature Engineering
    df['pm25_rolling_24h_mean'] = df['value'].rolling(window=24).mean()
    df['pm25_lag_1h'] = df['value'].shift(1)
    df['pm25_lag_24h'] = df['value'].shift(24)
    df['pm25_lag_7d'] = df['value'].shift(24 * 7)
    
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['target_24h_mean'] = df['value'].rolling(window=24).mean().shift(-24)

    features = ['pm25_rolling_24h_mean', 'pm25_lag_1h', 'pm25_lag_24h', 'pm25_lag_7d', 'hour', 'day_of_week', 'is_weekend']
    target = 'target_24h_mean'
    
    df = df.dropna(subset=features + [target])

    if len(df) < 10:
        print("Zu wenig Daten.")
        return

    X, y = df[features], df[target]
    tscv = TimeSeriesSplit(n_splits=3) # Kleiner Split zum Testen

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Training erfolgreich! R2: {r2_score(y_test, model.predict(X_test)):.2f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/air_quality_model.pkl")

if __name__ == "__main__":
    train_model()