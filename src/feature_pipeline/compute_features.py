import pandas as pd
import glob
import os

def compute_features():
    # 1. Alle Rohdaten-Dateien einsammeln
    files = glob.glob("data/raw/*.parquet")
    if not files:
        print("Keine Rohdaten gefunden!")
        return

    print(f"Lade {len(files)} Dateien...")
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list).drop_duplicates()

    # 2. Zeitstempel sortieren
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # 3. Pivot: Wir wollen Parameter als Spalten (pm25, no2)
    df_pivot = df.pivot_table(index='timestamp', columns='parameter', values='value')

    # 4. Feature Engineering (wie im Proposal versprochen!)
    # Berechne den gleitenden Durchschnitt der letzten 24 Stunden
    if 'pm25' in df_pivot.columns:
        # PM2.5 Trends
        df_pivot['pm25_rolling_24h'] = df_pivot['pm25'].rolling(window=24, min_periods=1).mean()
        df_pivot['pm25_lag_1h'] = df_pivot['pm25'].shift(1)

    # Neu: Wetter-Features (Falls vorhanden)
    if 'temperature' in df_pivot.columns:
        df_pivot['temp_lag_1h'] = df_pivot['temperature'].shift(1)
        
    if 'relativehumidity' in df_pivot.columns:
        df_pivot['hum_lag_1h'] = df_pivot['relativehumidity'].shift(1)

    # NEU: Zeilen mit NaN-Werten entfernen, die durch das Shifting/Rolling entstanden sind
    df_pivot = df_pivot.dropna()

    # 5. Speichern als "Cleaned Features"
    os.makedirs("data/processed", exist_ok=True)
    df_pivot.to_parquet("data/processed/features_latest.parquet")
    
    print("\n--- Feature Transformation Erfolgreich ---")
    print(f"Features berechnet für {len(df_pivot)} Zeitpunkte.")
    print(df_pivot.tail())

if __name__ == "__main__":
    compute_features()