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

    # 3. Pivot: Parameter als Spalten
    df_pivot = df.pivot_table(index='timestamp', columns='parameter', values='value')
    
    # Spaltennamen bereinigen
    df_pivot.columns = [str(col).lower().replace(".", "").replace(" ", "") for col in df_pivot.columns]
    df_pivot = df_pivot.sort_index()

    # WICHTIG: Den Index explizit als Datetime setzen für die Zeit-Berechnungen
    df_pivot.index = pd.to_datetime(df_pivot.index)

    print(f"Bereinigte Spalten gefunden: {df_pivot.columns.tolist()}")

    # 4. Feature Engineering & Imputation
    if 'pm25' in df_pivot.columns:
        # --- ZUERST: Lücken füllen (Imputation) ---
        # Füllt Lücken bis zu 3 Stunden, um Datenpunkte bei kurzen API-Ausfällen zu retten
        df_pivot['pm25'] = df_pivot['pm25'].ffill(limit=3).bfill(limit=3)
        
        # --- NEU: Langzeit-Features (Lags) ---
        df_pivot['pm25_lag_24h'] = df_pivot['pm25'].shift(24)   # Gleiche Uhrzeit gestern
        df_pivot['pm25_lag_7d'] = df_pivot['pm25'].shift(24*7) # Gleiche Uhrzeit letzte Woche
        
        # Bestehende Kurzzeit-Features
        df_pivot['pm25_rolling_24h_mean'] = df_pivot['pm25'].rolling(window=24, min_periods=1).mean()
        df_pivot['pm25_lag_1h'] = df_pivot['pm25'].shift(1)
        
        # Targets (Was wir vorhersagen wollen)
        df_pivot['target_next_hour'] = df_pivot['pm25'].shift(-1)
        df_pivot['target_24h_mean'] = df_pivot['pm25'].shift(-24).rolling(window=24, min_periods=1).mean()

    # Wetterdaten ebenfalls glätten
    if 'temperature' in df_pivot.columns:
        df_pivot['temperature'] = df_pivot['temperature'].ffill(limit=3)
        df_pivot['temp_lag_1h'] = df_pivot['temperature'].shift(1)
        
    if 'relativehumidity' in df_pivot.columns:
        df_pivot['relativehumidity'] = df_pivot['relativehumidity'].ffill(limit=3)
        df_pivot['hum_lag_1h'] = df_pivot['relativehumidity'].shift(1)

    # 1. Doppelte Zeitstempel entfernen
    df_pivot = df_pivot[~df_pivot.index.duplicated(keep='first')]

    # 2. Zeit-Features
    df_pivot['hour'] = df_pivot.index.hour
    df_pivot['day_of_week'] = df_pivot.index.dayofweek
    df_pivot['is_weekend'] = (df_pivot.index.dayofweek >= 5).astype(int)

    # 5. Cleanup
    # Wir löschen jetzt nur noch Zeilen, wo die Targets ODER die neue 7-Tage-Historie fehlen
    if 'target_next_hour' in df_pivot.columns and 'target_24h_mean' in df_pivot.columns:
        cols_to_check = ['target_next_hour', 'target_24h_mean']
        if 'pm25_lag_7d' in df_pivot.columns:
            cols_to_check.append('pm25_lag_7d')
            
        df_pivot = df_pivot.dropna(subset=cols_to_check)
    
    # Finaler Check gegen restliche kleine Lücken
    df_pivot = df_pivot.ffill().bfill()

    # 6. Speichern
    os.makedirs("data/processed", exist_ok=True)
    df_pivot.to_parquet("data/processed/features_latest.parquet")
    
    print("\n--- Feature Transformation Erfolgreich ---")
    print(f"Features & Targets berechnet für {len(df_pivot)} Zeitpunkte.")
    print(f"Neue Spalten: {df_pivot.columns.tolist()}")

if __name__ == "__main__":
    compute_features()