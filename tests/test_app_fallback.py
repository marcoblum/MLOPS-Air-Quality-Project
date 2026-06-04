import os
import pandas as pd
import numpy as np

# 1. Wir kopieren uns exakt deine Bereinigungs-Funktion aus der neuen app.py
def fill_missing_features_test(df):
    if df is None: 
        return None
    df.columns = [c.lower() for c in df.columns]
    
    # Das ist dein neuer Sicherheits-Schutz:
    required_base_wetter = {
        'temperature': 15.0,
        'relativehumidity': 70.0,
        'wind_speed': 5.0,
        'wind_direction': 180.0,
        'surface_pressure': 1013.25
    }
    for col, default_val in required_base_wetter.items():
        if col not in df.columns:
            df[col] = default_val

    # Imputation
    cols_to_impute = ['pm25', 'temperature', 'relativehumidity', 'wind_speed', 'wind_direction', 'surface_pressure']
    for col in cols_to_impute:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    if 'pm25' in df.columns:
        if 'pm25_rolling_24h_mean' not in df.columns:
            df['pm25_rolling_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()
        if 'pm25_rolling_24h_var' not in df.columns:
            df['pm25_rolling_24h_var'] = df['pm25'].rolling(window=24, min_periods=1).var().fillna(0)
        if 'pm25_lag_1h' not in df.columns:
            df['pm25_lag_1h'] = df['pm25'].shift(1).bfill()
        if 'pm25_lag_6h' not in df.columns:
            df['pm25_lag_6h'] = df['pm25'].shift(6).bfill()
        if 'pm25_lag_24h' not in df.columns:
            df['pm25_lag_24h'] = df['pm25'].shift(24).bfill()
                
        # Interaktionen berechnen
        df['pm25_wind_interaction'] = df['pm25_lag_1h'] / (df['wind_speed'] + 1.0)
        df['temp_humidity_interaction'] = df['temperature'] * df['relativehumidity']
                
    return df

def run_integration_test():
    print("🧪 Starte Fallback-Test für app.py...")
    
    # 2. Wir erstellen kaputte Testdaten (NUR pm25, KEINE Wetterdaten!)
    test_data = {
        'timestamp': pd.date_range(start="2026-06-01", periods=30, freq='h'),
        'pm25': np.random.uniform(5.0, 25.0, size=30)
    }
    df_broken = pd.DataFrame(test_data)
    
    print("📊 Erstellter Test-Datensatz enthält Spalten:", df_broken.columns.tolist())
    print("⚠️ Wetterspalten (wind_speed, surface_pressure, etc.) fehlen absichtlich!")

    # 3. Testfunktion ausführen
    print("\n⚙️ Führe 'fill_missing_features()' aus...")
    df_repaired = fill_missing_features_test(df_broken)
    
    # Zeit-Features wie in der app.py hinzufügen
    df_repaired['hour'] = df_repaired['timestamp'].dt.hour
    df_repaired['day_of_week'] = df_repaired['timestamp'].dt.dayofweek

    # 4. Überprüfung (Asserts), ob alle 14 Modell-Features jetzt da sind
    expected_features = [
        'pm25_rolling_24h_mean', 'pm25_rolling_24h_var', 
        'pm25_lag_1h', 'pm25_lag_6h', 'pm25_lag_24h',    
        'hour', 'day_of_week', 'temperature', 'relativehumidity', 
        'surface_pressure', 'wind_speed', 'wind_direction',
        'pm25_wind_interaction', 'temp_humidity_interaction'
    ]
    
    missing_after_repair = [col for col in expected_features if col not in df_repaired.columns]
    
    print("\n🔍 Überprüfe reparierte Spalten...")
    if len(missing_after_repair) == 0:
        print("✅ ERFOLG: Alle 14 notwendigen Features wurden erfolgreich generiert!")
        print(f"   -> Luftdruck-Fallback steht auf: {df_repaired['surface_pressure'].iloc[-1]} hPa")
        print(f"   -> Wind-Interaktion wurde berechnet: {df_repaired['pm25_wind_interaction'].iloc[-1]:.4f}")
    else:
        print(f"❌ TEST FEHLGESCHLAGEN: Folgende Spalten fehlen immer noch: {missing_after_repair}")

if __name__ == "__main__":
    run_integration_test()