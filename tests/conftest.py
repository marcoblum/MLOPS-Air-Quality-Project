import pandas as pd


def make_mock_data(hours=24):
    """Erstellt synthetische Testdaten für einen gegebenen Zeitraum."""
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h")
    data = []
    for i, ts in enumerate(timestamps):
        data.append({"timestamp": ts.isoformat(), "parameter": "pm25", "value": float(i + 1)})
        data.append({"timestamp": ts.isoformat(), "parameter": "temperature", "value": 15.0 + i * 0.1})
        data.append({"timestamp": ts.isoformat(), "parameter": "relativehumidity", "value": 60.0})
    return data