import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch
from tests.conftest import make_mock_data


REQUIRED_FEATURES = [
    "pm25",
    "pm25_rolling_24h_mean",
    "pm25_rolling_24h_var",
    "pm25_lag_1h",
    "pm25_lag_6h",
    "pm25_lag_24h",
    "hour",
    "day_of_week",
    "temperature",
    "relativehumidity",
]


def test_required_columns_present():
    """Alle für das Modell benötigten Feature-Spalten müssen vorhanden sein."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    for col in REQUIRED_FEATURES:
        assert col in df.columns, f"Fehlende Spalte: {col}"


def test_time_features_correct():
    """hour und day_of_week müssen korrekt aus dem Timestamp extrahiert werden."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    assert df.loc[0, "hour"] == 0
    assert df.loc[0, "day_of_week"] == 3  # 2026-01-01 = Donnerstag
    assert df.loc[1, "hour"] == 1