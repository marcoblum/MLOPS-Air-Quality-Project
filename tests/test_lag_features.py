import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch
from tests.conftest import make_mock_data


def test_lag_features_present():
    """Alle Lag-Spalten müssen im Output vorhanden sein."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    assert "pm25_lag_1h" in df.columns
    assert "pm25_lag_6h" in df.columns
    assert "pm25_lag_24h" in df.columns


def test_lag_1h_correct():
    """pm25_lag_1h muss den Wert der vorherigen Stunde enthalten."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    lag_val = df.loc[5, "pm25_lag_1h"]
    pm25_val = df.loc[4, "pm25"]
    assert lag_val == pytest.approx(pm25_val, rel=1e-3)