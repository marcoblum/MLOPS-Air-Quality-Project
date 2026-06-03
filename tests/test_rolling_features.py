import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch
from tests.conftest import make_mock_data


def test_rolling_average_present():
    """Rolling Average Spalte muss vorhanden sein."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    assert "pm25_rolling_24h_mean" in df.columns
    assert "pm25_rolling_24h_var" in df.columns


def test_rolling_average_correct():
    """pm25_rolling_24h_mean soll dem gleitenden Durchschnitt der letzten 24h entsprechen."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    expected_mean = df["pm25"].iloc[:24].mean()
    actual_mean = df.loc[23, "pm25_rolling_24h_mean"]
    assert actual_mean == pytest.approx(expected_mean, rel=1e-3)