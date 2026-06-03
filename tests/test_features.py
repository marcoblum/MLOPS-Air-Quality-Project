import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch


def test_transform_batch_empty_input():
    """Prüft, ob die Funktion bei leeren Daten korrekt mit None antwortet."""
    assert transform_batch([]) is None
    assert transform_batch(None) is None


def test_transform_batch_feature_calculation():
    """Integrationstest mit kleinem, vollständig definierten Mock-Datensatz."""
    mock_raw_data = [
        {"timestamp": "2026-06-01T00:00:00Z", "parameter": "pm25", "value": 10.0},
        {"timestamp": "2026-06-01T01:00:00Z", "parameter": "pm25", "value": 20.0},
        {"timestamp": "2026-06-01T02:00:00Z", "parameter": "pm25", "value": 30.0},
        {"timestamp": "2026-06-01T00:00:00Z", "parameter": "temperature", "value": 15.0},
        {"timestamp": "2026-06-01T01:00:00Z", "parameter": "temperature", "value": 16.0},
        {"timestamp": "2026-06-01T02:00:00Z", "parameter": "temperature", "value": 17.0},
        {"timestamp": "2026-06-01T00:00:00Z", "parameter": "relativehumidity", "value": 60.0},
        {"timestamp": "2026-06-01T01:00:00Z", "parameter": "relativehumidity", "value": 65.0},
        {"timestamp": "2026-06-01T02:00:00Z", "parameter": "relativehumidity", "value": 70.0},
    ]

    df_result = transform_batch(mock_raw_data, is_live_mode=True)

    assert df_result is not None
    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == 3
    assert df_result.loc[1, "pm25_lag_1h"] == 10.0