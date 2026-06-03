import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch
from tests.conftest import make_mock_data


def test_transform_batch_empty_input():
    """Leere oder None-Eingabe soll None zurückgeben."""
    assert transform_batch([]) is None
    assert transform_batch(None) is None


def test_transform_batch_returns_dataframe():
    """Gültige Eingabe soll einen DataFrame zurückgeben."""
    df = transform_batch(make_mock_data(hours=25), is_live_mode=True)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0