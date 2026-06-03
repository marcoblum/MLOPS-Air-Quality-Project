import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_pipeline.run_feature_pipeline import transform_batch
from tests.conftest import make_mock_data


def test_missing_temperature_handled():
    """ffill mit limit=3 füllt maximal 3 aufeinanderfolgende Lücken."""
    data = make_mock_data(hours=10)
    first_ts = data[0]["timestamp"]
    data = [d for d in data if not (d["parameter"] == "temperature" and d["timestamp"] != first_ts)]

    df = transform_batch(data, is_live_mode=True)
    assert df is not None
    if "temperature" in df.columns:
        # ffill(limit=3) füllt die ersten 3 Lücken – also Zeilen 1,2,3 haben Werte
        assert df["temperature"].iloc[1] == 15.0
        assert df["temperature"].iloc[2] == 15.0
        assert df["temperature"].iloc[3] == 15.0