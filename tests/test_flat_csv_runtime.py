import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from wavestitch import dataset_config
from wavestitch.data_utils import FlatCSVValidationError, Preprocessor


FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "SyntheticFlatCSV.json"


def _flat_rows():
    return [
        {
            "timestamp": "2023-01-01 00:00:00",
            "load": 10.0,
            "temperature": 5.0,
            "year": 2023,
            "site": "A",
            "ignored": "x",
        },
        {
            "timestamp": "2023-01-01T00:00:00Z",
            "load": 12.0,
            "temperature": 6.0,
            "year": 2023,
            "site": "B",
            "ignored": "x",
        },
        {
            "timestamp": "2023-01-01 01:00:00+00:00",
            "load": 14.0,
            "temperature": 7.0,
            "year": 2023,
            "site": "A",
            "ignored": "x",
        },
        {
            "timestamp": "2023-01-01T01:00:00Z",
            "load": 16.0,
            "temperature": 8.0,
            "year": 2023,
            "site": "B",
            "ignored": "x",
        },
        {
            "timestamp": "2024-01-01 00:00:00",
            "load": 18.0,
            "temperature": 9.0,
            "year": 2024,
            "site": "A",
            "ignored": "x",
        },
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "load": 20.0,
            "temperature": 10.0,
            "year": 2024,
            "site": "B",
            "ignored": "x",
        },
    ]


def _write_flat_runtime(tmp_path, monkeypatch, rows=None):
    csv_path = tmp_path / "synthetic_flat.csv"
    pd.DataFrame(_flat_rows() if rows is None else rows).to_csv(csv_path, index=False)

    raw_config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    raw_config["csv_path"] = str(csv_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "SyntheticFlatCSV.json").write_text(
        json.dumps(raw_config), encoding="utf-8"
    )
    monkeypatch.setattr(dataset_config, "DEFAULT_DATASET_CONFIG_DIR", config_dir)
    return csv_path


def test_flat_csv_end_to_end(tmp_path, monkeypatch):
    _write_flat_runtime(tmp_path, monkeypatch)

    preprocessor = Preprocessor("SyntheticFlatCSV", False)

    assert preprocessor.dataset_config.loader == "flat_csv"
    assert preprocessor.df_orig.shape == (6, 4)
    assert preprocessor.df_cleaned.shape == (6, 6)
    assert preprocessor.df_orig.columns.tolist() == [
        "load",
        "temperature",
        "year",
        "site",
    ]
    assert preprocessor.df_cleaned.columns.tolist() == [
        "load",
        "temperature",
        "year_sine",
        "year_cos",
        "site_sine",
        "site_cos",
    ]
    assert preprocessor.df_orig.index.tolist() == list(range(6))
    assert preprocessor.timestamps.index.tolist() == list(range(6))
    assert str(preprocessor.timestamps.dtype) == "datetime64[ns, UTC]"
    expected_timestamps = pd.to_datetime(
        [row["timestamp"] for row in _flat_rows()], format="mixed", utc=True
    )
    pd.testing.assert_series_equal(
        preprocessor.timestamps,
        pd.Series(expected_timestamps, name="timestamp"),
    )

    assert str(preprocessor.df_orig["load"].dtype) == "float64"
    assert str(preprocessor.df_orig["temperature"].dtype) == "float64"
    assert str(preprocessor.df_orig["year"].dtype) == "int64"
    assert str(preprocessor.df_orig["site"].dtype) == "string"
    assert preprocessor.train_indices == [0, 1, 2, 3]
    assert preprocessor.test_indices == [4, 5]
    assert preprocessor.hierarchical_features_uncyclic == ["year", "site"]
    assert preprocessor.hierarchical_features_cyclic == [
        "year_sine",
        "year_cos",
        "site_sine",
        "site_cos",
    ]

    expected_scaled = StandardScaler().fit_transform(
        preprocessor.df_orig[["load", "temperature"]]
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned[["load", "temperature"]].to_numpy(),
        expected_scaled,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned.loc[:3, "year_sine"], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned.loc[:3, "year_cos"], 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned.loc[4:, "year_sine"], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned.loc[4:, "year_cos"], -1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned["site_sine"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        preprocessor.df_cleaned["site_cos"],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        atol=1e-12,
    )

    rescaled = preprocessor.rescale(preprocessor.df_cleaned)
    np.testing.assert_allclose(
        rescaled[["load", "temperature"]],
        preprocessor.df_orig[["load", "temperature"]],
        rtol=1e-12,
        atol=1e-12,
    )
    decoded = preprocessor.decode(preprocessor.df_cleaned, rescale=True)
    assert decoded.columns.tolist() == preprocessor.df_orig.columns.tolist()
    assert decoded.index.tolist() == preprocessor.df_orig.index.tolist()
    np.testing.assert_allclose(
        decoded[["load", "temperature"]],
        preprocessor.df_orig[["load", "temperature"]],
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(decoded["year"], preprocessor.df_orig["year"])
    pd.testing.assert_series_equal(decoded["site"], preprocessor.df_orig["site"])


@pytest.mark.parametrize("missing_column", ["temperature", "site", "timestamp"])
def test_flat_csv_rejects_missing_required_columns(
    tmp_path, monkeypatch, missing_column
):
    rows = _flat_rows()
    for row in rows:
        del row[missing_column]
    _write_flat_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(FlatCSVValidationError, match=missing_column):
        Preprocessor("SyntheticFlatCSV", False)


def test_flat_csv_rejects_invalid_timestamps(tmp_path, monkeypatch):
    rows = _flat_rows()
    rows[2]["timestamp"] = "not-a-timestamp"
    _write_flat_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(FlatCSVValidationError, match="Invalid timestamp"):
        Preprocessor("SyntheticFlatCSV", False)


def test_flat_csv_rejects_non_numeric_signals(tmp_path, monkeypatch):
    rows = _flat_rows()
    rows[1]["load"] = "not-numeric"
    _write_flat_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(FlatCSVValidationError, match="signal column.*load"):
        Preprocessor("SyntheticFlatCSV", False)


def test_flat_csv_rejects_duplicate_temporal_keys(tmp_path, monkeypatch):
    rows = _flat_rows()
    rows.insert(1, dict(rows[0]))
    _write_flat_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(FlatCSVValidationError, match="Duplicate temporal key"):
        Preprocessor("SyntheticFlatCSV", False)


def test_flat_csv_rejects_rows_out_of_temporal_order(tmp_path, monkeypatch):
    rows = _flat_rows()
    rows[0], rows[2] = rows[2], rows[0]
    _write_flat_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(FlatCSVValidationError, match="not ordered by temporal_order"):
        Preprocessor("SyntheticFlatCSV", False)
