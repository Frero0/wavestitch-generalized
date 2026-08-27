import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from wavestitch import dataset_config
from wavestitch import data_utils
from wavestitch.data_utils import CyclicEncoder, Preprocessor


METRO_SIGNALS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "traffic_volume",
]
METRO_METADATA = ["year", "month", "day", "hour"]


def _write_metro_fixture(tmp_path):
    rows = []
    time_parts = [(1, 2, 3), (4, 15, 12), (10, 27, 21)]
    for year_offset, year in enumerate(range(2012, 2019)):
        for sample, (month, day, hour) in enumerate(time_parts):
            value = year_offset * len(time_parts) + sample
            rows.append(
                {
                    "holiday": "None",
                    "temp": 270.0 + value * 0.75,
                    "rain_1h": value * 0.1,
                    "snow_1h": value * 0.01,
                    "clouds_all": 10 + value * 3,
                    "weather_main": "Clouds",
                    "weather_description": "scattered clouds",
                    "date_time": "{:04d}-{:02d}-{:02d} {:02d}:00:00".format(
                        year, month, day, hour
                    ),
                    "traffic_volume": 1000 + value * 125,
                }
            )

    columns = [
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "date_time",
        "traffic_volume",
    ]
    csv_path = tmp_path / "metro.csv"
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    return csv_path


def _write_runtime_config(tmp_path, csv_path):
    source_config = (
        dataset_config.DEFAULT_DATASET_CONFIG_DIR / "MetroTraffic.json"
    )
    raw_config = json.loads(source_config.read_text(encoding="utf-8"))
    raw_config["csv_path"] = str(csv_path)

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "MetroTraffic.json").write_text(
        json.dumps(raw_config), encoding="utf-8"
    )
    return config_dir


def _legacy_metro_reference(csv_path, proportional_cyclic_encoding):
    """Reproduce the pre-DatasetConfig MetroTraffic branch."""

    original = pd.read_csv(csv_path)
    original["date_time"] = pd.to_datetime(original["date_time"])
    original["year"] = original["date_time"].dt.year
    original["month"] = original["date_time"].dt.month
    original["day"] = original["date_time"].dt.day
    original["hour"] = original["date_time"].dt.hour
    original.drop(
        columns=["date_time", "weather_main", "weather_description", "holiday"],
        inplace=True,
    )

    cleaned = original.copy()
    encoders = {}
    for column in METRO_METADATA:
        encoder = CyclicEncoder(column, cleaned, proportional_cyclic_encoding)
        encoders[column] = encoder
        cleaned = encoder.encode(cleaned)

    columns_to_scale = [
        column
        for column in cleaned.columns
        if column not in METRO_METADATA
        and "_sine" not in column
        and "_cos" not in column
    ]
    scaler = StandardScaler()
    cleaned[columns_to_scale] = scaler.fit_transform(cleaned[columns_to_scale])

    test_mask = original["year"].isin([2018])
    return SimpleNamespace(
        original=original,
        cleaned=cleaned,
        encoders=encoders,
        scaler=scaler,
        columns_to_scale=columns_to_scale,
        train_indices=original.index[~test_mask].to_list(),
        test_indices=original.index[test_mask].to_list(),
    )


@pytest.mark.parametrize("proportional_cyclic_encoding", [False, True])
def test_metrotraffic_config_runtime_matches_legacy(
    tmp_path, monkeypatch, proportional_cyclic_encoding
):
    csv_path = _write_metro_fixture(tmp_path)
    expected = _legacy_metro_reference(csv_path, proportional_cyclic_encoding)
    config_dir = _write_runtime_config(tmp_path, csv_path)

    monkeypatch.setattr(dataset_config, "DEFAULT_DATASET_CONFIG_DIR", config_dir)
    monkeypatch.setitem(
        data_utils.datasets, "MetroTraffic", "legacy/path/must/not/be/used.csv"
    )

    actual = Preprocessor("MetroTraffic", proportional_cyclic_encoding)

    assert actual.dataset_config is not None
    assert actual.dataset_config.dataset_id == "MetroTraffic"
    assert actual.preprocessing_mode == "upstream_legacy"
    assert actual.timestamp_column == "date_time"
    assert actual.signal_columns == METRO_SIGNALS
    assert actual.metadata_columns == METRO_METADATA
    assert actual.hierarchical_features_uncyclic == METRO_METADATA
    assert actual.cyclic_encoded_columns == METRO_METADATA
    assert actual.temporal_order == METRO_METADATA

    assert actual.df_orig.shape == expected.original.shape == (21, 9)
    assert actual.df_cleaned.shape == expected.cleaned.shape == (21, 13)
    assert len(actual.train_indices) == len(expected.train_indices) == 18
    assert len(actual.test_indices) == len(expected.test_indices) == 3
    assert actual.train_indices == expected.train_indices
    assert actual.test_indices == expected.test_indices

    assert actual.df_orig.columns.tolist() == expected.original.columns.tolist()
    assert actual.df_cleaned.columns.tolist() == expected.cleaned.columns.tolist()
    assert actual.df_orig.index.tolist() == expected.original.index.tolist()
    assert actual.df_cleaned.index.tolist() == expected.cleaned.index.tolist()
    pd.testing.assert_frame_equal(actual.df_orig, expected.original, check_exact=True)
    pd.testing.assert_frame_equal(
        actual.df_cleaned,
        expected.cleaned,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    assert set(actual.df_orig.loc[actual.train_indices, "year"]) == set(
        range(2012, 2018)
    )
    assert set(actual.df_orig.loc[actual.test_indices, "year"]) == {2018}
    pd.testing.assert_frame_equal(
        actual.df_orig.loc[actual.train_indices],
        expected.original.loc[expected.train_indices],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual.df_orig.loc[actual.test_indices],
        expected.original.loc[expected.test_indices],
        check_exact=True,
    )
    np.testing.assert_allclose(
        actual.df_cleaned.loc[actual.train_indices].to_numpy(),
        expected.cleaned.loc[expected.train_indices].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.df_cleaned.loc[actual.test_indices].to_numpy(),
        expected.cleaned.loc[expected.test_indices].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        actual.df_orig.columns.get_indexer(actual.signal_columns), np.arange(5)
    )
    np.testing.assert_array_equal(
        actual.df_orig.columns.get_indexer(actual.metadata_columns), np.arange(5, 9)
    )
    np.testing.assert_array_equal(
        actual.df_cleaned.columns.get_indexer(actual.signal_columns), np.arange(5)
    )
    np.testing.assert_array_equal(
        actual.df_cleaned.columns.get_indexer(actual.hierarchical_features_cyclic),
        np.arange(5, 13),
    )

    assert actual.cols_to_scale == expected.columns_to_scale == METRO_SIGNALS
    np.testing.assert_allclose(
        actual.df_cleaned[actual.cols_to_scale].to_numpy(),
        expected.cleaned[expected.columns_to_scale].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(actual.scaler.mean_, expected.scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose(actual.scaler.scale_, expected.scaler.scale_, rtol=1e-12)

    for column in METRO_METADATA:
        np.testing.assert_array_equal(
            actual.encoders[column].categories,
            expected.encoders[column].categories,
        )
        np.testing.assert_allclose(
            actual.encoders[column].angles,
            expected.encoders[column].angles,
            rtol=1e-12,
            atol=1e-12,
        )


def test_dataset_without_json_config_keeps_legacy_fallback():
    assert Preprocessor._load_dataset_config("AustraliaTourism") is None
