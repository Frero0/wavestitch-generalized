import json

import pandas as pd
import pytest

import dataset_config
from data_utils import DatasetSplitError, Preprocessor


def _rows(count=10):
    timestamps = pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC")
    return [
        {
            "timestamp": timestamp.isoformat(),
            "signal_a": float(index),
            "signal_b": float(index * 10),
            "phase": 0 if index < 8 else 1,
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _preprocessor(tmp_path, monkeypatch, split, rows=None):
    csv_path = tmp_path / "split_dataset.csv"
    pd.DataFrame(_rows() if rows is None else rows).to_csv(csv_path, index=False)
    config = {
        "dataset_id": "SplitDataset",
        "csv_path": str(csv_path),
        "loader": "flat_csv",
        "preprocessing_mode": "upstream_legacy",
        "timestamp_column": "timestamp",
        "signal_columns": ["signal_a", "signal_b"],
        "metadata_columns": ["phase"],
        "cyclic_columns": ["phase"],
        "dtype_overrides": {},
        "temporal_order": ["timestamp"],
        "split": split,
    }
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "SplitDataset.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    monkeypatch.setattr(dataset_config, "DEFAULT_DATASET_CONFIG_DIR", config_dir)
    return Preprocessor("SplitDataset", False)


def test_ratio_split_80_20_preserves_temporal_order(tmp_path, monkeypatch):
    preprocessor = _preprocessor(
        tmp_path,
        monkeypatch,
        {"mode": "ratio", "train_ratio": 0.8},
    )

    assert preprocessor.train_indices == list(range(8))
    assert preprocessor.test_indices == [8, 9]
    assert preprocessor.df_orig.loc[
        preprocessor.train_indices, "signal_a"
    ].tolist() == list(map(float, range(8)))
    assert preprocessor.df_orig.loc[
        preprocessor.test_indices, "signal_a"
    ].tolist() == [8.0, 9.0]
    assert preprocessor.timestamps.loc[preprocessor.train_indices].is_monotonic_increasing
    assert preprocessor.timestamps.loc[preprocessor.test_indices].is_monotonic_increasing
    assert (
        preprocessor.timestamps.loc[preprocessor.train_indices[-1]]
        < preprocessor.timestamps.loc[preprocessor.test_indices[0]]
    )


def test_timestamp_cutoff_puts_boundary_in_test(tmp_path, monkeypatch):
    cutoff = "2024-01-01T03:00:00Z"
    preprocessor = _preprocessor(
        tmp_path,
        monkeypatch,
        {"mode": "timestamp", "cutoff": cutoff},
    )

    assert preprocessor.train_indices == [0, 1, 2]
    assert preprocessor.test_indices == list(range(3, 10))
    cutoff_value = pd.Timestamp(cutoff)
    assert (preprocessor.timestamps.loc[preprocessor.train_indices] < cutoff_value).all()
    assert (preprocessor.timestamps.loc[preprocessor.test_indices] >= cutoff_value).all()
    assert preprocessor.timestamps.loc[3] == cutoff_value


def test_timestamp_split_rejects_invalid_cutoff(tmp_path, monkeypatch):
    with pytest.raises(DatasetSplitError, match="Invalid timestamp split cutoff"):
        _preprocessor(
            tmp_path,
            monkeypatch,
            {"mode": "timestamp", "cutoff": "not-a-timestamp"},
        )


@pytest.mark.parametrize(
    ("cutoff", "empty_set"),
    [
        ("2024-01-01T00:00:00Z", "train"),
        ("2025-01-01T00:00:00Z", "test"),
    ],
)
def test_timestamp_split_rejects_empty_sets(
    tmp_path, monkeypatch, cutoff, empty_set
):
    with pytest.raises(DatasetSplitError, match="empty {} set".format(empty_set)):
        _preprocessor(
            tmp_path,
            monkeypatch,
            {"mode": "timestamp", "cutoff": cutoff},
        )


def test_column_values_runtime_regression(tmp_path, monkeypatch):
    preprocessor = _preprocessor(
        tmp_path,
        monkeypatch,
        {"mode": "column_values", "column": "phase", "test_values": [1]},
    )

    assert preprocessor.train_indices == list(range(8))
    assert preprocessor.test_indices == [8, 9]
