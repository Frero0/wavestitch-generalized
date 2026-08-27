import json

import pytest

from wavestitch.dataset_config import (
    DatasetConfigError,
    DatasetConfigNotFoundError,
    DatasetSourceNotFoundError,
    SplitConfig,
    load_dataset_config,
    load_dataset_config_by_id,
)


def _valid_config():
    return {
        "dataset_id": "ExampleSeries",
        "csv_path": "data/example.csv",
        "loader": "flat_csv",
        "preprocessing_mode": "train_only",
        "timestamp_column": "timestamp",
        "signal_columns": ["load", "temperature"],
        "metadata_columns": ["year", "site"],
        "cyclic_columns": ["year"],
        "dtype_overrides": {"site": "object"},
        "temporal_order": ["timestamp", "site"],
        "split": {
            "mode": "column_values",
            "column": "year",
            "test_values": [2024],
        },
    }


def _write_config(tmp_path, contents, filename="ExampleSeries.json"):
    config_path = tmp_path / filename
    config_path.write_text(json.dumps(contents), encoding="utf-8")
    return config_path


def test_load_valid_configuration(tmp_path):
    config = load_dataset_config(_write_config(tmp_path, _valid_config()))

    assert config.dataset_id == "ExampleSeries"
    assert config.loader == "flat_csv"
    assert config.preprocessing_mode == "train_only"
    assert config.timestamp_column == "timestamp"
    assert config.dtype_overrides == {"site": "object"}


def test_load_metrotraffic_configuration():
    config = load_dataset_config_by_id("MetroTraffic")

    assert config.loader == "legacy"
    assert config.preprocessing_mode == "upstream_legacy"
    assert config.csv_path.endswith("Metro_Interstate_Traffic_Volume.csv")
    assert config.metadata_columns == ("year", "month", "day", "hour")
    assert config.split.column == "year"
    assert config.split.test_values == (2018,)
    assert config.synthesis_conditions == {
        "C": {},
        "M": {"day": 15},
        "F": {"hour": 6},
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(loader="spreadsheet"), "Unsupported loader"),
        (
            lambda data: data.update(cyclic_columns=["not_metadata"]),
            "Cyclic columns must also be metadata columns",
        ),
        (
            lambda data: data.update(signal_columns=["load", "load"]),
            "must not contain duplicate columns",
        ),
    ],
)
def test_invalid_configuration_has_clear_error(tmp_path, mutation, message):
    raw_config = _valid_config()
    mutation(raw_config)

    with pytest.raises(DatasetConfigError, match=message):
        load_dataset_config(_write_config(tmp_path, raw_config))


def test_nonexistent_configuration_path_has_clear_error(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(DatasetConfigNotFoundError, match="does not exist"):
        load_dataset_config(missing_path)


def test_nonexistent_csv_path_can_be_checked_explicitly(tmp_path):
    config = load_dataset_config(_write_config(tmp_path, _valid_config()))

    with pytest.raises(DatasetSourceNotFoundError, match="Dataset CSV does not exist"):
        config.resolve_csv_path(tmp_path, must_exist=True)


def test_missing_required_fields_are_reported_together(tmp_path):
    raw_config = _valid_config()
    del raw_config["signal_columns"]
    del raw_config["split"]

    with pytest.raises(DatasetConfigError) as error:
        load_dataset_config(_write_config(tmp_path, raw_config))

    assert "signal_columns" in str(error.value)
    assert "split" in str(error.value)


def test_split_is_parsed_to_typed_configuration(tmp_path):
    raw_config = _valid_config()
    raw_config["split"] = {
        "mode": "column_values",
        "column": "site",
        "test_values": ["north", "south"],
    }

    config = load_dataset_config(_write_config(tmp_path, raw_config))

    assert config.split.mode == "column_values"
    assert config.split.column == "site"
    assert config.split.test_values == ("north", "south")


def test_column_order_is_preserved(tmp_path):
    raw_config = _valid_config()
    raw_config["signal_columns"] = ["temperature", "load"]
    raw_config["metadata_columns"] = ["site", "year"]
    raw_config["cyclic_columns"] = ["year"]
    raw_config["temporal_order"] = ["site", "timestamp", "year"]

    config = load_dataset_config(_write_config(tmp_path, raw_config))

    assert config.signal_columns == ("temperature", "load")
    assert config.metadata_columns == ("site", "year")
    assert config.cyclic_columns == ("year",)
    assert config.temporal_order == ("site", "timestamp", "year")


def test_ratio_split_configuration_is_parsed():
    split = SplitConfig.from_dict({"mode": "ratio", "train_ratio": 0.8})

    assert split.mode == "ratio"
    assert split.train_ratio == 0.8
    assert split.column is None
    assert split.test_values == ()
    assert split.cutoff is None


def test_timestamp_split_configuration_is_parsed():
    split = SplitConfig.from_dict(
        {"mode": "timestamp", "cutoff": "2024-01-01T00:00:00Z"}
    )

    assert split.mode == "timestamp"
    assert split.cutoff == "2024-01-01T00:00:00Z"
    assert split.column is None
    assert split.test_values == ()
    assert split.train_ratio is None


@pytest.mark.parametrize(
    "train_ratio",
    [-1, 0, 1, 1.1, float("nan"), float("inf"), "0.8", True],
)
def test_ratio_split_rejects_invalid_values(train_ratio):
    with pytest.raises(DatasetConfigError, match="strictly between 0 and 1"):
        SplitConfig.from_dict({"mode": "ratio", "train_ratio": train_ratio})


def test_timestamp_split_rejects_empty_cutoff():
    with pytest.raises(DatasetConfigError, match="split.cutoff"):
        SplitConfig.from_dict({"mode": "timestamp", "cutoff": ""})


def test_occupancy_configuration_uses_train_only_timestamp_split():
    config = load_dataset_config_by_id("UCIOccupancyDetection")

    assert config.preprocessing_mode == "train_only"
    assert config.split.mode == "timestamp"
    assert config.split.cutoff == "2015-02-09T00:00:00Z"
    assert config.signal_columns == ("Temperature", "Humidity", "Light", "CO2")
    assert config.metadata_columns == ("Occupancy",)
    assert config.cyclic_columns == ()
    assert config.synthesis_conditions == {"C": {}}


@pytest.mark.parametrize("mode", ["global", "legacy", ""])
def test_preprocessing_mode_rejects_unsupported_values(tmp_path, mode):
    raw_config = _valid_config()
    raw_config["preprocessing_mode"] = mode

    with pytest.raises(DatasetConfigError, match="preprocessing_mode"):
        load_dataset_config(_write_config(tmp_path, raw_config))


def test_synthesis_conditions_are_optional(tmp_path):
    config = load_dataset_config(_write_config(tmp_path, _valid_config()))

    assert config.synthesis_conditions == {}


def test_synthesis_conditions_accept_empty_numeric_and_string_profiles(tmp_path):
    raw_config = _valid_config()
    raw_config["synthesis_conditions"] = {
        "C": {},
        "M": {"year": 2024},
        "F": {"site": "north"},
    }

    config = load_dataset_config(_write_config(tmp_path, raw_config))

    assert config.synthesis_conditions == raw_config["synthesis_conditions"]


def test_synthesis_conditions_reject_unknown_profile(tmp_path):
    raw_config = _valid_config()
    raw_config["synthesis_conditions"] = {"X": {}}

    with pytest.raises(DatasetConfigError, match="Unsupported synthesis profile"):
        load_dataset_config(_write_config(tmp_path, raw_config))


def test_synthesis_conditions_reject_non_object_profile(tmp_path):
    raw_config = _valid_config()
    raw_config["synthesis_conditions"] = {"C": ["year", 2024]}

    with pytest.raises(DatasetConfigError, match="must be a JSON object"):
        load_dataset_config(_write_config(tmp_path, raw_config))


def test_synthesis_conditions_reject_non_metadata_column(tmp_path):
    raw_config = _valid_config()
    raw_config["synthesis_conditions"] = {"C": {"timestamp": "2024-01-01"}}

    with pytest.raises(DatasetConfigError, match="not declared in metadata_columns"):
        load_dataset_config(_write_config(tmp_path, raw_config))


@pytest.mark.parametrize("invalid_value", [None, [2024], {"value": 2024}])
def test_synthesis_conditions_reject_null_or_non_scalar_values(
    tmp_path, invalid_value
):
    raw_config = _valid_config()
    raw_config["synthesis_conditions"] = {"C": {"year": invalid_value}}

    with pytest.raises(DatasetConfigError, match="non-null finite JSON scalar"):
        load_dataset_config(_write_config(tmp_path, raw_config))
