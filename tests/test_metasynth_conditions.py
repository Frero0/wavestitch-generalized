import numpy as np
import pandas as pd
import pytest

from data_utils import Preprocessor
from dataset_config import DatasetConfig, load_dataset_config_by_id
from metasynth import SynthesisConditionError, metadataMask


def _custom_config(synthesis_conditions):
    return DatasetConfig.from_dict(
        {
            "dataset_id": "CustomDataset",
            "csv_path": "unused.csv",
            "loader": "flat_csv",
            "preprocessing_mode": "train_only",
            "timestamp_column": "timestamp",
            "signal_columns": ["signal"],
            "metadata_columns": ["day", "hour", "site"],
            "cyclic_columns": [],
            "temporal_order": ["timestamp"],
            "split": {"mode": "ratio", "train_ratio": 0.8},
            "synthesis_conditions": synthesis_conditions,
        }
    )


def _metro_metadata():
    train_count = 11
    test_count = 7949
    train_index = pd.Index(range(1, train_count * 3, 3))
    test_index = pd.Index(range(10001, 10001 + test_count * 2, 2))

    train = pd.DataFrame(
        {
            "year": [2017] * train_count,
            "month": [1] * train_count,
            "day": [15] * train_count,
            "hour": [6] * train_count,
        },
        index=train_index,
    )
    test = pd.DataFrame(
        {
            "year": [2018] * test_count,
            "month": [1] * test_count,
            "day": [15] * 286 + [1] * (test_count - 286),
            "hour": [6] * 349 + [0] * (test_count - 349),
        },
        index=test_index,
    )
    return pd.concat([train, test]), test_index.to_list()


@pytest.mark.parametrize(
    ("profile", "expected_count"),
    [("C", 7949), ("M", 286), ("F", 349)],
)
def test_metrotraffic_configured_masks_exactly_match_upstream(
    profile, expected_count
):
    metadata, test_indices = _metro_metadata()
    config = load_dataset_config_by_id("MetroTraffic")

    actual = metadataMask(
        metadata,
        profile,
        "MetroTraffic",
        dataset_config=config,
        test_indices=test_indices,
    )
    upstream = {
        "C": metadata["year"] == 2018,
        "M": (metadata["year"] == 2018) & (metadata["day"] == 15),
        "F": (metadata["year"] == 2018) & (metadata["hour"] == 6),
    }[profile]

    pd.testing.assert_series_equal(actual, upstream, check_names=False)
    assert int(actual.sum()) == expected_count
    assert not actual.loc[metadata["year"] != 2018].any()


def test_empty_profile_is_exactly_the_test_split_and_preserves_index():
    metadata = pd.DataFrame(
        {
            "day": [15, 1, 15, 2],
            "hour": [6, 7, 8, 9],
            "site": ["south", "north", "north", "south"],
        },
        index=pd.Index([101, 7, 400, 12]),
    )
    config = _custom_config({"C": {}})

    actual = metadataMask(
        metadata,
        "C",
        "CustomDataset",
        dataset_config=config,
        test_indices=[7, 12],
    )

    expected = pd.Series([False, True, False, True], index=metadata.index)
    pd.testing.assert_series_equal(actual, expected)
    assert actual.index is metadata.index


def test_numeric_and_string_conditions_are_conjoined_with_test_scope():
    metadata = pd.DataFrame(
        {
            "day": [15, 15, 1, 15],
            "hour": [6, 7, 6, 8],
            "site": ["north", "south", "north", "north"],
        },
        index=[10, 20, 30, 40],
    )
    config = _custom_config({"M": {"day": 15}, "F": {"site": "north"}})

    numeric = metadataMask(
        metadata,
        "M",
        "CustomDataset",
        dataset_config=config,
        test_indices=[20, 30, 40],
    )
    string = metadataMask(
        metadata,
        "F",
        "CustomDataset",
        dataset_config=config,
        test_indices=[20, 30, 40],
    )

    pd.testing.assert_series_equal(
        numeric, pd.Series([False, True, False, True], index=metadata.index)
    )
    pd.testing.assert_series_equal(
        string, pd.Series([False, False, True, True], index=metadata.index)
    )


def test_uci_occupancy_c_selects_exactly_2014_test_rows():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)
    context_indices = preprocessor.train_indices[-5:]
    candidate_indices = context_indices + preprocessor.test_indices
    decoded = preprocessor.cyclicDecode(
        preprocessor.df_cleaned.loc[candidate_indices]
    )
    metadata = decoded[preprocessor.hierarchical_features_uncyclic]

    actual = metadataMask(
        metadata,
        "C",
        "UCIOccupancyDetection",
        dataset_config=preprocessor.dataset_config,
        test_indices=preprocessor.test_indices,
    )

    assert len(preprocessor.train_indices) == 6129
    assert len(preprocessor.test_indices) == 2014
    assert int(actual.sum()) == 2014
    assert not actual.loc[context_indices].any()
    assert actual.loc[preprocessor.test_indices].all()


def test_missing_configured_profile_raises_explicit_error():
    config = _custom_config({"C": {}})
    metadata = pd.DataFrame(
        {"day": [1], "hour": [0], "site": ["north"]}, index=[5]
    )

    with pytest.raises(SynthesisConditionError, match="profile 'M'.*not configured"):
        metadataMask(
            metadata,
            "M",
            "CustomDataset",
            dataset_config=config,
            test_indices=[5],
        )


def test_missing_decoded_metadata_column_raises_explicit_error():
    config = _custom_config({"F": {"site": "north"}})
    metadata = pd.DataFrame({"day": [1], "hour": [0]}, index=[5])

    with pytest.raises(SynthesisConditionError, match="missing.*site"):
        metadataMask(
            metadata,
            "F",
            "CustomDataset",
            dataset_config=config,
            test_indices=[5],
        )


def _legacy_reference(metadata, profile, dataset):
    conditions = {
        "MetroTraffic": {
            "C": {"year": 2018},
            "M": {"year": 2018, "day": 15},
            "F": {"year": 2018, "hour": 6},
        },
        "AustraliaTourism": {
            "C": {"year": 2016},
            "M": {"year": 2016, "State": "Queensland"},
            "F": {"year": 2016, "Purpose": "Holiday"},
        },
        "BeijingAirQuality": {
            "C": {"year": 2017},
            "M": {"year": 2017, "month": 2},
            "F": {"year": 2017, "hour": 11},
        },
        "RossmanSales": {
            "C": {"Year": 2015},
            "M": {"Year": 2015, "Month": 3},
            "F": {"Year": 2015, "Store": 9},
        },
        "PanamaEnergy": {
            "C": {"year": 2020},
            "M": {"year": 2020, "day": 5},
            "F": {"year": 2020, "city": "san"},
        },
    }
    result = pd.Series(True, index=metadata.index)
    for column, value in conditions[dataset][profile].items():
        result &= metadata[column] == value
    return result


@pytest.mark.parametrize(
    "dataset",
    [
        "MetroTraffic",
        "AustraliaTourism",
        "BeijingAirQuality",
        "RossmanSales",
        "PanamaEnergy",
    ],
)
@pytest.mark.parametrize("profile", ["C", "M", "F"])
def test_legacy_fallback_is_unchanged(dataset, profile):
    metadata = pd.DataFrame(
        {
            "year": [2016, 2017, 2018, 2020, 2020],
            "month": [2, 2, 1, 1, 3],
            "day": [15, 5, 15, 5, 1],
            "hour": [11, 6, 6, 11, 0],
            "State": ["Queensland", "Other", "Other", "Other", "Other"],
            "Purpose": ["Holiday", "Business", "Holiday", "Business", "Holiday"],
            "Year": [2015, 2014, 2015, 2014, 2015],
            "Month": [3, 3, 1, 3, 2],
            "Store": [9, 8, 9, 7, 6],
            "city": ["san", "toc", "dav", "san", "toc"],
        },
        index=[20, 4, 99, 7, 301],
    )

    actual = metadataMask(metadata, profile, dataset)
    expected = _legacy_reference(metadata, profile, dataset)

    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_percentage_mask_keeps_upstream_random_semantics():
    metadata = pd.DataFrame({"value": range(8)}, index=[20, 4, 99, 7, 301, 8, 44, 2])
    config = _custom_config({"C": {}})
    seed = 73

    np.random.seed(seed)
    selected = np.random.choice(metadata.index, size=int(len(metadata) * 0.375), replace=False)
    expected = pd.Series(False, index=metadata.index)
    expected.loc[selected] = True

    np.random.seed(seed)
    actual = metadataMask(
        metadata,
        "0.375",
        "CustomDataset",
        dataset_config=config,
        test_indices=[7],
    )

    pd.testing.assert_series_equal(actual, expected)
