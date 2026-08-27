from types import SimpleNamespace

import pandas as pd
import pytest

from wavestitch.data_utils import Preprocessor
from wavestitch.dataset_config import load_dataset_config_by_id
from wavestitch.metasynth import metadataMask
from wavestitch.window_validation import (
    TemporalWindowValidationError,
    validate_synthesis_context_mask,
    validate_synthesis_windowing,
    validate_training_windowing,
)


def _preprocessor(
    *,
    row_count=12,
    train_indices=None,
    test_indices=None,
    split_mode="ratio",
    timestamps=None,
):
    if train_indices is None:
        train_indices = list(range(8))
    if test_indices is None:
        test_indices = list(range(8, row_count))
    if timestamps is None:
        timestamps = pd.Series(
            pd.date_range("2024-01-01", periods=row_count, freq="h", tz="UTC"),
            index=range(row_count),
        )
    return SimpleNamespace(
        df_cleaned=pd.DataFrame({"value": range(row_count)}),
        train_indices=train_indices,
        test_indices=test_indices,
        timestamps=timestamps,
        dataset_config=SimpleNamespace(
            loader="flat_csv",
            split=SimpleNamespace(mode=split_mode),
        ),
    )


def test_dataset_too_short_for_training_window():
    preprocessor = _preprocessor(
        row_count=6,
        train_indices=[0, 1, 2],
        test_indices=[3, 4, 5],
    )

    with pytest.raises(TemporalWindowValidationError, match="fewer than window_size"):
        validate_training_windowing(preprocessor, window_size=4)


def test_dataset_too_short_for_synthesis_window_with_context():
    preprocessor = _preprocessor(
        row_count=6,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
    )

    with pytest.raises(TemporalWindowValidationError, match="fewer than window_size"):
        validate_synthesis_windowing(
            preprocessor, window_size=8, synthesis_stride=4
        )


@pytest.mark.parametrize("window_size", [0, -1, 1.5, True])
def test_window_size_must_be_a_positive_integer(window_size):
    with pytest.raises(TemporalWindowValidationError, match="window_size.*positive integer"):
        validate_training_windowing(
            _preprocessor(), window_size=window_size
        )


@pytest.mark.parametrize("stride", [0, -2])
def test_training_stride_must_be_positive(stride):
    with pytest.raises(TemporalWindowValidationError, match="training stride.*positive"):
        validate_training_windowing(
            _preprocessor(),
            window_size=4,
            requested_training_stride=stride,
        )


@pytest.mark.parametrize("stride", [0, -2])
def test_synthesis_stride_must_be_positive(stride):
    with pytest.raises(TemporalWindowValidationError, match="synthesis stride.*positive"):
        validate_synthesis_windowing(
            _preprocessor(), window_size=4, synthesis_stride=stride
        )


def test_synthesis_stride_must_not_exceed_window_size():
    with pytest.raises(TemporalWindowValidationError, match="exceeds window_size"):
        validate_synthesis_windowing(
            _preprocessor(), window_size=4, synthesis_stride=5
        )


@pytest.mark.parametrize(
    ("train_indices", "test_indices", "message"),
    [
        ([], list(range(12)), "Train split must not be empty"),
        (list(range(12)), [], "Test split must not be empty"),
    ],
)
def test_train_and_test_must_not_be_empty(train_indices, test_indices, message):
    preprocessor = _preprocessor(
        train_indices=train_indices,
        test_indices=test_indices,
    )

    with pytest.raises(TemporalWindowValidationError, match=message):
        validate_training_windowing(preprocessor, window_size=4)


def test_train_and_test_must_not_overlap():
    preprocessor = _preprocessor(
        row_count=6,
        train_indices=[0, 1, 2, 3],
        test_indices=[3, 4, 5],
    )

    with pytest.raises(TemporalWindowValidationError, match="indices overlap"):
        validate_training_windowing(preprocessor, window_size=2)


def test_inverted_split_is_rejected():
    preprocessor = _preprocessor(
        row_count=6,
        train_indices=[3, 4, 5],
        test_indices=[0, 1, 2],
    )

    with pytest.raises(TemporalWindowValidationError, match="Train split must temporally precede"):
        validate_training_windowing(preprocessor, window_size=2)


def test_temporally_inverted_ratio_boundary_is_rejected():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-01-02T00:00:00Z",
                "2024-01-02T01:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
            ]
        )
    )
    preprocessor = _preprocessor(
        row_count=4,
        train_indices=[0, 1],
        test_indices=[2, 3],
        timestamps=timestamps,
    )

    with pytest.raises(TemporalWindowValidationError, match="Train timestamps are not monotonic|does not temporally precede"):
        validate_training_windowing(preprocessor, window_size=2)


def test_interlaced_split_is_rejected():
    preprocessor = _preprocessor(
        row_count=6,
        train_indices=[0, 2, 4],
        test_indices=[1, 3, 5],
    )

    with pytest.raises(TemporalWindowValidationError, match="interlaced or non-contiguous"):
        validate_synthesis_windowing(
            preprocessor, window_size=2, synthesis_stride=1
        )


def test_training_window_that_would_cross_boundary_is_rejected():
    preprocessor = _preprocessor(
        row_count=7,
        train_indices=[0, 1, 3, 4],
        test_indices=[2, 5, 6],
    )

    with pytest.raises(TemporalWindowValidationError, match="window would cross.*boundary"):
        validate_training_windowing(preprocessor, window_size=2)


def test_training_uses_effective_upstream_stride_one():
    with pytest.warns(UserWarning, match="effective training stride 1"):
        plan = validate_training_windowing(
            _preprocessor(),
            window_size=4,
            requested_training_stride=3,
            effective_training_stride=1,
        )

    assert plan.training_stride == 1
    assert plan.training_window_count == 5


def test_synthesis_context_is_minimal_and_windows_end_at_test_boundary():
    preprocessor = _preprocessor(
        row_count=19,
        train_indices=list(range(10)),
        test_indices=list(range(10, 19)),
    )

    plan = validate_synthesis_windowing(
        preprocessor, window_size=8, synthesis_stride=4
    )

    assert plan.context_indices == (7, 8, 9)
    assert plan.context_count == 3
    assert plan.candidate_indices == tuple(range(7, 19))
    assert plan.synthesis_window_count == 2
    assert (
        len(plan.candidate_indices) - plan.window_size
    ) % plan.synthesis_stride == 0


def test_context_rows_are_excluded_from_synthesis_mask():
    preprocessor = _preprocessor(
        row_count=19,
        train_indices=list(range(10)),
        test_indices=list(range(10, 19)),
    )
    plan = validate_synthesis_windowing(
        preprocessor, window_size=8, synthesis_stride=4
    )
    valid_mask = pd.Series(
        [False] * plan.context_count + [True] * len(plan.test_indices),
        index=plan.candidate_indices,
    )

    assert validate_synthesis_context_mask(valid_mask, plan)
    invalid_mask = valid_mask.copy()
    invalid_mask.loc[plan.context_indices[0]] = True
    with pytest.raises(TemporalWindowValidationError, match="must never be part"):
        validate_synthesis_context_mask(invalid_mask, plan)


def test_metrotraffic_windowing_matches_upstream_counts():
    row_count = 48204
    test_start = 40255
    preprocessor = SimpleNamespace(
        df_cleaned=pd.DataFrame(index=range(row_count)),
        train_indices=list(range(test_start)),
        test_indices=list(range(test_start, row_count)),
        timestamps=None,
        dataset_config=load_dataset_config_by_id("MetroTraffic"),
    )

    training_plan = validate_training_windowing(
        preprocessor, window_size=32
    )
    synthesis_plan = validate_synthesis_windowing(
        preprocessor, window_size=32, synthesis_stride=8
    )

    assert training_plan.training_window_count == 40224
    assert synthesis_plan.context_count == 3
    assert synthesis_plan.context_indices == (40252, 40253, 40254)
    assert len(synthesis_plan.test_indices) == 7949
    assert synthesis_plan.synthesis_window_count == 991


def test_occupancy_windowing_and_context_mask_regression():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)
    training_plan = validate_training_windowing(
        preprocessor, window_size=8
    )
    synthesis_plan = validate_synthesis_windowing(
        preprocessor, window_size=8, synthesis_stride=4
    )
    candidate = preprocessor.df_cleaned.loc[
        list(synthesis_plan.candidate_indices)
    ]
    decoded = preprocessor.cyclicDecode(candidate)
    metadata = decoded[preprocessor.hierarchical_features_uncyclic]
    mask = metadataMask(
        metadata,
        "C",
        "UCIOccupancyDetection",
        dataset_config=preprocessor.dataset_config,
        test_indices=preprocessor.test_indices,
    )

    assert training_plan.training_window_count == 6122
    assert synthesis_plan.context_count == 2
    assert synthesis_plan.context_indices == (6128, 6129)
    assert len(synthesis_plan.test_indices) == 2014
    assert synthesis_plan.synthesis_window_count == 503
    assert int(mask.sum()) == 2014
    assert validate_synthesis_context_mask(mask, synthesis_plan)


@pytest.mark.parametrize("split_mode", ["ratio", "timestamp"])
def test_valid_chronological_ratio_and_timestamp_splits(split_mode):
    preprocessor = _preprocessor(split_mode=split_mode)

    training_plan = validate_training_windowing(
        preprocessor, window_size=4
    )
    synthesis_plan = validate_synthesis_windowing(
        preprocessor, window_size=4, synthesis_stride=2
    )

    assert training_plan.training_window_count == 5
    assert synthesis_plan.context_count == 0
    assert synthesis_plan.synthesis_window_count == 1
