"""Central validation and planning for WaveStitch temporal windows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


class TemporalWindowValidationError(ValueError):
    """Raised when a split cannot safely produce WaveStitch windows."""


@dataclass(frozen=True)
class WindowingPlan:
    train_indices: Tuple[object, ...]
    test_indices: Tuple[object, ...]
    window_size: int
    training_stride: int
    training_window_count: int
    synthesis_stride: Optional[int] = None
    context_indices: Tuple[object, ...] = ()
    candidate_indices: Tuple[object, ...] = ()
    synthesis_window_count: Optional[int] = None

    @property
    def context_count(self) -> int:
        return len(self.context_indices)


def _positive_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TemporalWindowValidationError(
            "{} must be a positive integer; received {!r}.".format(
                field_name, value
            )
        )
    return value


def _ordered_split(preprocessor):
    frame = preprocessor.df_cleaned
    if not frame.index.is_unique:
        raise TemporalWindowValidationError(
            "Windowing requires a unique dataframe index."
        )

    train_indices = tuple(preprocessor.train_indices or ())
    test_indices = tuple(preprocessor.test_indices or ())
    if not train_indices:
        raise TemporalWindowValidationError("Train split must not be empty.")
    if not test_indices:
        raise TemporalWindowValidationError("Test split must not be empty.")

    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set & test_set
    if overlap:
        examples = list(overlap)[:5]
        raise TemporalWindowValidationError(
            "Train and test indices overlap; examples: {}.".format(examples)
        )

    frame_indices = tuple(frame.index)
    frame_set = set(frame_indices)
    unknown = (train_set | test_set) - frame_set
    if unknown:
        raise TemporalWindowValidationError(
            "Split contains index values absent from the dataframe: {}.".format(
                list(unknown)[:5]
            )
        )
    uncovered = frame_set - (train_set | test_set)
    if uncovered:
        raise TemporalWindowValidationError(
            "Train/test split does not cover every dataframe row; uncovered examples: {}.".format(
                list(uncovered)[:5]
            )
        )

    positions = {index: position for position, index in enumerate(frame_indices)}
    train_positions = [positions[index] for index in train_indices]
    test_positions = [positions[index] for index in test_indices]
    if train_positions != sorted(train_positions):
        raise TemporalWindowValidationError(
            "Train indices do not preserve dataframe temporal order."
        )
    if test_positions != sorted(test_positions):
        raise TemporalWindowValidationError(
            "Test indices do not preserve dataframe temporal order."
        )

    expected_train = list(range(len(train_indices)))
    expected_test = list(range(len(train_indices), len(frame_indices)))
    if train_positions != expected_train or test_positions != expected_test:
        if min(train_positions) > min(test_positions):
            raise TemporalWindowValidationError(
                "Train split must temporally precede the test split."
            )
        raise TemporalWindowValidationError(
            "Split is interlaced or non-contiguous; a training window would cross "
            "a structural gap or the train/test boundary."
        )

    config = getattr(preprocessor, "dataset_config", None)
    split_mode = getattr(getattr(config, "split", None), "mode", None)
    if split_mode in {"ratio", "timestamp"}:
        timestamps = getattr(preprocessor, "timestamps", None)
        if timestamps is not None:
            train_timestamps = timestamps.loc[list(train_indices)]
            test_timestamps = timestamps.loc[list(test_indices)]
            if not train_timestamps.is_monotonic_increasing:
                raise TemporalWindowValidationError(
                    "Train timestamps are not monotonic after the configured split."
                )
            if not test_timestamps.is_monotonic_increasing:
                raise TemporalWindowValidationError(
                    "Test timestamps are not monotonic after the configured split."
                )
            if train_timestamps.iloc[-1] > test_timestamps.iloc[0]:
                raise TemporalWindowValidationError(
                    "Train split does not temporally precede the test split."
                )

    if getattr(config, "loader", None) == "flat_csv":
        if tuple(frame.loc[list(train_indices)].index) != train_indices:
            raise TemporalWindowValidationError(
                "flat_csv train order changed after splitting."
            )
        if tuple(frame.loc[list(test_indices)].index) != test_indices:
            raise TemporalWindowValidationError(
                "flat_csv test order changed after splitting."
            )

    return train_indices, test_indices


def validate_training_windowing(
    preprocessor,
    *,
    window_size,
    requested_training_stride=1,
    effective_training_stride=1,
):
    """Validate the upstream training layout without changing its stride semantics."""

    window_size = _positive_integer(window_size, "window_size")
    requested_training_stride = _positive_integer(
        requested_training_stride, "requested training stride"
    )
    effective_training_stride = _positive_integer(
        effective_training_stride, "effective training stride"
    )
    if effective_training_stride != 1:
        raise TemporalWindowValidationError(
            "The generalized WaveStitch trainer currently supports only the upstream effective "
            "training stride 1."
        )
    if requested_training_stride != effective_training_stride:
        warnings.warn(
            "The generalized WaveStitch trainer records requested stride {} but preserves the "
            "upstream effective training stride 1.".format(requested_training_stride),
            UserWarning,
            stacklevel=2,
        )

    train_indices, test_indices = _ordered_split(preprocessor)
    if len(train_indices) < window_size:
        raise TemporalWindowValidationError(
            "Train split has {} rows, fewer than window_size {}.".format(
                len(train_indices), window_size
            )
        )

    training_window_count = (
        (len(train_indices) - window_size) // effective_training_stride
    ) + 1
    if training_window_count <= 0:
        raise TemporalWindowValidationError(
            "No valid train-only window can be constructed."
        )

    return WindowingPlan(
        train_indices=train_indices,
        test_indices=test_indices,
        window_size=window_size,
        training_stride=effective_training_stride,
        training_window_count=training_window_count,
    )


def validate_synthesis_windowing(preprocessor, *, window_size, synthesis_stride):
    """Plan the minimal train context and validate synthesis window coverage."""

    window_size = _positive_integer(window_size, "window_size")
    synthesis_stride = _positive_integer(synthesis_stride, "synthesis stride")
    if synthesis_stride > window_size:
        raise TemporalWindowValidationError(
            "synthesis stride {} exceeds window_size {}; gapped synthesis windows "
            "are not supported.".format(synthesis_stride, window_size)
        )

    train_indices, test_indices = _ordered_split(preprocessor)
    context_count = (window_size - len(test_indices)) % synthesis_stride
    if context_count > len(train_indices):
        raise TemporalWindowValidationError(
            "Synthesis requires {} training context rows but only {} are available.".format(
                context_count, len(train_indices)
            )
        )
    context_indices = (
        train_indices[-context_count:] if context_count else ()
    )
    candidate_indices = context_indices + test_indices
    if len(candidate_indices) < window_size:
        raise TemporalWindowValidationError(
            "Test split plus aligned training context has {} rows, fewer than "
            "window_size {}.".format(len(candidate_indices), window_size)
        )
    remainder = (len(candidate_indices) - window_size) % synthesis_stride
    if remainder != 0:
        raise TemporalWindowValidationError(
            "Test length, window_size, and synthesis stride cannot produce complete "
            "aligned windows."
        )
    synthesis_window_count = (
        (len(candidate_indices) - window_size) // synthesis_stride
    ) + 1
    if synthesis_window_count <= 0:
        raise TemporalWindowValidationError(
            "No valid synthesis window can be constructed."
        )

    return WindowingPlan(
        train_indices=train_indices,
        test_indices=test_indices,
        window_size=window_size,
        training_stride=1,
        training_window_count=max(len(train_indices) - window_size + 1, 0),
        synthesis_stride=synthesis_stride,
        context_indices=context_indices,
        candidate_indices=candidate_indices,
        synthesis_window_count=synthesis_window_count,
    )


def validate_synthesis_context_mask(mask, plan):
    """Ensure train context remains conditioning and is never synthesized."""

    if not isinstance(mask, pd.Series):
        raise TemporalWindowValidationError(
            "Synthesis mask must be a pandas Series aligned to candidate rows."
        )
    if tuple(mask.index) != plan.candidate_indices:
        raise TemporalWindowValidationError(
            "Synthesis mask index is not aligned with the planned context/test rows."
        )
    if plan.context_indices and mask.loc[list(plan.context_indices)].any():
        raise TemporalWindowValidationError(
            "Training context rows must never be part of the synthesis mask."
        )
    return True
