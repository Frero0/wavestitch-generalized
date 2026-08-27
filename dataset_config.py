"""Typed dataset configuration loading and validation.

``data_utils.Preprocessor`` consumes this abstraction for the legacy-compatible
MetroTraffic path and for custom ``flat_csv`` datasets. Datasets without a
configuration remain on their upstream branches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


JsonScalar = Union[str, int, float, bool]
SUPPORTED_LOADERS = frozenset({"legacy", "flat_csv"})
SUPPORTED_PREPROCESSING_MODES = frozenset({"train_only", "upstream_legacy"})
SUPPORTED_SPLIT_MODES = frozenset({"column_values", "ratio", "timestamp"})
SUPPORTED_SYNTHESIS_PROFILES = frozenset({"C", "M", "F"})
DEFAULT_DATASET_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "datasets"


class DatasetConfigError(ValueError):
    """Raised when a dataset configuration is structurally invalid."""


class DatasetConfigNotFoundError(FileNotFoundError):
    """Raised when the requested JSON configuration does not exist."""


class DatasetSourceNotFoundError(FileNotFoundError):
    """Raised when a configured dataset source does not exist."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(
            "Field {!r} must be a non-empty string.".format(field_name)
        )
    return value


def _string_tuple(
    value: Any, field_name: str, *, allow_empty: bool = False
) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetConfigError(
            "Field {!r} must be a JSON array of non-empty strings.".format(field_name)
        )

    result = tuple(
        _require_non_empty_string(item, "{}[{}]".format(field_name, index))
        for index, item in enumerate(value)
    )
    if not result and not allow_empty:
        raise DatasetConfigError("Field {!r} must not be empty.".format(field_name))
    if len(set(result)) != len(result):
        raise DatasetConfigError(
            "Field {!r} must not contain duplicate columns.".format(field_name)
        )
    return result


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set, context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise DatasetConfigError(
            "Unknown field(s) in {}: {}.".format(context, ", ".join(unknown))
        )


def _require_fields(data: Mapping[str, Any], required: set, context: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise DatasetConfigError(
            "Missing required field(s) in {}: {}.".format(context, ", ".join(missing))
        )


@dataclass(frozen=True)
class SplitConfig:
    """Validated parameters for one supported train/test split strategy."""

    mode: str
    column: Optional[str] = None
    test_values: Tuple[JsonScalar, ...] = ()
    train_ratio: Optional[float] = None
    cutoff: Optional[str] = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.mode, "split.mode")
        if self.mode not in SUPPORTED_SPLIT_MODES:
            raise DatasetConfigError(
                "Unsupported split mode {!r}; expected one of: {}.".format(
                    self.mode, ", ".join(sorted(SUPPORTED_SPLIT_MODES))
                )
            )
        if self.mode == "column_values":
            _require_non_empty_string(self.column, "split.column")
            if isinstance(self.test_values, (str, bytes)) or not isinstance(
                self.test_values, Sequence
            ):
                raise DatasetConfigError(
                    "Field 'split.test_values' must be a non-empty JSON array of scalar values."
                )
            values = tuple(self.test_values)
            if not values:
                raise DatasetConfigError("Field 'split.test_values' must not be empty.")
            for index, value in enumerate(values):
                if value is None or isinstance(value, (dict, list, tuple)):
                    raise DatasetConfigError(
                        "Field 'split.test_values[{}]' must be a non-null JSON scalar.".format(
                            index
                        )
                    )
            object.__setattr__(self, "test_values", values)
            if self.train_ratio is not None or self.cutoff is not None:
                raise DatasetConfigError(
                    "column_values split does not accept train_ratio or cutoff."
                )

        elif self.mode == "ratio":
            if (
                isinstance(self.train_ratio, bool)
                or not isinstance(self.train_ratio, (int, float))
                or not math.isfinite(self.train_ratio)
                or not 0 < self.train_ratio < 1
            ):
                raise DatasetConfigError(
                    "Field 'split.train_ratio' must be a finite number strictly between 0 and 1."
                )
            object.__setattr__(self, "train_ratio", float(self.train_ratio))
            if self.column is not None or self.test_values or self.cutoff is not None:
                raise DatasetConfigError(
                    "ratio split accepts only mode and train_ratio."
                )

        elif self.mode == "timestamp":
            _require_non_empty_string(self.cutoff, "split.cutoff")
            if self.column is not None or self.test_values or self.train_ratio is not None:
                raise DatasetConfigError(
                    "timestamp split accepts only mode and cutoff."
                )

    @classmethod
    def from_dict(cls, data: Any) -> "SplitConfig":
        if not isinstance(data, Mapping):
            raise DatasetConfigError("Field 'split' must be a JSON object.")
        _require_fields(data, {"mode"}, "split")
        mode = _require_non_empty_string(data["mode"], "split.mode")
        schemas = {
            "column_values": {"mode", "column", "test_values"},
            "ratio": {"mode", "train_ratio"},
            "timestamp": {"mode", "cutoff"},
        }
        if mode not in schemas:
            return cls(mode=mode)
        required = schemas[mode]
        _require_fields(data, required, "split")
        _reject_unknown_fields(data, required, "split")
        return cls(
            mode=mode,
            column=data.get("column"),
            test_values=data.get("test_values", ()),
            train_ratio=data.get("train_ratio"),
            cutoff=data.get("cutoff"),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """Validated description of a WaveStitch dataset.

    Column collections are stored as tuples so their declared order is stable.
    Relative CSV paths are interpreted from the project root by
    :meth:`resolve_csv_path`.
    """

    dataset_id: str
    csv_path: str
    loader: str
    preprocessing_mode: str
    signal_columns: Tuple[str, ...]
    metadata_columns: Tuple[str, ...]
    cyclic_columns: Tuple[str, ...]
    temporal_order: Tuple[str, ...]
    split: SplitConfig
    timestamp_column: Optional[str] = None
    dtype_overrides: Mapping[str, str] = field(default_factory=dict)
    synthesis_conditions: Mapping[str, Mapping[str, JsonScalar]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_non_empty_string(self.dataset_id, "dataset_id")
        csv_path = _require_non_empty_string(self.csv_path, "csv_path")
        if Path(csv_path).suffix.lower() != ".csv":
            raise DatasetConfigError("Field 'csv_path' must point to a .csv file.")

        _require_non_empty_string(self.loader, "loader")
        if self.loader not in SUPPORTED_LOADERS:
            raise DatasetConfigError(
                "Unsupported loader {!r}; expected one of: {}.".format(
                    self.loader, ", ".join(sorted(SUPPORTED_LOADERS))
                )
            )

        _require_non_empty_string(self.preprocessing_mode, "preprocessing_mode")
        if self.preprocessing_mode not in SUPPORTED_PREPROCESSING_MODES:
            raise DatasetConfigError(
                "Unsupported preprocessing_mode {!r}; expected one of: {}.".format(
                    self.preprocessing_mode,
                    ", ".join(sorted(SUPPORTED_PREPROCESSING_MODES)),
                )
            )

        signal_columns = _string_tuple(self.signal_columns, "signal_columns")
        metadata_columns = _string_tuple(self.metadata_columns, "metadata_columns")
        cyclic_columns = _string_tuple(
            self.cyclic_columns, "cyclic_columns", allow_empty=True
        )
        temporal_order = _string_tuple(
            self.temporal_order, "temporal_order", allow_empty=True
        )
        object.__setattr__(self, "signal_columns", signal_columns)
        object.__setattr__(self, "metadata_columns", metadata_columns)
        object.__setattr__(self, "cyclic_columns", cyclic_columns)
        object.__setattr__(self, "temporal_order", temporal_order)

        overlap = sorted(set(signal_columns) & set(metadata_columns))
        if overlap:
            raise DatasetConfigError(
                "Signal and metadata columns must be disjoint; overlap: {}.".format(
                    ", ".join(overlap)
                )
            )

        cyclic_not_metadata = sorted(set(cyclic_columns) - set(metadata_columns))
        if cyclic_not_metadata:
            raise DatasetConfigError(
                "Cyclic columns must also be metadata columns; unknown: {}.".format(
                    ", ".join(cyclic_not_metadata)
                )
            )

        known_order_columns = set(signal_columns) | set(metadata_columns)
        if self.timestamp_column is not None:
            timestamp_column = _require_non_empty_string(
                self.timestamp_column, "timestamp_column"
            )
            object.__setattr__(self, "timestamp_column", timestamp_column)
            known_order_columns.add(timestamp_column)

        unknown_order_columns = sorted(set(temporal_order) - known_order_columns)
        if unknown_order_columns:
            raise DatasetConfigError(
                "Temporal-order columns are not declared elsewhere: {}.".format(
                    ", ".join(unknown_order_columns)
                )
            )

        if not isinstance(self.split, SplitConfig):
            raise DatasetConfigError("Field 'split' must be a SplitConfig object.")
        if (
            self.split.mode == "column_values"
            and self.split.column not in known_order_columns
        ):
            raise DatasetConfigError(
                "Split column {!r} is not a declared signal, metadata, or timestamp column.".format(
                    self.split.column
                )
            )
        if self.split.mode == "timestamp" and self.timestamp_column is None:
            raise DatasetConfigError(
                "timestamp split requires a configured timestamp_column."
            )

        if not isinstance(self.dtype_overrides, Mapping):
            raise DatasetConfigError("Field 'dtype_overrides' must be a JSON object.")
        normalized_dtypes: Dict[str, str] = {}
        for column, dtype in self.dtype_overrides.items():
            normalized_column = _require_non_empty_string(column, "dtype_overrides key")
            normalized_dtypes[normalized_column] = _require_non_empty_string(
                dtype, "dtype_overrides.{}".format(column)
            )
        object.__setattr__(
            self, "dtype_overrides", MappingProxyType(normalized_dtypes)
        )

        if not isinstance(self.synthesis_conditions, Mapping):
            raise DatasetConfigError(
                "Field 'synthesis_conditions' must be a JSON object."
            )
        normalized_conditions: Dict[str, Mapping[str, JsonScalar]] = {}
        for profile, conditions in self.synthesis_conditions.items():
            if profile not in SUPPORTED_SYNTHESIS_PROFILES:
                raise DatasetConfigError(
                    "Unsupported synthesis profile {!r}; expected only: {}.".format(
                        profile, ", ".join(sorted(SUPPORTED_SYNTHESIS_PROFILES))
                    )
                )
            if not isinstance(conditions, Mapping):
                raise DatasetConfigError(
                    "Field 'synthesis_conditions.{}' must be a JSON object.".format(
                        profile
                    )
                )

            normalized_profile: Dict[str, JsonScalar] = {}
            for column, value in conditions.items():
                normalized_column = _require_non_empty_string(
                    column, "synthesis_conditions.{} column".format(profile)
                )
                if normalized_column not in metadata_columns:
                    raise DatasetConfigError(
                        "Synthesis condition column {!r} for profile {!r} is not "
                        "declared in metadata_columns.".format(normalized_column, profile)
                    )
                if (
                    value is None
                    or not isinstance(value, (str, int, float, bool))
                    or (isinstance(value, float) and not math.isfinite(value))
                ):
                    raise DatasetConfigError(
                        "Field 'synthesis_conditions.{}.{}' must be a non-null "
                        "finite JSON scalar.".format(profile, normalized_column)
                    )
                normalized_profile[normalized_column] = value

            normalized_conditions[profile] = MappingProxyType(normalized_profile)

        object.__setattr__(
            self,
            "synthesis_conditions",
            MappingProxyType(normalized_conditions),
        )

    @classmethod
    def from_dict(cls, data: Any) -> "DatasetConfig":
        if not isinstance(data, Mapping):
            raise DatasetConfigError("Dataset configuration root must be a JSON object.")

        required = {
            "dataset_id",
            "csv_path",
            "loader",
            "preprocessing_mode",
            "signal_columns",
            "metadata_columns",
            "cyclic_columns",
            "temporal_order",
            "split",
        }
        optional = {"timestamp_column", "dtype_overrides", "synthesis_conditions"}
        _require_fields(data, required, "dataset configuration")
        _reject_unknown_fields(data, required | optional, "dataset configuration")

        return cls(
            dataset_id=data["dataset_id"],
            csv_path=data["csv_path"],
            loader=data["loader"],
            preprocessing_mode=data["preprocessing_mode"],
            signal_columns=data["signal_columns"],
            metadata_columns=data["metadata_columns"],
            cyclic_columns=data["cyclic_columns"],
            temporal_order=data["temporal_order"],
            split=SplitConfig.from_dict(data["split"]),
            timestamp_column=data.get("timestamp_column"),
            dtype_overrides=data.get("dtype_overrides", {}),
            synthesis_conditions=data.get("synthesis_conditions", {}),
        )

    def resolve_csv_path(
        self, base_dir: Optional[Union[str, Path]] = None, *, must_exist: bool = False
    ) -> Path:
        """Resolve the configured CSV path without coupling it to preprocessing."""

        source_path = Path(self.csv_path).expanduser()
        if not source_path.is_absolute():
            root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
            source_path = root / source_path
        source_path = source_path.resolve()

        if must_exist and not source_path.is_file():
            raise DatasetSourceNotFoundError(
                "Dataset CSV does not exist or is not a file: {}".format(source_path)
            )
        return source_path


def load_dataset_config(config_path: Union[str, Path]) -> DatasetConfig:
    """Load and validate one dataset JSON configuration."""

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise DatasetConfigNotFoundError(
            "Dataset configuration does not exist or is not a file: {}".format(path)
        )
    if path.suffix.lower() != ".json":
        raise DatasetConfigError(
            "Dataset configuration must be a .json file: {}".format(path)
        )

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise DatasetConfigError(
            "Invalid JSON in {} at line {}, column {}: {}".format(
                path, exc.lineno, exc.colno, exc.msg
            )
        ) from exc

    try:
        return DatasetConfig.from_dict(raw_config)
    except DatasetConfigError as exc:
        raise DatasetConfigError(
            "Invalid dataset configuration {}: {}".format(path, exc)
        ) from exc


def load_dataset_config_by_id(
    dataset_id: str, config_dir: Optional[Union[str, Path]] = None
) -> DatasetConfig:
    """Load ``<dataset_id>.json`` from the dataset configuration directory."""

    normalized_id = _require_non_empty_string(dataset_id, "dataset_id")
    if Path(normalized_id).name != normalized_id:
        raise DatasetConfigError("Field 'dataset_id' must not contain path separators.")
    directory = Path(config_dir) if config_dir is not None else DEFAULT_DATASET_CONFIG_DIR
    config = load_dataset_config(directory / "{}.json".format(normalized_id))
    if config.dataset_id != normalized_id:
        raise DatasetConfigError(
            "Configuration ID {!r} does not match requested dataset ID {!r}.".format(
                config.dataset_id, normalized_id
            )
        )
    return config
