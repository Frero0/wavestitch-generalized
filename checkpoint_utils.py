"""Structured WaveStitch checkpoint construction and compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


CHECKPOINT_FORMAT = "wavestitch"
CHECKPOINT_VERSION_V1 = 1
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = frozenset(
    {CHECKPOINT_VERSION_V1, CHECKPOINT_VERSION}
)

ARCHITECTURE_FIELDS = (
    "backbone",
    "hdim",
    "layers",
    "num_res_layers",
    "res_channels",
    "skip_channels",
    "diff_step_embed_in",
    "diff_step_embed_mid",
    "diff_step_embed_out",
    "s4_lmax",
    "s4_dstate",
    "s4_dropout",
    "s4_bidirectional",
    "s4_layernorm",
)


class CheckpointFormatError(ValueError):
    """Raised when a structured checkpoint is incomplete or malformed."""


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used with the current dataset layout."""


def dataset_config_snapshot(config):
    """Return a plain-Python, serializable snapshot without embedding dataset data."""

    if config is None:
        return None

    split = {"mode": config.split.mode}
    if config.split.mode == "column_values":
        split.update(
            column=config.split.column,
            test_values=list(config.split.test_values),
        )
    elif config.split.mode == "ratio":
        split["train_ratio"] = config.split.train_ratio
    elif config.split.mode == "timestamp":
        split["cutoff"] = config.split.cutoff

    return {
        "dataset_id": config.dataset_id,
        "csv_path": config.csv_path,
        "loader": config.loader,
        "preprocessing_mode": config.preprocessing_mode,
        "timestamp_column": config.timestamp_column,
        "signal_columns": list(config.signal_columns),
        "metadata_columns": list(config.metadata_columns),
        "cyclic_columns": list(config.cyclic_columns),
        "dtype_overrides": dict(config.dtype_overrides),
        "temporal_order": list(config.temporal_order),
        "split": split,
        "synthesis_conditions": {
            profile: dict(conditions)
            for profile, conditions in config.synthesis_conditions.items()
        },
    }


def build_structured_checkpoint(
    *,
    model,
    dataset_id,
    frame,
    preprocessor,
    signal_indices,
    metadata_indices,
    args,
    effective_training_stride,
    optimizer_steps,
):
    """Build the versioned checkpoint envelope used by new training runs."""

    signal_indices = np.asarray(signal_indices)
    metadata_indices = np.asarray(metadata_indices)
    model_columns = frame.columns.tolist()
    architecture = {
        field: getattr(args, field)
        for field in ARCHITECTURE_FIELDS
    }

    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "dataset_id": dataset_id,
        "model": {
            "in_dim": len(model_columns),
            "out_dim": len(signal_indices),
            "signal_columns": frame.columns[signal_indices].tolist(),
            "metadata_columns": list(preprocessor.metadata_columns),
            "encoded_metadata_columns": frame.columns[metadata_indices].tolist(),
            "model_columns": model_columns,
            "signal_indices": signal_indices.tolist(),
            "metadata_indices": metadata_indices.tolist(),
            "architecture": architecture,
        },
        "training": {
            "window_size": args.window_size,
            "effective_stride": effective_training_stride,
            "requested_stride": args.stride,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "optimizer_steps": optimizer_steps,
        },
        "diffusion": {
            "timesteps": args.timesteps,
            "beta_0": args.beta_0,
            "beta_T": args.beta_T,
        },
        "encoding": {
            "proportional_cyclic_encoding": bool(args.propCycEnc),
            "cyclic_columns": list(preprocessor.cyclic_encoded_columns),
            "encoded_metadata_columns": frame.columns[metadata_indices].tolist(),
        },
        "preprocessing": preprocessor.preprocessing_state_dict(),
        "dataset_config": dataset_config_snapshot(preprocessor.dataset_config),
    }


def is_structured_checkpoint(checkpoint):
    return (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("checkpoint_format") == CHECKPOINT_FORMAT
    )


def checkpoint_state_dict(checkpoint):
    """Return the state dict from either the new envelope or a legacy checkpoint."""

    if is_structured_checkpoint(checkpoint):
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise CheckpointFormatError(
                "Structured checkpoint field 'state_dict' is missing or corrupted."
            )
        return state_dict
    if not isinstance(checkpoint, Mapping):
        raise CheckpointFormatError(
            "Legacy checkpoint must contain a state_dict mapping."
        )
    return checkpoint


def checkpoint_preprocessing_state(checkpoint):
    """Return the fitted scaler/encoder state required for synthesis."""

    if not is_structured_checkpoint(checkpoint):
        raise CheckpointFormatError(
            "Legacy checkpoints do not contain fitted preprocessing state."
        )
    version = _require_supported_version(checkpoint)
    if version == CHECKPOINT_VERSION_V1:
        raise CheckpointCompatibilityError(
            "Structured checkpoint v1 does not contain fitted preprocessing "
            "state; leakage-free synthesis requires a structured checkpoint v2."
        )
    state = _require_mapping(checkpoint, "preprocessing")
    _require_fields(state, {"mode", "scaler", "encoders"}, "preprocessing")
    if state["mode"] not in {"train_only", "upstream_legacy"}:
        raise CheckpointFormatError(
            "Structured checkpoint contains unsupported preprocessing mode {!r}."
            .format(state["mode"])
        )
    return dict(state)


def _require_mapping(checkpoint, field):
    value = checkpoint.get(field)
    if not isinstance(value, Mapping):
        raise CheckpointFormatError(
            "Structured checkpoint field {!r} is missing or corrupted.".format(field)
        )
    return value


def _require_fields(mapping, fields, context):
    missing = sorted(set(fields) - set(mapping))
    if missing:
        raise CheckpointFormatError(
            "Structured checkpoint {} is missing field(s): {}.".format(
                context, ", ".join(missing)
            )
        )


def _require_supported_version(checkpoint):
    version = checkpoint.get("format_version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise CheckpointFormatError(
            "Unsupported WaveStitch checkpoint version {!r}; supported structured "
            "versions are 1 and 2.".format(version)
        )
    return version


def apply_structured_checkpoint_args(args, checkpoint):
    """Populate model/diffusion/encoding runtime arguments from a checkpoint."""

    if not is_structured_checkpoint(checkpoint):
        raise CheckpointFormatError("Cannot apply metadata from a legacy checkpoint.")
    _require_supported_version(checkpoint)

    model = _require_mapping(checkpoint, "model")
    architecture = _require_mapping(model, "architecture")
    training = _require_mapping(checkpoint, "training")
    diffusion = _require_mapping(checkpoint, "diffusion")
    encoding = _require_mapping(checkpoint, "encoding")
    _require_fields(architecture, ARCHITECTURE_FIELDS, "model.architecture")
    _require_fields(training, {"window_size"}, "training")
    _require_fields(diffusion, {"timesteps", "beta_0", "beta_T"}, "diffusion")
    _require_fields(
        encoding, {"proportional_cyclic_encoding"}, "encoding"
    )

    for field in ARCHITECTURE_FIELDS:
        setattr(args, field, architecture[field])
    args.window_size = training["window_size"]
    args.timesteps = diffusion["timesteps"]
    args.beta_0 = diffusion["beta_0"]
    args.beta_T = diffusion["beta_T"]
    args.propCycEnc = encoding["proportional_cyclic_encoding"]
    return args


def validate_structured_checkpoint(
    checkpoint,
    *,
    dataset_id,
    frame,
    preprocessor,
    signal_indices,
    metadata_indices,
):
    """Fail clearly when current preprocessing is incompatible with a checkpoint."""

    if not is_structured_checkpoint(checkpoint):
        raise CheckpointFormatError("Compatibility metadata is absent from a legacy checkpoint.")
    checkpoint_state_dict(checkpoint)
    version = _require_supported_version(checkpoint)
    if checkpoint.get("dataset_id") != dataset_id:
        raise CheckpointCompatibilityError(
            "Checkpoint dataset {!r} is incompatible with current dataset {!r}.".format(
                checkpoint.get("dataset_id"), dataset_id
            )
        )

    model = _require_mapping(checkpoint, "model")
    _require_fields(
        model,
        {
            "in_dim",
            "out_dim",
            "signal_columns",
            "metadata_columns",
            "encoded_metadata_columns",
            "model_columns",
            "signal_indices",
            "metadata_indices",
            "architecture",
        },
        "model",
    )
    _require_mapping(model, "architecture")

    current_signal_columns = frame.columns[np.asarray(signal_indices)].tolist()
    saved_signal_columns = model["signal_columns"]
    if not isinstance(saved_signal_columns, list):
        raise CheckpointFormatError(
            "Structured checkpoint model.signal_columns is missing or corrupted."
        )
    if saved_signal_columns != current_signal_columns:
        if set(saved_signal_columns) == set(current_signal_columns):
            raise CheckpointCompatibilityError(
                "Signal column order differs between checkpoint and current dataset: "
                "checkpoint={}, current={}.".format(
                    saved_signal_columns, current_signal_columns
                )
            )
        missing = [
            column for column in saved_signal_columns
            if column not in current_signal_columns
        ]
        extra = [
            column for column in current_signal_columns
            if column not in saved_signal_columns
        ]
        raise CheckpointCompatibilityError(
            "Signal columns are incompatible; missing from current dataset={}, "
            "unexpected current columns={}.".format(missing, extra)
        )

    current_metadata_columns = list(preprocessor.metadata_columns)
    saved_metadata_columns = model["metadata_columns"]
    if not isinstance(saved_metadata_columns, list):
        raise CheckpointFormatError(
            "Structured checkpoint model.metadata_columns is missing or corrupted."
        )
    if saved_metadata_columns != current_metadata_columns:
        raise CheckpointCompatibilityError(
            "Metadata columns are incompatible: checkpoint={}, current={}.".format(
                saved_metadata_columns, current_metadata_columns
            )
        )

    current_encoded_metadata = frame.columns[np.asarray(metadata_indices)].tolist()
    if model["encoded_metadata_columns"] != current_encoded_metadata:
        raise CheckpointCompatibilityError(
            "Encoded metadata columns are missing, reordered, or corrupted: "
            "checkpoint={}, current={}.".format(
                model["encoded_metadata_columns"], current_encoded_metadata
            )
        )

    current_model_columns = frame.columns.tolist()
    saved_model_columns = model["model_columns"]
    if saved_model_columns != current_model_columns:
        missing = [
            column for column in saved_model_columns
            if column not in current_model_columns
        ]
        if missing:
            raise CheckpointCompatibilityError(
                "Current dataset is missing model column(s) required by checkpoint: {}.".format(
                    ", ".join(missing)
                )
            )
        raise CheckpointCompatibilityError(
            "Model column order differs between checkpoint and current dataset."
        )

    current_in_dim = len(current_model_columns)
    current_out_dim = len(signal_indices)
    if model["in_dim"] != current_in_dim or model["out_dim"] != current_out_dim:
        raise CheckpointCompatibilityError(
            "Checkpoint dimensions are incompatible: checkpoint in_dim/out_dim={}/{}, "
            "current={}/{}.".format(
                model["in_dim"], model["out_dim"], current_in_dim, current_out_dim
            )
        )
    if model["signal_indices"] != np.asarray(signal_indices).tolist():
        raise CheckpointCompatibilityError(
            "Signal column indices differ between checkpoint and current dataset."
        )
    if model["metadata_indices"] != np.asarray(metadata_indices).tolist():
        raise CheckpointCompatibilityError(
            "Metadata column indices differ between checkpoint and current dataset."
        )

    saved_config = checkpoint.get("dataset_config")
    current_config = dataset_config_snapshot(preprocessor.dataset_config)
    if current_config is not None:
        if not isinstance(saved_config, Mapping):
            raise CheckpointFormatError(
                "Structured checkpoint dataset_config is missing or corrupted."
            )
        compatibility_fields = [
            "dataset_id",
            "loader",
            "timestamp_column",
            "signal_columns",
            "metadata_columns",
            "cyclic_columns",
            "dtype_overrides",
            "temporal_order",
            "split",
            "synthesis_conditions",
        ]
        if version >= CHECKPOINT_VERSION:
            compatibility_fields.append("preprocessing_mode")
        mismatched = [
            field for field in compatibility_fields
            if saved_config.get(field) != current_config.get(field)
        ]
        if mismatched:
            raise CheckpointCompatibilityError(
                "DatasetConfig is incompatible for field(s): {}.".format(
                    ", ".join(mismatched)
                )
            )

    if version >= CHECKPOINT_VERSION:
        preprocessing = checkpoint_preprocessing_state(checkpoint)
        if preprocessing["mode"] != preprocessor.preprocessing_mode:
            raise CheckpointCompatibilityError(
                "Checkpoint preprocessing mode {!r} is incompatible with current "
                "configuration mode {!r}.".format(
                    preprocessing["mode"], preprocessor.preprocessing_mode
                )
            )

    encoding = _require_mapping(checkpoint, "encoding")
    _require_fields(
        encoding,
        {
            "proportional_cyclic_encoding",
            "cyclic_columns",
            "encoded_metadata_columns",
        },
        "encoding",
    )
    if encoding["proportional_cyclic_encoding"] != bool(preprocessor.pce):
        raise CheckpointCompatibilityError(
            "Checkpoint encoding mode is incompatible with the current Preprocessor."
        )
    if encoding["cyclic_columns"] != list(preprocessor.cyclic_encoded_columns):
        raise CheckpointCompatibilityError(
            "Checkpoint cyclic metadata columns are incompatible with the current Preprocessor."
        )
    if encoding["encoded_metadata_columns"] != current_encoded_metadata:
        raise CheckpointCompatibilityError(
            "Checkpoint encoded metadata layout is incompatible with the current Preprocessor."
        )

    return model["in_dim"], model["out_dim"]
