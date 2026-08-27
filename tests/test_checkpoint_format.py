import copy
import io
import json
from argparse import Namespace

import pytest
import torch

from checkpoint_utils import (
    CheckpointCompatibilityError,
    CheckpointFormatError,
    apply_structured_checkpoint_args,
    build_structured_checkpoint,
    checkpoint_preprocessing_state,
    checkpoint_state_dict,
    dataset_config_snapshot,
    is_structured_checkpoint,
    validate_structured_checkpoint,
)
from data_utils import Preprocessor
from training_utils import fetchModel, resolve_model_columns


def _args():
    return Namespace(
        backbone="S4",
        hdim=64,
        layers=4,
        num_res_layers=1,
        res_channels=4,
        skip_channels=4,
        diff_step_embed_in=8,
        diff_step_embed_mid=8,
        diff_step_embed_out=8,
        s4_lmax=8,
        s4_dstate=8,
        s4_dropout=0.0,
        s4_bidirectional=True,
        s4_layernorm=True,
        window_size=8,
        stride=7,
        timesteps=4,
        beta_0=0.0001,
        beta_T=0.02,
        lr=0.0001,
        batch_size=2,
        seed=42,
        propCycEnc=False,
    )


def _occupancy_checkpoint():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)
    frame = preprocessor.df_cleaned.loc[preprocessor.train_indices]
    signal_indices, metadata_indices = resolve_model_columns(frame, preprocessor)
    args = _args()
    torch.manual_seed(args.seed)
    model = fetchModel(len(frame.columns), len(signal_indices), args).eval()
    checkpoint = build_structured_checkpoint(
        model=model,
        dataset_id="UCIOccupancyDetection",
        frame=frame,
        preprocessor=preprocessor,
        signal_indices=signal_indices,
        metadata_indices=metadata_indices,
        args=args,
        effective_training_stride=1,
        optimizer_steps=2,
    )
    return checkpoint, model, preprocessor, frame, signal_indices, metadata_indices


def _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices):
    return validate_structured_checkpoint(
        checkpoint,
        dataset_id="UCIOccupancyDetection",
        frame=frame,
        preprocessor=preprocessor,
        signal_indices=signal_indices,
        metadata_indices=metadata_indices,
    )


def test_structured_checkpoint_roundtrip_reconstructs_identical_model_output():
    checkpoint, source, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    inputs = torch.randn(2, 8, len(frame.columns))
    timesteps = torch.tensor([[1], [3]])
    with torch.no_grad():
        expected = source(inputs, timesteps)

    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)
    serialized.seek(0)
    restored_checkpoint = torch.load(serialized, map_location="cpu")

    restored_args = Namespace()
    apply_structured_checkpoint_args(restored_args, restored_checkpoint)
    in_dim, out_dim = _validate(
        restored_checkpoint,
        preprocessor,
        frame,
        signal_indices,
        metadata_indices,
    )
    restored = fetchModel(in_dim, out_dim, restored_args).eval()
    load_result = restored.load_state_dict(
        checkpoint_state_dict(restored_checkpoint), strict=True
    )
    with torch.no_grad():
        actual = restored(inputs, timesteps)

    assert is_structured_checkpoint(restored_checkpoint)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert restored_args.window_size == 8
    assert restored_args.timesteps == 4
    assert restored_args.res_channels == 4
    assert restored_args.propCycEnc is False
    torch.testing.assert_close(actual, expected)


def test_structured_checkpoint_contains_required_provenance():
    checkpoint, _, preprocessor, _, _, _ = _occupancy_checkpoint()

    assert checkpoint["dataset_id"] == "UCIOccupancyDetection"
    assert checkpoint["format_version"] == 2
    assert checkpoint["model"]["in_dim"] == 5
    assert checkpoint["model"]["out_dim"] == 4
    assert checkpoint["model"]["signal_indices"] == [0, 1, 2, 3]
    assert checkpoint["model"]["metadata_indices"] == [4]
    assert checkpoint["training"]["effective_stride"] == 1
    assert checkpoint["training"]["requested_stride"] == 7
    assert checkpoint["training"]["optimizer_steps"] == 2
    assert checkpoint["diffusion"]["timesteps"] == 4
    assert checkpoint["encoding"]["cyclic_columns"] == []
    assert checkpoint["preprocessing"]["mode"] == "train_only"
    assert checkpoint["dataset_config"] == dataset_config_snapshot(
        preprocessor.dataset_config
    )
    json.dumps(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "state_dict"
        }
    )


def test_legacy_state_dict_checkpoint_remains_strict_loadable():
    checkpoint, source, _, _, _, _ = _occupancy_checkpoint()
    legacy_checkpoint = source.state_dict()
    restored = fetchModel(
        checkpoint["model"]["in_dim"], checkpoint["model"]["out_dim"], _args()
    )

    assert not is_structured_checkpoint(legacy_checkpoint)
    result = restored.load_state_dict(
        checkpoint_state_dict(legacy_checkpoint), strict=True
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert set(checkpoint_state_dict(legacy_checkpoint)) == set(checkpoint["state_dict"])

    with pytest.raises(CheckpointFormatError, match="Legacy checkpoints.*state"):
        checkpoint_preprocessing_state(legacy_checkpoint)


def test_structured_v1_is_recognized_but_requires_v2_for_leakage_free_synthesis():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    checkpoint["format_version"] = 1
    checkpoint.pop("preprocessing")
    checkpoint["dataset_config"].pop("preprocessing_mode")

    restored_args = Namespace()
    apply_structured_checkpoint_args(restored_args, checkpoint)

    assert is_structured_checkpoint(checkpoint)
    assert restored_args.window_size == 8
    assert _validate(
        checkpoint, preprocessor, frame, signal_indices, metadata_indices
    ) == (5, 4)
    with pytest.raises(
        CheckpointCompatibilityError,
        match="checkpoint v1.*leakage-free synthesis.*checkpoint v2",
    ):
        checkpoint_preprocessing_state(checkpoint)


def test_structured_v2_missing_preprocessing_is_malformed_not_v1():
    checkpoint, _, _, _, _, _ = _occupancy_checkpoint()
    checkpoint.pop("preprocessing")

    with pytest.raises(CheckpointFormatError, match="preprocessing.*missing"):
        checkpoint_preprocessing_state(checkpoint)


def test_unknown_structured_checkpoint_version_is_rejected():
    checkpoint, _, _, _, _, _ = _occupancy_checkpoint()
    checkpoint["format_version"] = 3

    with pytest.raises(CheckpointFormatError, match="version 3.*1 and 2"):
        apply_structured_checkpoint_args(Namespace(), checkpoint)


def test_checkpoint_rejects_missing_signal_column():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    checkpoint["model"]["signal_columns"][0] = "MissingSignal"

    with pytest.raises(CheckpointCompatibilityError, match="Signal columns.*missing"):
        _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices)


def test_checkpoint_rejects_signal_column_order_mismatch():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    signal_columns = checkpoint["model"]["signal_columns"]
    signal_columns[0], signal_columns[1] = signal_columns[1], signal_columns[0]

    with pytest.raises(CheckpointCompatibilityError, match="Signal column order differs"):
        _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices)


def test_checkpoint_rejects_dimension_mismatch():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    checkpoint["model"]["out_dim"] = 6

    with pytest.raises(CheckpointCompatibilityError, match="dimensions are incompatible"):
        _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (
            lambda checkpoint: checkpoint["model"].pop("metadata_columns"),
            CheckpointFormatError,
            "missing field.*metadata_columns",
        ),
        (
            lambda checkpoint: checkpoint["model"].update(
                encoded_metadata_columns=["corrupted"]
            ),
            CheckpointCompatibilityError,
            "Encoded metadata columns.*corrupted",
        ),
    ],
)
def test_checkpoint_rejects_missing_or_corrupted_metadata(
    mutation, error_type, message
):
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    mutation(checkpoint)

    with pytest.raises(error_type, match=message):
        _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices)


def test_checkpoint_rejects_incompatible_dataset_config():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )
    checkpoint["dataset_config"]["split"]["cutoff"] = "2015-02-08T00:00:00Z"

    with pytest.raises(CheckpointCompatibilityError, match="DatasetConfig.*split"):
        _validate(checkpoint, preprocessor, frame, signal_indices, metadata_indices)


def test_occupancy_checkpoint_is_compatible_with_current_runtime():
    checkpoint, _, preprocessor, frame, signal_indices, metadata_indices = (
        _occupancy_checkpoint()
    )

    assert _validate(
        copy.deepcopy(checkpoint),
        preprocessor,
        frame,
        signal_indices,
        metadata_indices,
    ) == (5, 4)
