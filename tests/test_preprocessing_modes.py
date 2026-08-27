import io
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from wavestitch import dataset_config
from wavestitch import data_utils
from wavestitch.checkpoint_utils import (
    CheckpointCompatibilityError,
    build_structured_checkpoint,
    checkpoint_preprocessing_state,
    validate_structured_checkpoint,
)
from wavestitch.data_utils import MetadataCategoryError, PreprocessingStateError, Preprocessor
from scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning import (
    build_synthesis_preprocessor,
)
from wavestitch.training_utils import resolve_model_columns


def _rows():
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    groups = ["A", "B", "A", "B", "A", "A", "B", "A"]
    return [
        {
            "timestamp": timestamp.isoformat(),
            "signal_a": float(index + 1),
            "signal_b": float((index + 1) * 10),
            "group": groups[index],
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _write_runtime(tmp_path, monkeypatch, rows=None, *, dataset_id="TrainOnlyData"):
    csv_path = tmp_path / "train_only.csv"
    pd.DataFrame(_rows() if rows is None else rows).to_csv(csv_path, index=False)
    config = {
        "dataset_id": dataset_id,
        "csv_path": str(csv_path),
        "loader": "flat_csv",
        "preprocessing_mode": "train_only",
        "timestamp_column": "timestamp",
        "signal_columns": ["signal_a", "signal_b"],
        "metadata_columns": ["group"],
        "cyclic_columns": ["group"],
        "dtype_overrides": {"group": "string"},
        "temporal_order": ["timestamp"],
        "split": {"mode": "timestamp", "cutoff": "2024-01-01T05:00:00Z"},
        "synthesis_conditions": {"C": {}},
    }
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    (config_dir / (dataset_id + ".json")).write_text(
        json.dumps(config), encoding="utf-8"
    )
    monkeypatch.setattr(dataset_config, "DEFAULT_DATASET_CONFIG_DIR", config_dir)
    return csv_path


def _args():
    return Namespace(
        backbone="S4", hdim=4, layers=1, num_res_layers=1,
        res_channels=4, skip_channels=4, diff_step_embed_in=8,
        diff_step_embed_mid=8, diff_step_embed_out=8, s4_lmax=8,
        s4_dstate=8, s4_dropout=0.0, s4_bidirectional=True,
        s4_layernorm=True, window_size=4, stride=1, timesteps=4,
        beta_0=0.0001, beta_T=0.02, lr=0.0001, batch_size=2,
        seed=42, propCycEnc=True,
    )


class _NoopModel:
    @staticmethod
    def state_dict():
        return {}


def _checkpoint(preprocessor):
    frame = preprocessor.df_cleaned.loc[preprocessor.train_indices]
    signal_indices, metadata_indices = resolve_model_columns(frame, preprocessor)
    return build_structured_checkpoint(
        model=_NoopModel(), dataset_id="TrainOnlyData", frame=frame,
        preprocessor=preprocessor, signal_indices=signal_indices,
        metadata_indices=metadata_indices, args=_args(),
        effective_training_stride=1, optimizer_steps=0,
    )


def test_train_only_test_changes_do_not_affect_fit_but_train_changes_do(
    tmp_path, monkeypatch
):
    _write_runtime(tmp_path, monkeypatch)
    baseline = Preprocessor("TrainOnlyData", True)

    test_changed = _rows()
    for row in test_changed[5:]:
        row["signal_a"] += 10000
        row["signal_b"] -= 5000
        row["group"] = "B"
    _write_runtime(tmp_path, monkeypatch, test_changed)
    changed_test = Preprocessor("TrainOnlyData", True)

    np.testing.assert_array_equal(changed_test.scaler.mean_, baseline.scaler.mean_)
    np.testing.assert_array_equal(changed_test.scaler.scale_, baseline.scaler.scale_)
    np.testing.assert_array_equal(
        changed_test.encoders["group"].categories,
        baseline.encoders["group"].categories,
    )
    np.testing.assert_array_equal(
        np.asarray(changed_test.encoders["group"].angles),
        np.asarray(baseline.encoders["group"].angles),
    )

    train_changed = _rows()
    train_changed[0]["signal_a"] += 100
    train_changed[0]["signal_b"] -= 200
    train_changed[0]["group"] = "B"
    _write_runtime(tmp_path, monkeypatch, train_changed)
    changed_train = Preprocessor("TrainOnlyData", True)

    assert not np.array_equal(changed_train.scaler.mean_, baseline.scaler.mean_)
    assert not np.array_equal(changed_train.scaler.scale_, baseline.scaler.scale_)
    assert changed_train.encoders["group"].mapper != (
        baseline.encoders["group"].mapper
    )


def test_train_only_uses_one_state_for_both_splits_and_roundtrips(
    tmp_path, monkeypatch
):
    _write_runtime(tmp_path, monkeypatch)
    preprocessor = Preprocessor("TrainOnlyData", True)
    train_original = preprocessor.df_orig.loc[preprocessor.train_indices]
    all_original = preprocessor.df_orig[["signal_a", "signal_b"]]

    expected_scaler = StandardScaler().fit(train_original[["signal_a", "signal_b"]])
    np.testing.assert_allclose(preprocessor.scaler.mean_, expected_scaler.mean_)
    np.testing.assert_allclose(preprocessor.scaler.scale_, expected_scaler.scale_)
    np.testing.assert_allclose(
        preprocessor.df_cleaned[["signal_a", "signal_b"]],
        expected_scaler.transform(all_original),
    )
    assert int(preprocessor.scaler.n_samples_seen_) == 5
    assert preprocessor.df_cleaned.index.equals(preprocessor.df_orig.index)

    decoded = preprocessor.decode(preprocessor.df_cleaned, rescale=True)
    np.testing.assert_allclose(
        decoded[["signal_a", "signal_b"]], all_original, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_series_equal(decoded["group"], preprocessor.df_orig["group"])


def test_train_only_rejects_metadata_category_unseen_in_train(tmp_path, monkeypatch):
    rows = _rows()
    rows[-1]["group"] = "C"
    _write_runtime(tmp_path, monkeypatch, rows)

    with pytest.raises(MetadataCategoryError, match="group.*not seen.*train"):
        Preprocessor("TrainOnlyData", True)


def test_checkpoint_roundtrip_preserves_preprocessing_and_synthesis_does_not_refit(
    tmp_path, monkeypatch
):
    _write_runtime(tmp_path, monkeypatch)
    fitted = Preprocessor("TrainOnlyData", True)
    checkpoint = _checkpoint(fitted)
    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)
    serialized.seek(0)
    restored_checkpoint = torch.load(serialized, map_location="cpu")

    assert checkpoint_preprocessing_state(restored_checkpoint) == (
        fitted.preprocessing_state_dict()
    )

    def fail_fit(*args, **kwargs):
        raise AssertionError("synthesis attempted to refit the scaler")

    def fail_encoder_init(*args, **kwargs):
        raise AssertionError("synthesis attempted to refit an encoder")

    monkeypatch.setattr(data_utils.StandardScaler, "fit_transform", fail_fit)
    monkeypatch.setattr(data_utils.CyclicEncoder, "__init__", fail_encoder_init)
    restored = build_synthesis_preprocessor(
        "TrainOnlyData", True, restored_checkpoint
    )

    pd.testing.assert_frame_equal(restored.df_cleaned, fitted.df_cleaned)
    assert restored.preprocessing_state_dict() == fitted.preprocessing_state_dict()


def test_checkpoint_and_config_preprocessing_mode_mismatch_is_rejected(
    tmp_path, monkeypatch
):
    _write_runtime(tmp_path, monkeypatch)
    fitted = Preprocessor("TrainOnlyData", True)
    checkpoint = _checkpoint(fitted)
    checkpoint["preprocessing"]["mode"] = "upstream_legacy"

    with pytest.raises(PreprocessingStateError, match="mode.*incompatible"):
        build_synthesis_preprocessor("TrainOnlyData", True, checkpoint)

    compatible_frame = fitted.df_cleaned.loc[fitted.train_indices]
    signal_indices, metadata_indices = resolve_model_columns(
        compatible_frame, fitted
    )
    with pytest.raises(CheckpointCompatibilityError, match="preprocessing mode"):
        validate_structured_checkpoint(
            checkpoint, dataset_id="TrainOnlyData", frame=compatible_frame,
            preprocessor=fitted, signal_indices=signal_indices,
            metadata_indices=metadata_indices,
        )


def test_occupancy_train_only_regression_fits_exactly_6129_train_rows():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)
    train = preprocessor.df_orig.loc[preprocessor.train_indices]
    expected = StandardScaler().fit(train[preprocessor.signal_columns])

    assert preprocessor.preprocessing_mode == "train_only"
    assert len(preprocessor.train_indices) == 6129
    assert len(preprocessor.test_indices) == 2014
    assert preprocessor.cols_to_scale == [
        "Temperature", "Humidity", "Light", "CO2"
    ]
    assert preprocessor.encoders == {}
    assert preprocessor.df_cleaned.columns.tolist() == [
        "Temperature", "Humidity", "Light", "CO2", "Occupancy"
    ]
    assert int(preprocessor.scaler.n_samples_seen_) == 6129
    np.testing.assert_allclose(preprocessor.scaler.mean_, expected.mean_, rtol=1e-12)
    np.testing.assert_allclose(preprocessor.scaler.scale_, expected.scale_, rtol=1e-12)
    pd.testing.assert_series_equal(
        preprocessor.df_cleaned["Occupancy"], preprocessor.df_orig["Occupancy"]
    )
