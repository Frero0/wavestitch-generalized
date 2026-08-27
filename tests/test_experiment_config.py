import argparse

import numpy as np
import pandas as pd
import pytest

import experiment_runtime
from dataset_config import load_dataset_config_by_id
from experiment_config import (
    ExperimentCheckpointConflictError,
    ExperimentConfig,
    ExperimentConfigError,
    add_common_cli_arguments,
    add_synthesis_cli_arguments,
    add_training_cli_arguments,
    load_experiment_config,
    resolve_experiment,
    validate_checkpoint_experiment_conflicts,
)
from experiment_runtime import build_dry_run_report


def _minimal_config(**overrides):
    data = {
        "dataset_id": "UCIOccupancyDetection",
        "training": {"window_size": 8, "batch_size": 4},
        "synthesis": {"profile": "C", "stride": 1},
    }
    data.update(overrides)
    return data


def _structured_checkpoint():
    resolved = resolve_experiment(
        ExperimentConfig.from_dict(_minimal_config()), {}
    )
    architecture = {
        field: resolved.values[field]
        for field in (
            "backbone", "hdim", "layers", "num_res_layers", "res_channels",
            "skip_channels", "diff_step_embed_in", "diff_step_embed_mid",
            "diff_step_embed_out", "s4_lmax", "s4_dstate", "s4_dropout",
            "s4_bidirectional", "s4_layernorm",
        )
    }
    return {
        "checkpoint_format": "wavestitch",
        "format_version": 1,
        "dataset_id": "UCIOccupancyDetection",
        "model": {"architecture": architecture},
        "training": {"window_size": 8},
        "diffusion": {
            "timesteps": 200,
            "beta_0": 0.0001,
            "beta_T": 0.02,
        },
        "encoding": {"proportional_cyclic_encoding": False},
    }


def test_valid_experiment_config_parses_typed_sections():
    config = ExperimentConfig.from_dict(
        _minimal_config(
            diffusion={"timesteps": 4, "beta_0": 0.0001, "beta_T": 0.02},
            architecture={"res_channels": 8, "s4_bidirectional": False},
            artifacts={"output_dir": "artifacts"},
        )
    )

    assert config.dataset_id == "UCIOccupancyDetection"
    assert config.training.window_size == 8
    assert config.diffusion.timesteps == 4
    assert config.architecture.res_channels == 8
    assert config.architecture.s4_bidirectional is False
    assert config.synthesis.profile == "C"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "dataset_id"),
        ({"dataset_id": "x", "unknown": 1}, "Unknown field"),
        ({"dataset_id": "x", "training": []}, "JSON object"),
        ({"dataset_id": "x", "training": {"window_size": 0}}, "positive integer"),
        ({"dataset_id": "x", "diffusion": {"beta_0": 2}}, "between 0 and 1"),
        ({"dataset_id": "x", "architecture": {"s4_dropout": 1}}, r"\[0, 1\)"),
        ({"dataset_id": "x", "synthesis": {"profile": "X"}}, "C, M, F"),
    ],
)
def test_invalid_experiment_config_is_rejected(data, message):
    with pytest.raises(ExperimentConfigError, match=message):
        ExperimentConfig.from_dict(data)


def test_repository_experiment_configs_are_valid():
    metro = load_experiment_config("configs/experiments/metrotraffic_upstream.json")
    occupancy = load_experiment_config("configs/experiments/uci_occupancy_smoke.json")

    assert metro.dataset_id == "MetroTraffic"
    assert metro.training.epochs == 300
    assert metro.synthesis.stride == 8
    assert occupancy.dataset_id == "UCIOccupancyDetection"
    assert occupancy.training.max_steps == 2
    assert occupancy.synthesis.max_windows == 1


def test_invalid_json_experiment_file_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"dataset_id": ', encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="Invalid JSON"):
        load_experiment_config(path)


def test_experiment_file_must_be_json(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("dataset_id: example", encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="must be a .json"):
        load_experiment_config(path)


def test_precedence_is_cli_then_config_then_default():
    config = ExperimentConfig.from_dict(
        _minimal_config(training={"window_size": 8, "epochs": 12})
    )
    resolved = resolve_experiment(config, {"epochs": 3})

    assert resolved.values["epochs"] == 3
    assert resolved.sources["epochs"] == "CLI override"
    assert resolved.values["window_size"] == 8
    assert resolved.sources["window_size"] == "experiment config"
    assert resolved.values["learning_rate"] == 0.0001
    assert resolved.sources["learning_rate"] == "default"


def test_missing_dataset_is_rejected_without_checkpoint_inference():
    with pytest.raises(ExperimentConfigError, match="dataset ID is required"):
        resolve_experiment(None, {})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"training_batch_size": 0}, "positive integer"),
        ({"learning_rate": -1}, "positive number"),
        ({"beta_0": 0.03, "beta_T": 0.02}, "smaller than beta_T"),
        ({"s4_dropout": 1.0}, r"\[0, 1\)"),
    ],
)
def test_cli_values_outside_valid_range_are_rejected(override, message):
    with pytest.raises(ExperimentConfigError, match=message):
        resolve_experiment(
            ExperimentConfig.from_dict(_minimal_config()), override
        )


@pytest.mark.parametrize(
    "override",
    [
        {"window_size": 16},
        {"timesteps": 10},
        {"res_channels": 16},
        {"dataset": "MetroTraffic"},
    ],
)
def test_explicit_config_or_cli_checkpoint_conflicts_are_rejected(override):
    checkpoint = _structured_checkpoint()
    resolved = resolve_experiment(
        ExperimentConfig.from_dict(_minimal_config()), override
    )

    with pytest.raises(
        ExperimentCheckpointConflictError,
        match="conflicts with structured checkpoint",
    ):
        validate_checkpoint_experiment_conflicts(checkpoint, resolved)


def test_defaults_do_not_override_structured_checkpoint_metadata():
    checkpoint = _structured_checkpoint()
    checkpoint["model"]["architecture"]["res_channels"] = 7
    config = ExperimentConfig.from_dict(
        {"dataset_id": "UCIOccupancyDetection", "synthesis": {"profile": "C"}}
    )
    resolved = resolve_experiment(config, {})

    validate_checkpoint_experiment_conflicts(checkpoint, resolved)


def test_legacy_checkpoint_has_no_metadata_conflict_check():
    resolved = resolve_experiment(
        ExperimentConfig.from_dict(_minimal_config()), {"window_size": 99}
    )
    validate_checkpoint_experiment_conflicts({"layer.weight": object()}, resolved)


def test_legacy_training_cli_remains_supported_and_overrides_defaults():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_common_cli_arguments(parser)
    add_training_cli_arguments(parser)
    cli = vars(
        parser.parse_args(
            ["-d", "UCIOccupancyDetection", "-epochs", "2", "-lr", "0.005"]
        )
    )
    resolved = resolve_experiment(None, cli)

    assert resolved.values["dataset"] == "UCIOccupancyDetection"
    assert resolved.values["epochs"] == 2
    assert resolved.values["learning_rate"] == 0.005
    assert resolved.values["window_size"] == 32


def test_legacy_synthesis_cli_remains_supported():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_common_cli_arguments(parser)
    add_synthesis_cli_arguments(parser)
    cli = vars(
        parser.parse_args(
            ["-d", "MetroTraffic", "-synth_mask", "M", "-stride", "8"]
        )
    )
    resolved = resolve_experiment(None, cli, require_synthesis_profile=True)

    assert resolved.values["dataset"] == "MetroTraffic"
    assert resolved.values["synthesis_profile"] == "M"
    assert resolved.values["synthesis_stride"] == 8


def test_occupancy_dry_run_reports_resolved_layout_and_windows():
    config = load_experiment_config("configs/experiments/uci_occupancy_smoke.json")
    resolved = resolve_experiment(config, {})
    args = resolved.namespace("training")

    report = build_dry_run_report(args, resolved, phase="training")

    assert report["rows"] == {"total": 8143, "train": 6129, "test": 2014}
    assert report["dimensions"] == {"in_dim": 5, "out_dim": 4}
    assert report["training"]["windows"] == 6122
    assert report["synthesis"]["rows_selected_by_mask"] == 2014
    assert report["synthesis"]["max_windows"] == 1


class _MetroDryRunPreprocessor:
    def __init__(self, dataset, proportional_cyclic_encoding):
        assert dataset == "MetroTraffic"
        assert proportional_cyclic_encoding is False
        self.dataset_config = load_dataset_config_by_id("MetroTraffic")
        self.signal_columns = list(self.dataset_config.signal_columns)
        self.metadata_columns = list(self.dataset_config.metadata_columns)
        self.hierarchical_features_uncyclic = list(self.metadata_columns)
        self.hierarchical_features_cyclic = [
            "{}_{}".format(column, component)
            for column in self.metadata_columns
            for component in ("sine", "cos")
        ]
        columns = self.signal_columns + self.hierarchical_features_cyclic
        self.df_cleaned = pd.DataFrame(
            np.zeros((48204, len(columns))), columns=columns
        )
        self.train_indices = list(range(40255))
        self.test_indices = list(range(40255, 48204))
        self.timestamps = None

    def cyclicDecode(self, frame):
        decoded = frame[self.signal_columns].copy()
        decoded["year"] = np.where(frame.index < 40255, 2017, 2018)
        decoded["month"] = 1
        decoded["day"] = 1
        decoded["hour"] = 0
        return decoded


def test_metrotraffic_dry_run_preserves_upstream_protocol(monkeypatch):
    monkeypatch.setattr(
        experiment_runtime, "Preprocessor", _MetroDryRunPreprocessor
    )
    config = load_experiment_config("configs/experiments/metrotraffic_upstream.json")
    resolved = resolve_experiment(config, {})
    args = resolved.namespace("training")

    report = build_dry_run_report(args, resolved, phase="training")

    assert report["rows"]["train"] == 40255
    assert report["rows"]["test"] == 7949
    assert report["dimensions"] == {"in_dim": 13, "out_dim": 5}
    assert report["training"]["windows"] == 40224
    assert report["synthesis"]["context_rows"] == 3
    assert report["synthesis"]["windows"] == 991
    assert report["synthesis"]["rows_selected_by_mask"] == 7949
