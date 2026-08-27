"""Typed experiment configuration and CLI/default precedence resolution."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Optional


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


class ExperimentCheckpointConflictError(ValueError):
    """Raised when explicit experiment values conflict with a checkpoint."""


MODEL_DEFAULTS = {
    "backbone": "S4",
    "hdim": 64,
    "layers": 4,
    "num_res_layers": 4,
    "res_channels": 64,
    "skip_channels": 64,
    "diff_step_embed_in": 32,
    "diff_step_embed_mid": 64,
    "diff_step_embed_out": 64,
    "s4_lmax": 100,
    "s4_dstate": 64,
    "s4_dropout": 0.0,
    "s4_bidirectional": True,
    "s4_layernorm": True,
    "propCycEnc": False,
}

DEFAULTS = {
    "window_size": 32,
    "training_stride": 1,
    "training_batch_size": 1024,
    "epochs": 1000,
    "learning_rate": 1e-4,
    "seed": 42,
    "max_steps": None,
    "timesteps": 200,
    "beta_0": 0.0001,
    "beta_T": 0.02,
    **MODEL_DEFAULTS,
    "checkpoint_path": None,
    "output_dir": "generated",
    "synthesis_stride": 1,
    "synthesis_profile": None,
    "n_trials": 5,
    "synthesis_batch_size": 1024,
    "max_windows": None,
}

MODEL_FIELDS = tuple(MODEL_DEFAULTS)
CHECKPOINT_BOUND_FIELDS = (
    "window_size",
    "timesteps",
    "beta_0",
    "beta_T",
    *MODEL_FIELDS,
)


def _reject_unknown(data, allowed, context):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ExperimentConfigError(
            "Unknown field(s) in {}: {}.".format(context, ", ".join(unknown))
        )


def _mapping(value, field_name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(
            "Field {!r} must be a JSON object.".format(field_name)
        )
    return value


def _positive_int(value, field_name, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentConfigError(
            "Field {!r} must be a positive integer.".format(field_name)
        )
    return value


def _non_negative_int(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentConfigError(
            "Field {!r} must be a non-negative integer.".format(field_name)
        )
    return value


def _positive_number(value, field_name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ExperimentConfigError(
            "Field {!r} must be a finite positive number.".format(field_name)
        )
    return float(value)


def _probability(value, field_name):
    value = _positive_number(value, field_name)
    if value >= 1:
        raise ExperimentConfigError(
            "Field {!r} must be strictly between 0 and 1.".format(field_name)
        )
    return value


def _optional_string(value, field_name):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(
            "Field {!r} must be a non-empty string.".format(field_name)
        )
    return value


def _optional_bool(value, field_name):
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ExperimentConfigError(
            "Field {!r} must be a boolean.".format(field_name)
        )
    return value


@dataclass(frozen=True)
class TrainingExperimentConfig:
    window_size: Optional[int] = None
    stride: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    seed: Optional[int] = None
    max_steps: Optional[int] = None


@dataclass(frozen=True)
class DiffusionExperimentConfig:
    timesteps: Optional[int] = None
    beta_0: Optional[float] = None
    beta_T: Optional[float] = None


@dataclass(frozen=True)
class ArchitectureExperimentConfig:
    backbone: Optional[str] = None
    hdim: Optional[int] = None
    layers: Optional[int] = None
    num_res_layers: Optional[int] = None
    res_channels: Optional[int] = None
    skip_channels: Optional[int] = None
    diff_step_embed_in: Optional[int] = None
    diff_step_embed_mid: Optional[int] = None
    diff_step_embed_out: Optional[int] = None
    s4_lmax: Optional[int] = None
    s4_dstate: Optional[int] = None
    s4_dropout: Optional[float] = None
    s4_bidirectional: Optional[bool] = None
    s4_layernorm: Optional[bool] = None
    proportional_cyclic_encoding: Optional[bool] = None


@dataclass(frozen=True)
class ArtifactExperimentConfig:
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass(frozen=True)
class SynthesisExperimentConfig:
    stride: Optional[int] = None
    profile: Optional[str] = None
    trials: Optional[int] = None
    batch_size: Optional[int] = None
    max_windows: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_id: str
    training: TrainingExperimentConfig = field(default_factory=TrainingExperimentConfig)
    diffusion: DiffusionExperimentConfig = field(default_factory=DiffusionExperimentConfig)
    architecture: ArchitectureExperimentConfig = field(
        default_factory=ArchitectureExperimentConfig
    )
    artifacts: ArtifactExperimentConfig = field(default_factory=ArtifactExperimentConfig)
    synthesis: SynthesisExperimentConfig = field(
        default_factory=SynthesisExperimentConfig
    )

    @classmethod
    def from_dict(cls, data: Any):
        if not isinstance(data, Mapping):
            raise ExperimentConfigError(
                "Experiment configuration root must be a JSON object."
            )
        allowed = {
            "dataset_id",
            "training",
            "diffusion",
            "architecture",
            "artifacts",
            "synthesis",
        }
        _reject_unknown(data, allowed, "experiment configuration")
        dataset_id = data.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ExperimentConfigError(
                "Missing required non-empty field 'dataset_id'."
            )

        training = _mapping(data.get("training"), "training")
        training_fields = {
            "window_size", "stride", "batch_size", "epochs",
            "learning_rate", "seed", "max_steps",
        }
        _reject_unknown(training, training_fields, "training")
        training_config = TrainingExperimentConfig(
            window_size=_positive_int(training.get("window_size"), "training.window_size", optional=True),
            stride=_positive_int(training.get("stride"), "training.stride", optional=True),
            batch_size=_positive_int(training.get("batch_size"), "training.batch_size", optional=True),
            epochs=_positive_int(training.get("epochs"), "training.epochs", optional=True),
            learning_rate=(
                _positive_number(training["learning_rate"], "training.learning_rate")
                if "learning_rate" in training else None
            ),
            seed=(
                _non_negative_int(training["seed"], "training.seed")
                if "seed" in training else None
            ),
            max_steps=_positive_int(training.get("max_steps"), "training.max_steps", optional=True),
        )

        diffusion = _mapping(data.get("diffusion"), "diffusion")
        _reject_unknown(diffusion, {"timesteps", "beta_0", "beta_T"}, "diffusion")
        diffusion_config = DiffusionExperimentConfig(
            timesteps=_positive_int(diffusion.get("timesteps"), "diffusion.timesteps", optional=True),
            beta_0=_probability(diffusion["beta_0"], "diffusion.beta_0") if "beta_0" in diffusion else None,
            beta_T=_probability(diffusion["beta_T"], "diffusion.beta_T") if "beta_T" in diffusion else None,
        )

        architecture = _mapping(data.get("architecture"), "architecture")
        architecture_fields = {
            "backbone", "hdim", "layers", "num_res_layers", "res_channels",
            "skip_channels", "diff_step_embed_in", "diff_step_embed_mid",
            "diff_step_embed_out", "s4_lmax", "s4_dstate", "s4_dropout",
            "s4_bidirectional", "s4_layernorm", "proportional_cyclic_encoding",
        }
        _reject_unknown(architecture, architecture_fields, "architecture")
        positive_architecture_fields = architecture_fields - {
            "backbone", "s4_dropout", "s4_bidirectional", "s4_layernorm",
            "proportional_cyclic_encoding",
        }
        validated_architecture = {
            name: _positive_int(architecture.get(name), "architecture.{}".format(name), optional=True)
            for name in positive_architecture_fields
        }
        backbone = _optional_string(architecture.get("backbone"), "architecture.backbone")
        dropout = architecture.get("s4_dropout")
        if dropout is not None and (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ExperimentConfigError(
                "Field 'architecture.s4_dropout' must be in [0, 1)."
            )
        architecture_config = ArchitectureExperimentConfig(
            backbone=backbone,
            s4_dropout=float(dropout) if dropout is not None else None,
            s4_bidirectional=_optional_bool(architecture.get("s4_bidirectional"), "architecture.s4_bidirectional"),
            s4_layernorm=_optional_bool(architecture.get("s4_layernorm"), "architecture.s4_layernorm"),
            proportional_cyclic_encoding=_optional_bool(
                architecture.get("proportional_cyclic_encoding"),
                "architecture.proportional_cyclic_encoding",
            ),
            **validated_architecture,
        )

        artifacts = _mapping(data.get("artifacts"), "artifacts")
        _reject_unknown(artifacts, {"checkpoint_path", "output_dir"}, "artifacts")
        artifact_config = ArtifactExperimentConfig(
            checkpoint_path=_optional_string(artifacts.get("checkpoint_path"), "artifacts.checkpoint_path"),
            output_dir=_optional_string(artifacts.get("output_dir"), "artifacts.output_dir"),
        )

        synthesis = _mapping(data.get("synthesis"), "synthesis")
        _reject_unknown(
            synthesis,
            {"stride", "profile", "trials", "batch_size", "max_windows"},
            "synthesis",
        )
        profile = synthesis.get("profile")
        if profile is not None and profile not in {"C", "M", "F"}:
            raise ExperimentConfigError(
                "Field 'synthesis.profile' must be one of C, M, F."
            )
        synthesis_config = SynthesisExperimentConfig(
            stride=_positive_int(synthesis.get("stride"), "synthesis.stride", optional=True),
            profile=profile,
            trials=_positive_int(synthesis.get("trials"), "synthesis.trials", optional=True),
            batch_size=_positive_int(synthesis.get("batch_size"), "synthesis.batch_size", optional=True),
            max_windows=_positive_int(synthesis.get("max_windows"), "synthesis.max_windows", optional=True),
        )

        return cls(
            dataset_id=dataset_id,
            training=training_config,
            diffusion=diffusion_config,
            architecture=architecture_config,
            artifacts=artifact_config,
            synthesis=synthesis_config,
        )

    def flat_values(self):
        architecture = vars(self.architecture).copy()
        architecture["propCycEnc"] = architecture.pop(
            "proportional_cyclic_encoding"
        )
        return {
            "dataset": self.dataset_id,
            "window_size": self.training.window_size,
            "training_stride": self.training.stride,
            "training_batch_size": self.training.batch_size,
            "epochs": self.training.epochs,
            "learning_rate": self.training.learning_rate,
            "seed": self.training.seed,
            "max_steps": self.training.max_steps,
            "timesteps": self.diffusion.timesteps,
            "beta_0": self.diffusion.beta_0,
            "beta_T": self.diffusion.beta_T,
            **architecture,
            "checkpoint_path": self.artifacts.checkpoint_path,
            "output_dir": self.artifacts.output_dir,
            "synthesis_stride": self.synthesis.stride,
            "synthesis_profile": self.synthesis.profile,
            "n_trials": self.synthesis.trials,
            "synthesis_batch_size": self.synthesis.batch_size,
            "max_windows": self.synthesis.max_windows,
        }


@dataclass(frozen=True)
class ResolvedExperiment:
    values: Mapping[str, Any]
    sources: Mapping[str, str]

    def namespace(self, phase):
        values = dict(self.values)
        values["dataset"] = values["dataset"]
        values["lr"] = values["learning_rate"]
        values["synth_mask"] = values["synthesis_profile"]
        if phase == "training":
            values["stride"] = values["training_stride"]
            values["batch_size"] = values["training_batch_size"]
        elif phase == "synthesis":
            values["stride"] = values["synthesis_stride"]
            values["batch_size"] = values["synthesis_batch_size"]
        else:
            raise ExperimentConfigError("Unknown experiment phase {!r}.".format(phase))
        return SimpleNamespace(**values)


def load_experiment_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(
            "Experiment configuration does not exist: {}".format(config_path)
        )
    if config_path.suffix.lower() != ".json":
        raise ExperimentConfigError("Experiment configuration must be a .json file.")
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ExperimentConfigError(
            "Invalid JSON in experiment configuration at line {}, column {}: {}.".format(
                exc.lineno, exc.colno, exc.msg
            )
        ) from exc
    return ExperimentConfig.from_dict(data)


def _validate_resolved(values, *, require_synthesis_profile, allow_missing_dataset):
    if not allow_missing_dataset and not values.get("dataset"):
        raise ExperimentConfigError(
            "dataset ID is required via CLI or experiment configuration."
        )
    for field_name in (
        "window_size", "training_stride", "training_batch_size", "epochs",
        "timesteps", "hdim", "layers", "num_res_layers", "res_channels",
        "skip_channels", "diff_step_embed_in", "diff_step_embed_mid",
        "diff_step_embed_out", "s4_lmax", "s4_dstate", "synthesis_stride",
        "n_trials", "synthesis_batch_size",
    ):
        _positive_int(values[field_name], field_name)
    _positive_number(values["learning_rate"], "learning_rate")
    _non_negative_int(values["seed"], "seed")
    if values["max_steps"] is not None:
        _positive_int(values["max_steps"], "max_steps")
    if values["max_windows"] is not None:
        _positive_int(values["max_windows"], "max_windows")
    beta_0 = _probability(values["beta_0"], "beta_0")
    beta_T = _probability(values["beta_T"], "beta_T")
    if beta_0 >= beta_T:
        raise ExperimentConfigError("beta_0 must be strictly smaller than beta_T.")
    if not isinstance(values["backbone"], str) or not values["backbone"].strip():
        raise ExperimentConfigError("backbone must be a non-empty string.")
    dropout = values["s4_dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise ExperimentConfigError("s4_dropout must be in [0, 1).")
    for field_name in ("s4_bidirectional", "s4_layernorm", "propCycEnc"):
        if not isinstance(values[field_name], bool):
            raise ExperimentConfigError("{} must be a boolean.".format(field_name))
    if (
        values["synthesis_profile"] is not None
        and values["synthesis_profile"] not in {"C", "M", "F"}
    ):
        raise ExperimentConfigError("synthesis profile must be one of C, M, F.")
    if require_synthesis_profile and values["synthesis_profile"] not in {"C", "M", "F"}:
        raise ExperimentConfigError(
            "synthesis profile is required and must be one of C, M, F."
        )


def resolve_experiment(
    config: Optional[ExperimentConfig],
    cli_values: Mapping[str, Any],
    *,
    require_synthesis_profile=False,
    allow_missing_dataset=False,
    inferred_dataset=None,
):
    values = dict(DEFAULTS)
    values["dataset"] = None
    sources = {key: "default" for key in values}

    if config is not None:
        for key, value in config.flat_values().items():
            if value is not None:
                values[key] = value
                sources[key] = "experiment config"

    for key, value in cli_values.items():
        if key not in values:
            continue
        values[key] = value
        sources[key] = "CLI override"

    if values["dataset"] is None and inferred_dataset is not None:
        values["dataset"] = inferred_dataset
        sources["dataset"] = "structured checkpoint"

    if values["checkpoint_path"] is None and values["dataset"]:
        filename = "model_prop.pth" if values["propCycEnc"] else "model.pth"
        values["checkpoint_path"] = str(
            Path("saved_models") / values["dataset"] / filename
        )
        sources["checkpoint_path"] = "default"

    _validate_resolved(
        values,
        require_synthesis_profile=require_synthesis_profile,
        allow_missing_dataset=allow_missing_dataset,
    )
    return ResolvedExperiment(
        values=MappingProxyType(values),
        sources=MappingProxyType(sources),
    )


def validate_checkpoint_experiment_conflicts(checkpoint, resolved):
    """Reject explicit config/CLI model values that disagree with checkpoint metadata."""

    from checkpoint_utils import is_structured_checkpoint

    if not is_structured_checkpoint(checkpoint):
        return
    model = checkpoint["model"]
    architecture = model["architecture"]
    expected = {
        "dataset": checkpoint["dataset_id"],
        "window_size": checkpoint["training"]["window_size"],
        "timesteps": checkpoint["diffusion"]["timesteps"],
        "beta_0": checkpoint["diffusion"]["beta_0"],
        "beta_T": checkpoint["diffusion"]["beta_T"],
        "propCycEnc": checkpoint["encoding"]["proportional_cyclic_encoding"],
        **{field: architecture[field] for field in MODEL_FIELDS if field != "propCycEnc"},
    }
    conflicts = []
    for field_name, checkpoint_value in expected.items():
        if resolved.sources.get(field_name) == "default":
            continue
        configured_value = resolved.values.get(field_name)
        if configured_value != checkpoint_value:
            conflicts.append(
                "{}={} ({}) vs checkpoint={}".format(
                    field_name,
                    configured_value,
                    resolved.sources.get(field_name),
                    checkpoint_value,
                )
            )
    if conflicts:
        raise ExperimentCheckpointConflictError(
            "Experiment configuration conflicts with structured checkpoint: {}.".format(
                "; ".join(conflicts)
            )
        )


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def add_common_cli_arguments(parser):
    parser.add_argument('-experiment_config', '--experiment-config', dest='experiment_config')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-dataset', '--dataset', '-d', dest='dataset')
    parser.add_argument('-backbone', '--backbone')
    parser.add_argument('-beta_0', '--beta-0', dest='beta_0', type=float)
    parser.add_argument('-beta_T', '--beta-T', dest='beta_T', type=float)
    parser.add_argument('-timesteps', '--timesteps', '-T', type=int)
    parser.add_argument('-hdim', '--hidden-dim', dest='hdim', type=int)
    parser.add_argument('-layers', '--layers', type=int)
    parser.add_argument('-window_size', '--window-size', dest='window_size', type=int)
    parser.add_argument('-num_res_layers', '--num-res-layers', dest='num_res_layers', type=int)
    parser.add_argument('-res_channels', '--res-channels', dest='res_channels', type=int)
    parser.add_argument('-skip_channels', '--skip-channels', dest='skip_channels', type=int)
    parser.add_argument('-diff_step_embed_in', '--diff-step-embed-in', dest='diff_step_embed_in', type=int)
    parser.add_argument('-diff_step_embed_mid', '--diff-step-embed-mid', dest='diff_step_embed_mid', type=int)
    parser.add_argument('-diff_step_embed_out', '--diff-step-embed-out', dest='diff_step_embed_out', type=int)
    parser.add_argument('-s4_lmax', '--s4-lmax', dest='s4_lmax', type=int)
    parser.add_argument('-s4_dstate', '--s4-dstate', dest='s4_dstate', type=int)
    parser.add_argument('-s4_dropout', '--s4-dropout', dest='s4_dropout', type=float)
    parser.add_argument('-s4_bidirectional', '--s4-bidirectional', dest='s4_bidirectional', type=parse_bool)
    parser.add_argument('-s4_layernorm', '--s4-layernorm', dest='s4_layernorm', type=parse_bool)
    parser.add_argument('-propCycEnc', '--proportional-cyclic-encoding', dest='propCycEnc', type=parse_bool)
    parser.add_argument('-checkpoint_path', '--checkpoint-path', dest='checkpoint_path')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float)
    parser.add_argument('-seed', '--seed', type=int)


def add_training_cli_arguments(parser):
    parser.add_argument('-batch_size', '--batch-size', dest='training_batch_size', type=int)
    parser.add_argument('-epochs', '--epochs', type=int)
    parser.add_argument('-stride', '--training-stride', dest='training_stride', type=int)
    parser.add_argument('-max_steps', '--max-steps', dest='max_steps', type=int)


def add_synthesis_cli_arguments(parser):
    parser.add_argument('-batch_size', '--batch-size', dest='synthesis_batch_size', type=int)
    parser.add_argument('-stride', '--synthesis-stride', dest='synthesis_stride', type=int)
    parser.add_argument('-synth_mask', '--synthesis-profile', dest='synthesis_profile')
    parser.add_argument('-n_trials', '--trials', dest='n_trials', type=int)
    parser.add_argument('-output_dir', '--output-dir', dest='output_dir')
    parser.add_argument('-max_windows', '--max-windows', dest='max_windows', type=int)
