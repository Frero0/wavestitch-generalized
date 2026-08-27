"""Shared dry-run inspection for WaveStitch experiment entry points."""

from __future__ import annotations

import json

import torch

from wavestitch.checkpoint_utils import (
    checkpoint_preprocessing_state,
    is_structured_checkpoint,
    validate_structured_checkpoint,
)
from wavestitch.data_utils import Preprocessor
from wavestitch.metasynth import metadataMask
from wavestitch.training_utils import resolve_model_columns
from wavestitch.window_validation import (
    validate_synthesis_context_mask,
    validate_synthesis_windowing,
    validate_training_windowing,
)


def build_dry_run_report(args, resolved, *, phase, checkpoint=None):
    """Load and validate an experiment without constructing or running a model."""

    preprocessing_state = (
        checkpoint_preprocessing_state(checkpoint)
        if is_structured_checkpoint(checkpoint)
        else None
    )
    if preprocessing_state is None:
        preprocessor = Preprocessor(args.dataset, args.propCycEnc)
    else:
        preprocessor = Preprocessor(
            args.dataset,
            args.propCycEnc,
            preprocessing_state=preprocessing_state,
        )
    frame = preprocessor.df_cleaned
    training_plan = validate_training_windowing(
        preprocessor,
        window_size=args.window_size,
        requested_training_stride=resolved.values["training_stride"],
        effective_training_stride=1,
    )
    training_frame = frame.loc[list(training_plan.train_indices)]
    signal_indices, metadata_indices = resolve_model_columns(
        training_frame, preprocessor
    )

    synthesis = None
    profile = resolved.values["synthesis_profile"]
    if profile is not None:
        synthesis_plan = validate_synthesis_windowing(
            preprocessor,
            window_size=args.window_size,
            synthesis_stride=resolved.values["synthesis_stride"],
        )
        candidate_frame = frame.loc[list(synthesis_plan.candidate_indices)]
        decoded = preprocessor.cyclicDecode(candidate_frame)
        metadata = decoded[preprocessor.hierarchical_features_uncyclic]
        mask = metadataMask(
            metadata,
            profile,
            args.dataset,
            dataset_config=preprocessor.dataset_config,
            test_indices=preprocessor.test_indices,
        )
        validate_synthesis_context_mask(mask, synthesis_plan)
        candidate_signal_indices, candidate_metadata_indices = resolve_model_columns(
            candidate_frame, preprocessor
        )
        if is_structured_checkpoint(checkpoint):
            validate_structured_checkpoint(
                checkpoint,
                dataset_id=args.dataset,
                frame=candidate_frame,
                preprocessor=preprocessor,
                signal_indices=candidate_signal_indices,
                metadata_indices=candidate_metadata_indices,
            )
        synthesis = {
            "profile": profile,
            "stride": synthesis_plan.synthesis_stride,
            "context_rows": synthesis_plan.context_count,
            "candidate_rows": len(synthesis_plan.candidate_indices),
            "windows": synthesis_plan.synthesis_window_count,
            "rows_selected_by_mask": int(mask.sum()),
            "batch_size": resolved.values["synthesis_batch_size"],
            "trials": resolved.values["n_trials"],
            "max_windows": resolved.values["max_windows"],
        }

    report = {
        "mode": "dry-run",
        "phase": phase,
        "dataset": args.dataset,
        "rows": {
            "total": len(frame),
            "train": len(training_plan.train_indices),
            "test": len(training_plan.test_indices),
        },
        "columns": {
            "signal": list(preprocessor.signal_columns),
            "metadata": list(preprocessor.metadata_columns),
            "encoded_metadata": training_frame.columns[metadata_indices].tolist(),
            "signal_indices": signal_indices.tolist(),
            "metadata_indices": metadata_indices.tolist(),
        },
        "dimensions": {
            "in_dim": len(training_frame.columns),
            "out_dim": len(signal_indices),
        },
        "training": {
            "window_size": training_plan.window_size,
            "requested_stride": resolved.values["training_stride"],
            "effective_stride": training_plan.training_stride,
            "windows": training_plan.training_window_count,
            "batch_size": resolved.values["training_batch_size"],
            "epochs": resolved.values["epochs"],
            "max_steps": resolved.values["max_steps"],
            "optimizer": "Adam",
            "learning_rate": resolved.values["learning_rate"],
            "seed": resolved.values["seed"],
        },
        "diffusion": {
            "timesteps": args.timesteps,
            "beta_0": args.beta_0,
            "beta_T": args.beta_T,
        },
        "architecture": {
            field: getattr(args, field)
            for field in (
                "backbone", "hdim", "layers", "num_res_layers",
                "res_channels", "skip_channels", "diff_step_embed_in",
                "diff_step_embed_mid", "diff_step_embed_out", "s4_lmax",
                "s4_dstate", "s4_dropout", "s4_bidirectional",
                "s4_layernorm", "propCycEnc",
            )
        },
        "synthesis": synthesis,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "preprocessing": {
            "mode": (
                preprocessor.dataset_config.preprocessing_mode
                if preprocessor.dataset_config is not None
                else preprocessor.preprocessing_mode
            ),
            "state_source": (
                "structured checkpoint"
                if preprocessing_state is not None
                else "dataset fit"
            ),
        },
        "artifacts": {
            "checkpoint": resolved.values["checkpoint_path"],
            "checkpoint_format_version": (
                checkpoint.get("format_version")
                if is_structured_checkpoint(checkpoint)
                else None
            ),
            "output_directory": resolved.values["output_dir"],
        },
        "sources": dict(resolved.sources),
    }
    return report


def print_dry_run_report(report):
    print(json.dumps(report, indent=2, sort_keys=True))
