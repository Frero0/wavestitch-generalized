"""Evaluate isolated UCI Occupancy sampler ablations against existing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_uci_occupancy_full as evaluation


SIGNALS = ["Temperature", "Humidity", "Light", "CO2"]
METADATA = ["Occupancy"]
EXPECTED_COLUMNS = SIGNALS + METADATA
TRIAL_FILE = (
    "synth_wavestitch_pipeline_stride_30_trial_0"
    "_cycStd_grad_correction.csv"
)
METHOD_DEFAULT = "Default trial 0"
METHOD_DEFAULT_MEAN = "Default 5-trial mean"
METHOD_A = "Variant A — no gradient correction"
METHOD_B = "Variant B — sqrt posterior variance"


class SamplerEvaluationError(ValueError):
    """Raised when sampler-ablation artifacts are incompatible."""


def _load_trial(directory, real_occupancy):
    frame = evaluation._read_generated_csv(
        directory / TRIAL_FILE, EXPECTED_COLUMNS, evaluation.EXPECTED_ROWS
    )
    if not np.array_equal(
        frame["Occupancy"].to_numpy(), real_occupancy.to_numpy()
    ):
        raise SamplerEvaluationError(
            "{} does not preserve test Occupancy conditioning.".format(directory)
        )
    return frame


def _evaluate_one(preprocessor, checkpoint, real_physical, real_standardized, frame):
    prepared = {
        "checkpoint": checkpoint,
        "preprocessor": preprocessor,
        "signal_columns": SIGNALS,
        "metadata_columns": METADATA,
        "trial_paths": [],
        "synthetic_trials": [frame],
        "real_physical": real_physical,
        "real_standardized": real_standardized,
    }
    return evaluation.evaluate(prepared)


def _percentage_change(value, reference):
    if reference == 0:
        return np.nan
    return float(100 * (value / reference - 1))


def _metric_rows(method, results):
    frame = results["trial_metrics"].copy()
    frame.insert(0, "method", method)
    return frame


def _detail_rows(method, results, key):
    frame = results[key].copy()
    frame.insert(0, "method", method)
    return frame


def _default_references(default_results_dir):
    metrics = pd.read_csv(default_results_dir / "trial_metrics.csv")
    summary = pd.read_csv(default_results_dir / "metric_summary.csv")
    signals = pd.read_csv(default_results_dir / "signal_diagnostics.csv")
    constraints = pd.read_csv(default_results_dir / "constraint_diagnostics.csv")
    correlations = pd.read_csv(default_results_dir / "occupancy_correlations.csv")
    return metrics, summary, signals, constraints, correlations


def _build_headline(
    default_metrics,
    default_summary,
    default_signals,
    default_constraints,
    results_a,
    results_b,
):
    rows = []
    metric_columns = {
        "MSE standardized": "mse_standardized",
        "ACD": "acd",
        "xcorrD": "xcorrD",
    }
    for diagnostic, column in metric_columns.items():
        default_trial = float(default_metrics.loc[default_metrics["trial"] == 0, column].iloc[0])
        default_mean = float(default_summary.loc[default_summary["metric"] == column, "mean"].iloc[0])
        value_a = float(results_a["trial_metrics"][column].iloc[0])
        value_b = float(results_b["trial_metrics"][column].iloc[0])
        rows.append(
            {
                "diagnostic": diagnostic,
                "target": "lower is better",
                "real_value": np.nan,
                "default_trial_0": default_trial,
                "default_5trial_mean": default_mean,
                "variant_a": value_a,
                "variant_b": value_b,
                "variant_a_pct_vs_default_trial_0": _percentage_change(value_a, default_trial),
                "variant_b_pct_vs_default_trial_0": _percentage_change(value_b, default_trial),
                "variant_a_pct_vs_default_5mean": _percentage_change(value_a, default_mean),
                "variant_b_pct_vs_default_5mean": _percentage_change(value_b, default_mean),
            }
        )

    default_all = default_signals[default_signals["cohort"] == "all"]
    for signal in SIGNALS:
        reference = default_all[default_all["signal"] == signal]
        default_trial = float(reference.loc[reference["trial"] == 0, "synthetic_std"].iloc[0])
        default_mean = float(reference["synthetic_std"].mean())
        real_std = float(reference["real_std"].iloc[0])
        diag_a = results_a["signal_diagnostics"]
        diag_b = results_b["signal_diagnostics"]
        value_a = float(
            diag_a[(diag_a["cohort"] == "all") & (diag_a["signal"] == signal)]["synthetic_std"].iloc[0]
        )
        value_b = float(
            diag_b[(diag_b["cohort"] == "all") & (diag_b["signal"] == signal)]["synthetic_std"].iloc[0]
        )
        rows.append(
            {
                "diagnostic": "{} std".format(signal),
                "target": "closer to real std",
                "real_value": real_std,
                "default_trial_0": default_trial,
                "default_5trial_mean": default_mean,
                "variant_a": value_a,
                "variant_b": value_b,
                "variant_a_pct_vs_default_trial_0": _percentage_change(value_a, default_trial),
                "variant_b_pct_vs_default_trial_0": _percentage_change(value_b, default_trial),
                "variant_a_pct_vs_default_5mean": _percentage_change(value_a, default_mean),
                "variant_b_pct_vs_default_5mean": _percentage_change(value_b, default_mean),
            }
        )

    default_synthetic = default_constraints[
        default_constraints["series"] == "synthetic"
    ]
    constraint_names = {
        "Light below zero percent": "light_below_zero_percent",
        "Light exact zero percent": "light_exact_zero_percent",
    }
    real_targets = {
        "Light below zero percent": 0.0,
        "Light exact zero percent": 69.36444885799405,
    }
    for diagnostic, key in constraint_names.items():
        reference = default_synthetic[default_synthetic["diagnostic"] == key]
        default_trial = float(reference.loc[reference["trial"] == 0, "value"].iloc[0])
        default_mean = float(reference["value"].mean())
        value_a = float(
            results_a["constraint_diagnostics"].query(
                "series == 'synthetic' and diagnostic == @key"
            )["value"].iloc[0]
        )
        value_b = float(
            results_b["constraint_diagnostics"].query(
                "series == 'synthetic' and diagnostic == @key"
            )["value"].iloc[0]
        )
        rows.append(
            {
                "diagnostic": diagnostic,
                "target": "closer to real value",
                "real_value": real_targets[diagnostic],
                "default_trial_0": default_trial,
                "default_5trial_mean": default_mean,
                "variant_a": value_a,
                "variant_b": value_b,
                "variant_a_pct_vs_default_trial_0": _percentage_change(value_a, default_trial),
                "variant_b_pct_vs_default_trial_0": _percentage_change(value_b, default_trial),
                "variant_a_pct_vs_default_5mean": _percentage_change(value_a, default_mean),
                "variant_b_pct_vs_default_5mean": _percentage_change(value_b, default_mean),
            }
        )
    return pd.DataFrame(rows)


def run(args):
    if args.results_dir.exists():
        raise SamplerEvaluationError(
            "Refusing to overwrite existing results directory: {}".format(args.results_dir)
        )
    checkpoint = evaluation._load_checkpoint(args.checkpoint_path)
    preprocessor = evaluation.restore_evaluation_preprocessor(checkpoint)
    if checkpoint["model"]["signal_columns"] != SIGNALS:
        raise SamplerEvaluationError("Checkpoint signal layout is incompatible.")
    if checkpoint["model"]["metadata_columns"] != METADATA:
        raise SamplerEvaluationError("Checkpoint metadata layout is incompatible.")

    real_physical = (
        preprocessor.df_orig.loc[preprocessor.test_indices, EXPECTED_COLUMNS]
        .reset_index(drop=True)
        .apply(pd.to_numeric, errors="raise")
    )
    real_standardized = (
        preprocessor.df_cleaned.loc[preprocessor.test_indices, SIGNALS]
        .reset_index(drop=True)
        .astype(float)
    )
    real_occupancy = real_physical["Occupancy"]
    default_frame = _load_trial(args.default_generated_dir, real_occupancy)
    frame_a = _load_trial(args.variant_a_generated_dir, real_occupancy)
    frame_b = _load_trial(args.variant_b_generated_dir, real_occupancy)

    results_default = _evaluate_one(
        preprocessor, checkpoint, real_physical, real_standardized, default_frame
    )
    results_a = _evaluate_one(
        preprocessor, checkpoint, real_physical, real_standardized, frame_a
    )
    results_b = _evaluate_one(
        preprocessor, checkpoint, real_physical, real_standardized, frame_b
    )
    default_references = _default_references(args.default_results_dir)
    default_metrics, default_summary, default_signals, _, _ = default_references
    headline = _build_headline(
        default_metrics,
        default_summary,
        default_signals,
        default_references[3],
        results_a,
        results_b,
    )

    args.results_dir.mkdir(parents=True)
    pd.concat(
        [
            _metric_rows(METHOD_DEFAULT, results_default),
            _metric_rows(METHOD_A, results_a),
            _metric_rows(METHOD_B, results_b),
        ],
        ignore_index=True,
    ).to_csv(args.results_dir / "trial_metrics.csv", index=False)
    headline.to_csv(args.results_dir / "headline_comparison.csv", index=False)
    for filename, key in (
        ("signal_diagnostics.csv", "signal_diagnostics"),
        ("constraint_diagnostics.csv", "constraint_diagnostics"),
        ("occupancy_correlations.csv", "occupancy_correlations"),
    ):
        pd.concat(
            [
                _detail_rows(METHOD_DEFAULT, results_default, key),
                _detail_rows(METHOD_A, results_a, key),
                _detail_rows(METHOD_B, results_b, key),
            ],
            ignore_index=True,
        ).to_csv(args.results_dir / filename, index=False)

    payload = {
        "dataset": "UCIOccupancyDetection",
        "profile": "C",
        "checkpoint": str(args.checkpoint_path),
        "checkpoint_format_version": 2,
        "preprocessing_mode": "train_only",
        "preprocessing_refit": False,
        "trial": 0,
        "seed": 42,
        "default": "legacy posterior-variance amplitude plus gradient correction",
        "variant_a": "gradient correction disabled; legacy reverse noise unchanged",
        "variant_b": "sqrt posterior-variance amplitude; gradient correction unchanged",
        "headline_comparison": headline.to_dict(orient="records"),
    }
    (args.results_dir / "evaluation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    readable = [
        "# UCI Occupancy sampler ablations",
        "",
        "- Same checkpoint, dataset, preprocessing, architecture, conditioning and seed",
        "- Default trial 0 reused from the completed five-trial experiment",
        "- Variant A changes only gradient correction",
        "- Variant B changes only reverse-noise amplitude",
        "",
        headline.to_string(index=False),
    ]
    (args.results_dir / "summary.md").write_text("\n".join(readable) + "\n", encoding="utf-8")
    print(headline.to_string(index=False))
    print("Results: {}".format(args.results_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--default-generated-dir", type=Path, required=True)
    parser.add_argument("--variant-a-generated-dir", type=Path, required=True)
    parser.add_argument("--variant-b-generated-dir", type=Path, required=True)
    parser.add_argument("--default-results-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (
        SamplerEvaluationError,
        evaluation.EvaluationValidationError,
        FileNotFoundError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
