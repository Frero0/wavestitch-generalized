"""Final UCI Occupancy real-vs-synthetic evaluation for WaveStitch profile C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpoint_utils import (
    CHECKPOINT_VERSION,
    checkpoint_preprocessing_state,
    is_structured_checkpoint,
    validate_structured_checkpoint,
)
from data_utils import Preprocessor

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except ImportError:  # Diagnostics are optional; core WaveStitch metrics still work.
    ks_2samp = None
    wasserstein_distance = None


DATASET_ID = "UCIOccupancyDetection"
PROFILE = "C"
EXPECTED_ROWS = 2014
EXPECTED_TRIALS = 5
ACD_LAGS = 100
TRIAL_BASENAME = (
    "synth_wavestitch_pipeline_stride_30_trial_{}"
    "_cycStd_grad_correction.csv"
)
INDEX_COLUMN = "Unnamed: 0"
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class EvaluationValidationError(ValueError):
    """Raised when final-experiment artifacts do not match the approved protocol."""


def _load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not is_structured_checkpoint(checkpoint):
        raise EvaluationValidationError(
            "Final evaluation requires a structured checkpoint v2, not a raw checkpoint."
        )
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise EvaluationValidationError(
            "Final evaluation requires structured checkpoint v2; found version {!r}.".format(
                checkpoint.get("format_version")
            )
        )
    if checkpoint.get("dataset_id") != DATASET_ID:
        raise EvaluationValidationError(
            "Checkpoint dataset {!r} is not {!r}.".format(
                checkpoint.get("dataset_id"), DATASET_ID
            )
        )
    return checkpoint


def restore_evaluation_preprocessor(checkpoint):
    """Restore preprocessing from v2 state; never fit on evaluation data."""

    preprocessing_state = checkpoint_preprocessing_state(checkpoint)
    if preprocessing_state.get("mode") != "train_only":
        raise EvaluationValidationError(
            "Final UCI Occupancy evaluation requires preprocessing mode 'train_only'."
        )
    proportional = checkpoint["encoding"]["proportional_cyclic_encoding"]
    preprocessor = Preprocessor(
        checkpoint["dataset_id"],
        proportional,
        preprocessing_state=preprocessing_state,
    )
    if preprocessor.preprocessing_state_dict() != preprocessing_state:
        raise EvaluationValidationError(
            "Restored preprocessing state differs from the checkpoint state."
        )
    return preprocessor


def _expected_trial_paths(input_dir: Path, trials: int):
    return [input_dir / TRIAL_BASENAME.format(trial) for trial in range(trials)]


def _validate_trial_file_set(input_dir: Path, trials: int):
    expected = _expected_trial_paths(input_dir, trials)
    actual = sorted(
        input_dir.glob(
            "synth_wavestitch_pipeline_stride_30_trial_*"
            "_cycStd_grad_correction.csv"
        )
    )
    if set(actual) != set(expected):
        missing = sorted(path.name for path in set(expected) - set(actual))
        unexpected = sorted(path.name for path in set(actual) - set(expected))
        raise EvaluationValidationError(
            "Trial file set is not exactly trials 0-{}; missing={}, unexpected={}.".format(
                trials - 1, missing, unexpected
            )
        )
    return expected


def _read_generated_csv(path: Path, expected_columns, expected_rows: int):
    frame = pd.read_csv(path)
    if frame.columns.tolist() == [INDEX_COLUMN, *expected_columns]:
        expected_index = np.arange(expected_rows)
        if not np.array_equal(frame[INDEX_COLUMN].to_numpy(), expected_index):
            raise EvaluationValidationError(
                "{} has a non-canonical saved row index.".format(path.name)
            )
        frame = frame.drop(columns=[INDEX_COLUMN])
    elif frame.columns.tolist() != list(expected_columns):
        raise EvaluationValidationError(
            "{} columns are {}; expected {} (plus optional {}).".format(
                path.name, frame.columns.tolist(), list(expected_columns), INDEX_COLUMN
            )
        )
    if len(frame) != expected_rows:
        raise EvaluationValidationError(
            "{} has {} rows; expected {}.".format(path.name, len(frame), expected_rows)
        )
    if frame.isna().any().any():
        raise EvaluationValidationError("{} contains missing values.".format(path.name))
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise EvaluationValidationError(
            "{} contains non-numeric or non-finite values.".format(path.name)
        )
    return numeric


def prepare_evaluation(checkpoint_path: Path, input_dir: Path, trials: int):
    checkpoint = _load_checkpoint(checkpoint_path)
    preprocessor = restore_evaluation_preprocessor(checkpoint)
    preprocessing = checkpoint_preprocessing_state(checkpoint)
    signal_columns = list(checkpoint["model"]["signal_columns"])
    metadata_columns = list(checkpoint["model"]["metadata_columns"])
    expected_signals = ["Temperature", "Humidity", "Light", "CO2"]
    if signal_columns != expected_signals or metadata_columns != ["Occupancy"]:
        raise EvaluationValidationError(
            "Checkpoint model columns are incompatible: signal={}, metadata={}.".format(
                signal_columns, metadata_columns
            )
        )
    expected_columns = signal_columns + metadata_columns
    if preprocessing["scaler"]["columns"] != signal_columns:
        raise EvaluationValidationError(
            "Checkpoint scaler columns differ from checkpoint signal columns."
        )
    if int(preprocessing["scaler"]["n_samples_seen"]) != 6129:
        raise EvaluationValidationError(
            "Checkpoint scaler was not fitted on exactly 6129 train rows."
        )
    if len(preprocessor.test_indices) != EXPECTED_ROWS:
        raise EvaluationValidationError(
            "Restored test split has {} rows; expected {}.".format(
                len(preprocessor.test_indices), EXPECTED_ROWS
            )
        )

    processed_test = preprocessor.df_cleaned.loc[preprocessor.test_indices]
    signal_indices = processed_test.columns.get_indexer(signal_columns)
    encoded_metadata = list(preprocessor.hierarchical_features_cyclic)
    metadata_indices = processed_test.columns.get_indexer(encoded_metadata)
    if (signal_indices < 0).any() or (metadata_indices < 0).any():
        raise EvaluationValidationError(
            "Checkpoint-declared model columns are absent from the restored test frame."
        )
    validate_structured_checkpoint(
        checkpoint,
        dataset_id=DATASET_ID,
        frame=processed_test,
        preprocessor=preprocessor,
        signal_indices=signal_indices,
        metadata_indices=metadata_indices,
    )
    if processed_test.columns[signal_indices].tolist() != signal_columns:
        raise EvaluationValidationError("Resolved signal layout is incompatible.")
    if processed_test.columns[metadata_indices].tolist() != metadata_columns:
        raise EvaluationValidationError("Resolved metadata layout is incompatible.")

    trial_paths = _validate_trial_file_set(input_dir, trials)
    synthetic_trials = [
        _read_generated_csv(path, expected_columns, EXPECTED_ROWS)
        for path in trial_paths
    ]

    real_physical = (
        preprocessor.df_orig.loc[preprocessor.test_indices, expected_columns]
        .reset_index(drop=True)
        .apply(pd.to_numeric, errors="raise")
    )
    real_standardized = (
        processed_test[signal_columns].reset_index(drop=True).astype(float)
    )
    for trial, synthetic in enumerate(synthetic_trials):
        if not np.array_equal(
            synthetic["Occupancy"].to_numpy(),
            real_physical["Occupancy"].to_numpy(),
        ):
            raise EvaluationValidationError(
                "Trial {} does not preserve row-aligned Occupancy conditioning.".format(
                    trial
                )
            )

    real_artifact = _read_generated_csv(
        input_dir / "real.csv", expected_columns, EXPECTED_ROWS
    )
    if not np.allclose(
        real_artifact[signal_columns].to_numpy(dtype=float),
        real_physical[signal_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-4,
    ) or not np.array_equal(
        real_artifact["Occupancy"].to_numpy(),
        real_physical["Occupancy"].to_numpy(),
    ):
        raise EvaluationValidationError(
            "real.csv does not match the checkpoint-restored test split."
        )

    return {
        "checkpoint": checkpoint,
        "preprocessor": preprocessor,
        "signal_columns": signal_columns,
        "metadata_columns": metadata_columns,
        "trial_paths": trial_paths,
        "synthetic_trials": synthetic_trials,
        "real_physical": real_physical,
        "real_standardized": real_standardized,
    }


def standardized_mse(real, synthetic):
    """Upstream elementwise MSE, averaged across rows then signals."""

    return float(((synthetic - real) ** 2).mean().mean())


def _autocorrelation_matrix(frame, lags=ACD_LAGS):
    values = frame.to_numpy(dtype=float)
    stds = np.std(values, axis=0)
    constants = stds == 0
    safe_stds = stds.copy()
    safe_stds[constants] = 1.0
    centered = (values - np.mean(values, axis=0)) / safe_stds
    autocorrelation = np.ones((values.shape[1], lags))
    for lag in range(1, lags):
        autocorrelation[:, lag] = np.mean(
            centered[lag:, :] * centered[:-lag, :], axis=0
        )
    return autocorrelation, constants


def acd(real, synthetic, lags=ACD_LAGS):
    """Upstream ACD: mean absolute autocorrelation difference over signals/lags."""

    real_ac, real_constant = _autocorrelation_matrix(real, lags)
    synth_ac, synth_constant = _autocorrelation_matrix(synthetic, lags)
    usable = ~(real_constant | synth_constant)
    if not usable.any():
        raise EvaluationValidationError("ACD is undefined: all signals are constant.")
    return float(np.mean(np.abs(real_ac - synth_ac)[usable, :]))


def xcorr_distance(real, synthetic):
    """Upstream xcorrD: mean absolute difference of full correlation matrices."""

    difference = (real.corr() - synthetic.corr()).abs()
    if difference.isna().any().any():
        raise EvaluationValidationError(
            "xcorrD is undefined because at least one signal is constant."
        )
    return float(difference.mean().mean())


def _describe(values):
    array = np.asarray(values, dtype=float)
    result = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for quantile in QUANTILES:
        result["q{:02d}".format(round(quantile * 100))] = float(
            np.quantile(array, quantile)
        )
    return result


def _signal_diagnostic_row(trial, cohort, signal, real, synthetic):
    row = {"trial": trial, "cohort": cohort, "signal": signal}
    row.update({"real_" + key: value for key, value in _describe(real).items()})
    row.update(
        {"synthetic_" + key: value for key, value in _describe(synthetic).items()}
    )
    if wasserstein_distance is None or ks_2samp is None:
        row.update(
            wasserstein_1=np.nan,
            ks_statistic=np.nan,
            ks_pvalue=np.nan,
        )
    else:
        ks_result = ks_2samp(real, synthetic)
        row.update(
            wasserstein_1=float(wasserstein_distance(real, synthetic)),
            ks_statistic=float(ks_result.statistic),
            ks_pvalue=float(ks_result.pvalue),
        )
    return row


def evaluate(prepared):
    preprocessor = prepared["preprocessor"]
    signals = prepared["signal_columns"]
    real_physical = prepared["real_physical"]
    real_standardized = prepared["real_standardized"]
    trial_metric_rows = []
    signal_rows = []
    constraint_rows = []
    correlation_rows = []

    for trial, synthetic_physical in enumerate(prepared["synthetic_trials"]):
        synthetic_standardized = preprocessor.scale(
            synthetic_physical[signals]
        )[signals].astype(float)
        trial_metric_rows.append(
            {
                "trial": trial,
                "mse_standardized": standardized_mse(
                    real_standardized, synthetic_standardized
                ),
                "acd": acd(real_physical[signals], synthetic_physical[signals]),
                "xcorrD": xcorr_distance(
                    real_physical[signals], synthetic_physical[signals]
                ),
            }
        )

        cohorts = [("all", np.ones(len(real_physical), dtype=bool))]
        cohorts.extend(
            ("Occupancy={}".format(value), real_physical["Occupancy"].eq(value))
            for value in (0, 1)
        )
        for cohort_name, mask in cohorts:
            if not np.any(mask):
                raise EvaluationValidationError(
                    "Cohort {} is empty.".format(cohort_name)
                )
            for signal in signals:
                signal_rows.append(
                    _signal_diagnostic_row(
                        trial,
                        cohort_name,
                        signal,
                        real_physical.loc[mask, signal].to_numpy(dtype=float),
                        synthetic_physical.loc[mask, signal].to_numpy(dtype=float),
                    )
                )

        for series_name, frame in (
            ("real", real_physical),
            ("synthetic", synthetic_physical),
        ):
            constraint_rows.extend(
                [
                    {
                        "trial": trial,
                        "series": series_name,
                        "diagnostic": "light_below_zero_percent",
                        "value": float(100 * frame["Light"].lt(0).mean()),
                    },
                    {
                        "trial": trial,
                        "series": series_name,
                        "diagnostic": "co2_below_zero_percent",
                        "value": float(100 * frame["CO2"].lt(0).mean()),
                    },
                    {
                        "trial": trial,
                        "series": series_name,
                        "diagnostic": "light_exact_zero_percent",
                        "value": float(100 * frame["Light"].eq(0).mean()),
                    },
                ]
            )

        for signal in signals:
            real_corr = float(real_physical["Occupancy"].corr(real_physical[signal]))
            synthetic_corr = float(
                synthetic_physical["Occupancy"].corr(synthetic_physical[signal])
            )
            correlation_rows.append(
                {
                    "trial": trial,
                    "signal": signal,
                    "real_occupancy_correlation": real_corr,
                    "synthetic_occupancy_correlation": synthetic_corr,
                    "absolute_difference": abs(real_corr - synthetic_corr),
                }
            )

    trial_metrics = pd.DataFrame(trial_metric_rows)
    summary_rows = []
    for metric in ("mse_standardized", "acd", "xcorrD"):
        values = trial_metrics[metric].to_numpy()
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "trials": len(values),
            }
        )
    return {
        "trial_metrics": trial_metrics,
        "metric_summary": pd.DataFrame(summary_rows),
        "signal_diagnostics": pd.DataFrame(signal_rows),
        "constraint_diagnostics": pd.DataFrame(constraint_rows),
        "occupancy_correlations": pd.DataFrame(correlation_rows),
    }


def _write_results(output_dir: Path, prepared, results, checkpoint_path, input_dir):
    if output_dir.exists():
        raise EvaluationValidationError(
            "Refusing to overwrite existing result directory: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True)
    csv_files = {
        "trial_metrics.csv": results["trial_metrics"],
        "metric_summary.csv": results["metric_summary"],
        "signal_diagnostics.csv": results["signal_diagnostics"],
        "constraint_diagnostics.csv": results["constraint_diagnostics"],
        "occupancy_correlations.csv": results["occupancy_correlations"],
    }
    for name, frame in csv_files.items():
        frame.to_csv(output_dir / name, index=False)

    summary_payload = {
        "dataset": DATASET_ID,
        "profile": PROFILE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": CHECKPOINT_VERSION,
        "preprocessing_mode": "train_only",
        "preprocessing_state_source": "structured checkpoint v2",
        "preprocessing_refit": False,
        "signal_columns": prepared["signal_columns"],
        "conditioning_columns_excluded_from_generative_metrics": ["Occupancy"],
        "rows_per_trial": EXPECTED_ROWS,
        "trial_files": [path.name for path in prepared["trial_paths"]],
        "input_directory": str(input_dir),
        "scipy_diagnostics_available": bool(
            wasserstein_distance is not None and ks_2samp is not None
        ),
        "metric_summary": results["metric_summary"].to_dict(orient="records"),
        "methodology": {
            "mse_standardized": (
                "Upstream elementwise mean-squared error on the four signal columns, "
                "standardized only with the train-only scaler restored from checkpoint v2."
            ),
            "acd": (
                "Upstream mean absolute autocorrelation difference over four signals "
                "and 100 lags."
            ),
            "xcorrD": (
                "Upstream mean absolute difference between full Pearson correlation "
                "matrices of the four signals."
            ),
            "legacy_difference": (
                "Legacy scripts infer non-hierarchical columns and instantiate their "
                "dataset preprocessor. This evaluator selects checkpoint-declared signal "
                "columns explicitly and restores train-only state; the ACD/xcorrD formulas "
                "and population aggregation across trials are unchanged."
            ),
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    metric_table = [
        "| metric | mean | std | trials |",
        "|---|---:|---:|---:|",
    ]
    metric_table.extend(
        "| {metric} | {mean:.10g} | {std:.10g} | {trials} |".format(**row)
        for row in results["metric_summary"].to_dict(orient="records")
    )
    readable = [
        "# UCI Occupancy WaveStitch evaluation",
        "",
        "- Profile: C",
        "- Trials: 5",
        "- Rows per trial: 2014",
        "- Signals: Temperature, Humidity, Light, CO2",
        "- Conditioning excluded from generative metrics: Occupancy",
        "- Preprocessing: train_only restored from checkpoint v2; no refit",
        "",
        "## WaveStitch metrics",
        "",
        *metric_table,
        "",
        "See the CSV files and evaluation_summary.json for per-trial and diagnostic details.",
    ]
    (output_dir / "summary.md").write_text("\n".join(readable) + "\n", encoding="utf-8")


def _dry_run_report(prepared, checkpoint_path, input_dir, output_dir):
    preprocessing = prepared["checkpoint"]["preprocessing"]
    return {
        "mode": "dry-run",
        "dataset": DATASET_ID,
        "profile": PROFILE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": prepared["checkpoint"]["format_version"],
        "preprocessing": {
            "mode": preprocessing["mode"],
            "state_source": "structured checkpoint v2",
            "refit": False,
            "scaler_columns": preprocessing["scaler"]["columns"],
            "scaler_fit_rows": int(preprocessing["scaler"]["n_samples_seen"]),
        },
        "input_directory": str(input_dir),
        "trial_count": len(prepared["trial_paths"]),
        "trial_files": [path.name for path in prepared["trial_paths"]],
        "rows_per_trial": [len(frame) for frame in prepared["synthetic_trials"]],
        "signal_columns": prepared["signal_columns"],
        "conditioning_columns": prepared["metadata_columns"],
        "conditioning_excluded_from_generative_metrics": True,
        "scipy_diagnostics_available": bool(
            wasserstein_distance is not None and ks_2samp is not None
        ),
        "output_directory": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=EXPECTED_TRIALS)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials != EXPECTED_TRIALS:
        parser.error("Final evaluation requires exactly 5 trials.")
    if args.profile != PROFILE:
        parser.error("Final evaluation requires profile C.")
    try:
        prepared = prepare_evaluation(
            args.checkpoint_path, args.input_dir, args.trials
        )
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_report(
                        prepared,
                        args.checkpoint_path,
                        args.input_dir,
                        args.output_dir,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        results = evaluate(prepared)
        _write_results(
            args.output_dir,
            prepared,
            results,
            args.checkpoint_path,
            args.input_dir,
        )
        print(results["metric_summary"].to_string(index=False))
        print("Results: {}".format(args.output_dir))
    except (EvaluationValidationError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
