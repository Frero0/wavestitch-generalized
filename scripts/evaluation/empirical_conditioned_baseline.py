"""Leakage-free empirical conditioned baseline for UCI Occupancy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import evaluate_uci_occupancy_full as evaluation


METHOD = "empirical_conditioned_run_block_bootstrap"
DATASET = "UCIOccupancyDetection"
PROFILE = "C"
SIGNALS = ["Temperature", "Humidity", "Light", "CO2"]
METADATA = ["Occupancy"]


class EmpiricalBaselineError(ValueError):
    """Raised when the baseline cannot satisfy its no-leakage contract."""


def _constant_runs(values):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise EmpiricalBaselineError("Occupancy conditioning must be one-dimensional and non-empty.")
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [
        (int(boundaries[index]), int(boundaries[index + 1]), values[boundaries[index]])
        for index in range(len(boundaries) - 1)
    ]


def _compatible_train_starts(train_occupancy, state, length):
    values = np.asarray(train_occupancy)
    return np.asarray(
        [
            start
            for start in range(len(values) - length + 1)
            if np.all(values[start:start + length] == state)
        ],
        dtype=int,
    )


def conditioned_run_block_bootstrap(
    train_frame,
    test_occupancy,
    *,
    signal_columns=SIGNALS,
    occupancy_column="Occupancy",
    block_size=120,
    seed=42,
):
    """Sample train-only constant-state blocks matching test Occupancy exactly."""

    if block_size <= 0:
        raise EmpiricalBaselineError("block_size must be positive.")
    required = list(signal_columns) + [occupancy_column]
    missing = [column for column in required if column not in train_frame]
    if missing:
        raise EmpiricalBaselineError(
            "Train frame is missing column(s): {}.".format(", ".join(missing))
        )
    train = train_frame[required].reset_index(drop=False).rename(
        columns={"index": "_source_original_index"}
    )
    test_condition = pd.Series(test_occupancy, name=occupancy_column).reset_index(drop=True)
    train_states = train[occupancy_column].to_numpy()
    generated_parts = []
    provenance = []
    rng = np.random.default_rng(seed)

    for run_start, run_end, state in _constant_runs(test_condition.to_numpy()):
        destination = run_start
        while destination < run_end:
            length = min(block_size, run_end - destination)
            candidates = _compatible_train_starts(train_states, state, length)
            if len(candidates) == 0:
                raise EmpiricalBaselineError(
                    "No train block of length {} exists for Occupancy={!r}.".format(
                        length, state
                    )
                )
            source_start = int(rng.choice(candidates))
            source_end = source_start + length
            part = train.iloc[source_start:source_end][list(signal_columns)].copy()
            part[occupancy_column] = state
            generated_parts.append(part)
            provenance.append(
                {
                    "destination_start": destination,
                    "destination_end_exclusive": destination + length,
                    "occupancy": int(state),
                    "length": length,
                    "source_train_position_start": source_start,
                    "source_train_position_end_exclusive": source_end,
                    "source_original_index_start": int(
                        train.iloc[source_start]["_source_original_index"]
                    ),
                    "source_original_index_end": int(
                        train.iloc[source_end - 1]["_source_original_index"]
                    ),
                    "compatible_candidate_count": int(len(candidates)),
                }
            )
            destination += length

    generated = pd.concat(generated_parts, ignore_index=True)[required]
    if len(generated) != len(test_condition):
        raise EmpiricalBaselineError(
            "Generated length {} differs from conditioned test length {}.".format(
                len(generated), len(test_condition)
            )
        )
    if not np.array_equal(
        generated[occupancy_column].to_numpy(), test_condition.to_numpy()
    ):
        raise EmpiricalBaselineError("Generated Occupancy does not match test conditioning.")
    return generated, provenance


def _metric_table(frame):
    lines = ["| metric | mean | std | trials |", "|---|---:|---:|---:|"]
    lines.extend(
        "| {metric} | {mean:.10g} | {std:.10g} | {trials} |".format(**row)
        for row in frame.to_dict(orient="records")
    )
    return lines


def _build_comparisons(baseline_results, wavestitch_results_dir):
    wav_metrics = pd.read_csv(wavestitch_results_dir / "metric_summary.csv")
    base_metrics = baseline_results["metric_summary"]
    metric_comparison = wav_metrics.merge(
        base_metrics, on="metric", suffixes=("_wavestitch", "_baseline")
    )
    metric_comparison["baseline_minus_wavestitch"] = (
        metric_comparison["mean_baseline"] - metric_comparison["mean_wavestitch"]
    )
    metric_comparison["better_lower_mean"] = np.where(
        metric_comparison["mean_baseline"] < metric_comparison["mean_wavestitch"],
        "empirical_baseline",
        "wavestitch",
    )

    wav_signal = pd.read_csv(wavestitch_results_dir / "signal_diagnostics.csv")
    base_signal = baseline_results["signal_diagnostics"]
    diagnostic_columns = [
        "synthetic_mean",
        "synthetic_std",
        "synthetic_q05",
        "synthetic_q50",
        "synthetic_q95",
        "wasserstein_1",
        "ks_statistic",
    ]
    wav_global = (
        wav_signal[wav_signal["cohort"] == "all"]
        .groupby("signal", sort=False)[diagnostic_columns]
        .mean()
        .add_suffix("_wavestitch")
        .reset_index()
    )
    base_global = (
        base_signal[base_signal["cohort"] == "all"]
        .groupby("signal", sort=False)[diagnostic_columns]
        .mean()
        .add_suffix("_baseline")
        .reset_index()
    )
    signal_comparison = wav_global.merge(base_global, on="signal")

    wav_constraints = pd.read_csv(
        wavestitch_results_dir / "constraint_diagnostics.csv"
    )
    base_constraints = baseline_results["constraint_diagnostics"]
    wav_constraints = (
        wav_constraints[wav_constraints["series"] == "synthetic"]
        .groupby("diagnostic", sort=False)["value"]
        .mean()
        .rename("wavestitch_percent")
        .reset_index()
    )
    base_constraints = (
        base_constraints[base_constraints["series"] == "synthetic"]
        .groupby("diagnostic", sort=False)["value"]
        .mean()
        .rename("baseline_percent")
        .reset_index()
    )
    constraint_comparison = wav_constraints.merge(base_constraints, on="diagnostic")

    wav_corr = pd.read_csv(wavestitch_results_dir / "occupancy_correlations.csv")
    base_corr = baseline_results["occupancy_correlations"]
    wav_corr = (
        wav_corr.groupby("signal", sort=False)[
            ["synthetic_occupancy_correlation", "absolute_difference"]
        ]
        .mean()
        .add_suffix("_wavestitch")
        .reset_index()
    )
    base_corr = (
        base_corr.groupby("signal", sort=False)[
            ["real_occupancy_correlation", "synthetic_occupancy_correlation", "absolute_difference"]
        ]
        .mean()
        .add_suffix("_baseline")
        .reset_index()
    )
    correlation_comparison = wav_corr.merge(base_corr, on="signal")
    return {
        "comparison_metrics": metric_comparison,
        "comparison_signal_diagnostics": signal_comparison,
        "comparison_constraints": constraint_comparison,
        "comparison_occupancy_correlations": correlation_comparison,
    }


def _write_outputs(
    generated_dir,
    results_dir,
    prepared,
    results,
    comparisons,
    provenance,
    *,
    checkpoint_path,
    wavestitch_results_dir,
    block_size,
    seeds,
):
    if generated_dir.exists() or results_dir.exists():
        raise EmpiricalBaselineError(
            "Refusing to overwrite existing generated/results directories."
        )
    generated_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    for trial, frame in enumerate(prepared["synthetic_trials"]):
        frame.to_csv(generated_dir / "empirical_conditioned_trial_{}.csv".format(trial), index=False)
    generation_summary = {
        "method": METHOD,
        "dataset": DATASET,
        "profile": PROFILE,
        "block_size": block_size,
        "trials": len(prepared["synthetic_trials"]),
        "seeds": seeds,
        "signal_source": "train split only",
        "conditioning_source": "test Occupancy only",
        "test_signal_values_used_for_generation": False,
        "exact_test_occupancy_preserved": True,
        "rows_per_trial": len(prepared["real_physical"]),
        "trial_blocks": provenance,
    }
    (generated_dir / "generation_provenance.json").write_text(
        json.dumps(generation_summary, indent=2), encoding="utf-8"
    )

    result_frames = {
        "trial_metrics.csv": results["trial_metrics"],
        "metric_summary.csv": results["metric_summary"],
        "signal_diagnostics.csv": results["signal_diagnostics"],
        "constraint_diagnostics.csv": results["constraint_diagnostics"],
        "occupancy_correlations.csv": results["occupancy_correlations"],
        "comparison_metrics.csv": comparisons["comparison_metrics"],
        "comparison_signal_diagnostics.csv": comparisons["comparison_signal_diagnostics"],
        "comparison_constraints.csv": comparisons["comparison_constraints"],
        "comparison_occupancy_correlations.csv": comparisons[
            "comparison_occupancy_correlations"
        ],
    }
    for name, frame in result_frames.items():
        frame.to_csv(results_dir / name, index=False)

    summary = {
        "method": METHOD,
        "dataset": DATASET,
        "profile": PROFILE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": 2,
        "preprocessing_mode": "train_only",
        "preprocessing_state_source": "structured checkpoint v2",
        "preprocessing_refit": False,
        "signal_columns": SIGNALS,
        "conditioning_columns_excluded_from_generative_metrics": METADATA,
        "block_size": block_size,
        "rows_per_trial": len(prepared["real_physical"]),
        "trials": len(prepared["synthetic_trials"]),
        "wavestitch_results_directory": str(wavestitch_results_dir),
        "metric_summary": results["metric_summary"].to_dict(orient="records"),
        "metric_comparison": comparisons["comparison_metrics"].to_dict(orient="records"),
    }
    (results_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    readable = [
        "# UCI Occupancy empirical conditioned baseline",
        "",
        "- Signals sampled exclusively from the train split",
        "- Conditioning: exact test Occupancy sequence",
        "- Run-conditioned block bootstrap, maximum block length: {}".format(block_size),
        "- Trials: 5",
        "- Preprocessing: train_only scaler restored from checkpoint v2; no refit",
        "",
        "## Baseline metrics",
        "",
        *_metric_table(results["metric_summary"]),
        "",
        "## WaveStitch vs empirical baseline",
        "",
        comparisons["comparison_metrics"].to_string(index=False),
    ]
    (results_dir / "summary.md").write_text("\n".join(readable) + "\n", encoding="utf-8")


def run(args):
    checkpoint = evaluation._load_checkpoint(args.checkpoint_path)
    preprocessor = evaluation.restore_evaluation_preprocessor(checkpoint)
    if list(checkpoint["model"]["signal_columns"]) != SIGNALS:
        raise EmpiricalBaselineError("Checkpoint signal layout is incompatible.")
    if list(checkpoint["model"]["metadata_columns"]) != METADATA:
        raise EmpiricalBaselineError("Checkpoint metadata layout is incompatible.")
    train = preprocessor.df_orig.loc[preprocessor.train_indices, SIGNALS + METADATA]
    test_occupancy = preprocessor.df_orig.loc[preprocessor.test_indices, "Occupancy"]
    if len(train) != 6129 or len(test_occupancy) != 2014:
        raise EmpiricalBaselineError("Unexpected UCI Occupancy split cardinality.")

    trials = []
    provenance = []
    seeds = [args.seed + trial for trial in range(args.trials)]
    for trial, seed in enumerate(seeds):
        generated, trial_provenance = conditioned_run_block_bootstrap(
            train,
            test_occupancy,
            block_size=args.block_size,
            seed=seed,
        )
        trials.append(generated)
        provenance.append(
            {"trial": trial, "seed": seed, "blocks": trial_provenance}
        )

    real_physical = (
        preprocessor.df_orig.loc[preprocessor.test_indices, SIGNALS + METADATA]
        .reset_index(drop=True)
        .apply(pd.to_numeric, errors="raise")
    )
    real_standardized = (
        preprocessor.df_cleaned.loc[preprocessor.test_indices, SIGNALS]
        .reset_index(drop=True)
        .astype(float)
    )
    prepared = {
        "checkpoint": checkpoint,
        "preprocessor": preprocessor,
        "signal_columns": SIGNALS,
        "metadata_columns": METADATA,
        "trial_paths": [],
        "synthetic_trials": trials,
        "real_physical": real_physical,
        "real_standardized": real_standardized,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "method": METHOD,
                    "train_rows_available": len(train),
                    "test_rows_conditioned": len(test_occupancy),
                    "trials": len(trials),
                    "block_size": args.block_size,
                    "exact_occupancy_matches": [
                        bool(
                            np.array_equal(
                                frame["Occupancy"].to_numpy(),
                                test_occupancy.to_numpy(),
                            )
                        )
                        for frame in trials
                    ],
                    "test_signal_values_used_for_generation": False,
                    "generated_directory": str(args.generated_dir),
                    "results_directory": str(args.results_dir),
                },
                indent=2,
            )
        )
        return

    results = evaluation.evaluate(prepared)
    comparisons = _build_comparisons(results, args.wavestitch_results_dir)
    _write_outputs(
        args.generated_dir,
        args.results_dir,
        prepared,
        results,
        comparisons,
        provenance,
        checkpoint_path=args.checkpoint_path,
        wavestitch_results_dir=args.wavestitch_results_dir,
        block_size=args.block_size,
        seeds=seeds,
    )
    print(results["metric_summary"].to_string(index=False))
    print(comparisons["comparison_metrics"].to_string(index=False))
    print("Generated: {}".format(args.generated_dir))
    print("Results: {}".format(args.results_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--wavestitch-results-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=120)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.trials != 5:
        parser.error("This diagnostic requires exactly 5 trials.")
    try:
        run(args)
    except (EmpiricalBaselineError, evaluation.EvaluationValidationError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
