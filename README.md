# WaveStitch Generalized

WaveStitch Generalized is a configurable and reproducible research implementation for conditional multivariate time-series generation. It is built on top of the original [WaveStitch repository](https://github.com/adis98/HierarchicalTS), while preserving a legacy path for reference reproduction.

> This is an independent follow-up project. It is not the official repository of the original WaveStitch authors, and the results reported here are this project's own reproduction and validation results.

## Why this repository exists

The upstream implementation couples preprocessing and experiment logic to a fixed set of datasets. This project generalizes that workflow so a flat time-series CSV can be described by configuration, validated before execution, preprocessed without test leakage, trained, reconstructed from a structured checkpoint, synthesized, and evaluated through explicit experiment protocols. It also studies what happens when WaveStitch is moved outside its original benchmark setting.

## Main features

- Typed JSON `DatasetConfig` with ordered signals, metadata, temporal keys, split rules, and synthesis conditions.
- Generic `flat_csv` loader with schema, timestamp, ordering, type, and split validation.
- Configurable `column_values`, chronological `ratio`, and `timestamp` splits.
- Config-driven C/M/F synthesis masks without inventing a hierarchy for datasets that do not have one.
- Leakage-free `train_only` preprocessing: scalers and encoders are fitted on train rows only.
- Reproducible `upstream_legacy` preprocessing for MetroTraffic and upstream comparisons.
- Structured checkpoint v2 with architecture, diffusion, column layout, dataset snapshot, and fitted preprocessing state.
- Automatic model and preprocessing reconstruction during structured-checkpoint synthesis; no refit on test data.
- JSON experiment configurations plus CLI overrides for training, synthesis, and dry-runs.
- Window, context, stride, mask, and checkpoint/config compatibility validation.
- Generic WaveStitch training and pipeline synthesis for configured datasets.
- UCI Occupancy evaluation, empirical conditioned baseline, and isolated sampler-ablation tooling.
- Regression tests for upstream compatibility, leakage prevention, checkpoint round-trips, and sampler defaults.

The primary generalized entry points are `training_wavestitch.py` and `synthesis_wavestitch_pipeline_strided_preconditioning.py`. Other scripts retained from upstream are research utilities and may still follow their original dataset-specific interfaces.

## Installation and quick start

Python 3.10 or newer is recommended. Start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

The requirements do not assume CUDA. CPU and macOS execution use the built-in slow Cauchy-kernel fallback and are functionally supported, but full S4 training can be very slow. On a compatible NVIDIA system, install the PyTorch build matching the local CUDA runtime before installing the remaining requirements. PyKeOps or the upstream CUDA Cauchy extension can accelerate the S4 kernel on supported GPU environments; they are optional and are not part of the portable CPU installation.

After preparing a dataset, validate a complete protocol without training:

```bash
python training_wavestitch.py \
  --experiment-config configs/experiments/uci_occupancy_smoke.json \
  --dry-run
```

Dry-run loads and validates data, resolves preprocessing and model dimensions, and checks split/window counts. It exits before model execution.

## Custom dataset example

For an ordered CSV at `data/example/readings.csv` with columns `timestamp,temperature,power,site`, create `configs/datasets/Example.json`:

```json
{
  "dataset_id": "Example",
  "csv_path": "data/example/readings.csv",
  "loader": "flat_csv",
  "preprocessing_mode": "train_only",
  "timestamp_column": "timestamp",
  "signal_columns": ["temperature", "power"],
  "metadata_columns": ["site"],
  "cyclic_columns": ["site"],
  "dtype_overrides": {"site": "string"},
  "temporal_order": ["timestamp"],
  "split": {"mode": "ratio", "train_ratio": 0.8},
  "synthesis_conditions": {
    "C": {},
    "F": {"site": "north"}
  }
}
```

`train_only` is the recommended mode. A metadata category present at transform time but absent from train raises an explicit error rather than receiving a silent, semantically ambiguous encoding. See [Dataset configuration](docs/DATASET_CONFIG.md) for the complete schema and constraints.

## Reproducing the experiments

Datasets, checkpoints, generated series, logs, and runtime markers are intentionally not distributed in this repository.

### MetroTraffic reference experiment

Obtain the [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and place its CSV at the path declared in `configs/datasets/MetroTraffic.json`. The configuration uses `upstream_legacy` preprocessing to preserve the original 40,255/7,949-row split, 13-column representation, 40,224 training windows, and C/M/F masks.

```bash
python training_wavestitch.py \
  --experiment-config configs/experiments/metrotraffic_upstream.json \
  --dry-run

# Remove --dry-run only when a full training run is intended.
python training_wavestitch.py \
  --experiment-config configs/experiments/metrotraffic_upstream.json

python synthesis_wavestitch_pipeline_strided_preconditioning.py \
  --experiment-config configs/experiments/metrotraffic_upstream.json \
  --synthesis-profile C \
  --output-dir generated/metrotraffic-reference
```

Repeat synthesis with profiles M and F for the other configured upstream conditions. The exact upstream authors' checkpoint was not available; the numbers below are a technical reproduction/reference validation, not a claim of exact checkpoint reproduction.

### UCI Occupancy experiment

Download the [UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy+detection), extract `datatraining.txt`, and place an unchanged copy at `data/uci-occupancy-detection/occupancy_training.csv`. The file has 8,143 one-minute rows. The timestamp cutoff in `configs/datasets/UCIOccupancyDetection.json` produces 6,129 train and 2,014 test rows.

```bash
python training_wavestitch.py \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --dry-run

# Full training (300 epochs; potentially very slow on CPU).
python -u training_wavestitch.py \
  --experiment-config configs/experiments/uci_occupancy_full.json

python -u synthesis_wavestitch_pipeline_strided_preconditioning.py \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --output-dir generated/uci-occupancy-reproduction
```

The structured v2 checkpoint is authoritative for the window, architecture, diffusion schedule, column layout, and preprocessing fit state. Synthesis restores this state and never refits it. Detailed protocols and evaluator commands are in [Experiments](docs/EXPERIMENTS.md).

## Scientific results

All three metrics are distances; lower is better. Reported dispersion is the population standard deviation over five trials.

### MetroTraffic

| Profile | MSE | ACD | xCorr |
|---|---:|---:|---:|
| C | 0.533067 ± 0.008743 | 0.145402 ± 0.009127 | 0.098706 ± 0.003757 |
| M | 0.323540 ± 0.032583 | 0.113842 ± 0.013006 | 0.121345 ± 0.005321 |
| F | 0.153312 ± 0.008988 | 0.045260 ± 0.001747 | 0.053775 ± 0.009166 |

These are this project's technical reproduction/reference-validation results. The exact checkpoint used by the upstream authors was unavailable.

### UCI Occupancy

| MSE | ACD | xcorrD |
|---:|---:|---:|
| 3.7096 ± 0.0562 | 0.3480 ± 0.0343 | 0.4088 ± 0.0643 |

The end-to-end pipeline completed correctly, but generative quality was insufficient. Temperature, Humidity, and CO2 showed variance collapse; Humidity and the high CO2 tail shifted strongly from train to test; and continuous diffusion failed to reproduce Light's large exact-zero mass, producing negative Light values in about 46% of synthetic rows. `Occupancy` alone did not identify the environmental regime well enough.

## Diagnostic experiments

The train-only empirical conditioned block-bootstrap baseline used only train signals and the test `Occupancy` sequence. Across five trials it achieved MSE `3.2069 ± 0.4878`, ACD `0.2657 ± 0.0654`, and xcorrD `0.3741 ± 0.0879`. It improved all three WaveStitch means but remained far from the shifted test regime, especially for Humidity and CO2. Verdict: **`DATA/CONDITIONING LIMIT DOMINANT`**.

Two isolated one-trial sampler ablations reused the same trained checkpoint. Disabling gradient correction produced no relevant benefit. Replacing the legacy reverse-noise amplitude with `sqrt(posterior variance)` partially increased dispersion and reduced negative Light values, but did not restore Light's zero mass or improve overall metrics. Verdict: **`SAMPLER CONTRIBUTION MODERATE`**.

Therefore, distribution shift and insufficient conditioning are the dominant limitations in this experiment; the legacy reverse-noise behavior contributes moderately to under-dispersion. `--sqrt-posterior-variance` remains an experimental, non-default variant. `--disable-gradient-correction` is an ablation switch only, not a recommended improvement. The default behavior remains upstream-compatible.

See [Results](docs/RESULTS.md) and the machine-readable summaries in [`results/`](results/README.md).

## Limitations

- Panel/grouped time series are not yet supported.
- Multi-interval or interlaced splits are not supported.
- Synthesis stride greater than the window size is rejected.
- Train-only preprocessing can reveal genuine train-to-test shifts previously masked by legacy full-dataset fitting.
- Continuous diffusion does not guarantee physical support constraints or discrete masses such as `Light = 0`.
- The CPU Cauchy fallback is much slower than CUDA acceleration.
- No single hyperparameter set is claimed to generalize to every dataset.

The full scope and interpretation are documented in [Limitations](docs/LIMITATIONS.md).

## Relationship with the original bachelor thesis

This repository is a later follow-up and generalization of the author's bachelor-thesis work. The original thesis applied WaveStitch to proprietary or otherwise non-public energy/HPC data. Those research-centre data are not included here and must not be reconstructed or simulated as if they were the originals.

Future public thesis repository: _link to be added when available_.

## Testing and repository policy

Run the complete suite with:

```bash
pytest -q
```

The release baseline is 158 passing tests. Raw datasets, virtual environments, model checkpoints, generated CSVs, evaluation workspaces, logs, PID/status files, cache directories, and machine-specific launchers are excluded from version control. Small, curated scientific summaries are retained under `results/` with provenance notes.

## Upstream attribution and license

WaveStitch Generalized derives from [adis98/HierarchicalTS](https://github.com/adis98/HierarchicalTS), base commit `dcb1e98eb3bc31f4fa1c0ce3bfef4dcd8e473e47`. The upstream MIT `LICENSE` and copyright notice are retained unchanged. See [Upstream provenance](docs/UPSTREAM.md) for the boundary between preserved upstream code and this independent follow-up.
