# WaveStitch Generalized

I built this repository to continue the work I started in my bachelor's thesis on synthetic multivariate time series. The thesis applied WaveStitch to energy and environmental measurements from an HPC research centre; here I separated that application-specific work from the model pipeline and turned the pipeline into something I can configure, validate and test on public datasets.

This is not the official WaveStitch repository. It is my generalization of [adis98/HierarchicalTS](https://github.com/adis98/HierarchicalTS), with the original MIT license and attribution preserved. My submitted thesis, including its scope and data-availability notes, is in [Frero0/bachelors-thesis-unito](https://github.com/Frero0/bachelors-thesis-unito).

## What I changed

The upstream code assumes a small set of datasets and keeps much of the experiment logic inside dataset-specific scripts. I added a configuration layer around that workflow so that a flat time-series CSV can be described without editing the loader itself.

The generalized path now provides:

- typed JSON configurations for datasets and experiments;
- chronological, timestamp-based and column-based train/test splits;
- preprocessing fitted only on the training partition, with an explicit legacy mode for upstream reproduction;
- configurable C/M/F conditioning masks;
- structured checkpoints containing the model layout, diffusion settings, dataset snapshot and fitted preprocessing state;
- validation of schemas, windows, strides, masks and checkpoint/config compatibility before an expensive run starts;
- reproducible training, synthesis and evaluation commands;
- regression tests for the generalized path and for behavior retained from upstream.

I kept the older model variants, ablations and plotting utilities because they document the research path, but they are separated from the main entry points instead of filling the repository root.

## Repository layout

| Path | Contents |
|---|---|
| `wavestitch/` | Reusable loading, preprocessing, configuration, checkpointing and model code |
| `scripts/training/` | Training entry points |
| `scripts/synthesis/` | WaveStitch and comparison-model synthesis entry points |
| `scripts/evaluation/` | UCI Occupancy evaluation, baseline and sampler ablations |
| `scripts/analysis/` | Historical analyses, tables, plots and animation utilities |
| `scripts/experiments/` | Shell launchers for the retained upstream-style experiments |
| `configs/datasets/` | Dataset schemas and split/conditioning rules |
| `configs/experiments/` | Reproducible experiment settings |
| `tests/` | Unit and regression tests |
| `results/` | Curated, small result tables that can be versioned |
| `docs/` | Configuration references, protocols, limitations and upstream provenance |

Raw data, checkpoints, generated samples, logs and local run state are intentionally ignored by Git.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The requirements are portable and do not force a CUDA build. CPU and macOS runs use the slow Cauchy-kernel fallback; this is enough for tests and dry-runs, but full S4 training is much faster with a compatible NVIDIA/CUDA setup and either PyKeOps or the upstream CUDA extension.

## Start with a dry-run

The quickest way to inspect the complete generalized pipeline is the small UCI Occupancy protocol:

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_smoke.json \
  --dry-run
```

A dry-run loads the dataset, applies the declared split, resolves the processed column layout, checks the available windows and validates the model configuration. It stops before model execution.

To add a dataset, create a JSON file in `configs/datasets/` and point `csv_path` to an ordered flat CSV. The configuration declares signal columns, metadata columns, temporal ordering, split rules and synthesis conditions. The complete schema and an example are in [Dataset configuration](docs/DATASET_CONFIG.md); experiment-level settings are documented in [Experiment configuration](docs/EXPERIMENT_CONFIG.md).

## Reproducing the public experiments

### MetroTraffic reference

Download the [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and place it at the path declared in `configs/datasets/MetroTraffic.json`. This configuration deliberately uses `upstream_legacy` preprocessing so that I can compare the generalized pipeline with the original data layout.

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/metrotraffic_upstream.json \
  --dry-run

python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/metrotraffic_upstream.json

python -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning \
  --experiment-config configs/experiments/metrotraffic_upstream.json \
  --synthesis-profile C \
  --output-dir generated/metrotraffic-reference
```

Repeat synthesis with profiles `M` and `F` for the other conditioning levels. I did not have the exact checkpoint used by the original WaveStitch authors, so these figures are reference-reproduction results rather than an exact checkpoint reproduction.

| Profile | MSE | ACD | xCorr |
|---|---:|---:|---:|
| C | 0.533067 ± 0.008743 | 0.145402 ± 0.009127 | 0.098706 ± 0.003757 |
| M | 0.323540 ± 0.032583 | 0.113842 ± 0.013006 | 0.121345 ± 0.005321 |
| F | 0.153312 ± 0.008988 | 0.045260 ± 0.001747 | 0.053775 ± 0.009166 |

### UCI Occupancy case study

Download the [UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy+detection), extract `datatraining.txt` and save an unchanged copy as `data/uci-occupancy-detection/occupancy_training.csv`. The configured timestamp cutoff produces 6,129 training rows and 2,014 test rows.

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --dry-run

python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_full.json

python -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --output-dir generated/uci-occupancy-reproduction
```

Across five trials I obtained the following distances (lower is better; dispersion is the population standard deviation):

| MSE | ACD | xCorrD |
|---:|---:|---:|
| 3.7096 ± 0.0562 | 0.3480 ± 0.0343 | 0.4088 ± 0.0643 |

The pipeline completed correctly, but the generated data were not good enough. Temperature, Humidity and CO2 lost too much variance; the test partition also contains a strong Humidity/CO2 shift; and continuous diffusion did not reproduce the large exact-zero mass of `Light`. Conditioning only on `Occupancy` did not identify the environmental regime.

I checked that conclusion with two additional experiments. A train-only conditioned block-bootstrap baseline improved the three mean distances but still missed the shifted test regime. Sampler ablations showed that the legacy reverse-noise rule contributes to under-dispersion, but changing it did not solve the main failure. My interpretation is therefore that data shift and insufficient conditioning dominate this case, while the sampler has a smaller but measurable effect.

The protocols, evaluator commands and machine-readable tables are in [Experiments](docs/EXPERIMENTS.md), [Results](docs/RESULTS.md) and [`results/`](results/README.md).

## Boundaries of the implementation

At the moment the generalized loader supports ordered flat CSV time series, not grouped/panel series. Splits must be contiguous rather than interlaced, synthesis stride cannot exceed the window size, and the model does not enforce physical bounds or discrete masses. `train_only` preprocessing can also expose real train/test shifts that legacy full-dataset fitting hides. I do not claim that one model configuration generalizes to every dataset.

The original HPC measurements used in my thesis are not public and are not included, reconstructed or simulated here. See [Limitations](docs/LIMITATIONS.md) for the technical list and the [thesis repository](https://github.com/Frero0/bachelors-thesis-unito) for the original research context.

## Tests, provenance and license

The current regression baseline is **158 passing tests**:

```bash
python -m pytest -q
```

This work derives from [adis98/HierarchicalTS](https://github.com/adis98/HierarchicalTS) at base commit `dcb1e98eb3bc31f4fa1c0ce3bfef4dcd8e473e47`. I retained the upstream MIT license and copyright notice. [Upstream provenance](docs/UPSTREAM.md) records what remains upstream-derived and what I added in this repository.

## Author

**Federico Santorsola**

- [GitHub](https://github.com/Frero0)
- [federico.santorsola@edu.unito.it](mailto:federico.santorsola@edu.unito.it)
