# Experiment protocols

This document records the protocols used for the versioned results. It does not embed raw data, generated series, or checkpoints. All commands assume the repository root as the working directory and an activated environment created from `requirements.txt`.

## Common methodology

The generalized pipeline declares signal and metadata columns separately. Only signal columns are diffused and included in generative metrics. Metadata provides conditioning. Structured v2 synthesis reconstructs model and preprocessing state from the checkpoint and rejects incompatible explicit configuration; it never refits a scaler or encoder.

MSE is elementwise mean-squared error after transformation with the fitted scaler. ACD is the upstream mean absolute autocorrelation difference across signals and 100 lags. xcorrD is the upstream mean absolute difference between full Pearson signal-correlation matrices. The UCI evaluator explicitly selects checkpoint-declared signals while preserving the upstream formulas and population aggregation over trials.

## MetroTraffic reference validation

### Data and preprocessing

The source is the [UCI Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume). It is not redistributed here. `configs/datasets/MetroTraffic.json` uses the original dataset-specific loader and `upstream_legacy` preprocessing.

- Signals: `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `traffic_volume`
- Metadata: `year`, `month`, `day`, `hour`, cyclically encoded as upstream
- Split: calendar year 2018 is test
- Train/test rows: 40,255 / 7,949
- Preprocessed columns: 13
- Window: 32; effective training stride: 1
- Training windows: 40,224
- Synthesis stride: 8; trials: 5
- Conditions: C = full test; M = day 15; F = hour 6

Architecture, diffusion, optimizer, and artifact paths are recorded in `configs/experiments/metrotraffic_upstream.json`. The experiment uses 200 diffusion steps and the S4 configuration inherited from the WaveStitch protocol.

This run is a technical reproduction/reference validation. Hardware and installed-kernel choices can materially change runtime. The exact checkpoint used by the upstream authors was unavailable, so an exact checkpoint comparison is not claimed.

## UCI Occupancy full experiment

### Data

The source is `datatraining.txt` from the [UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy+detection), stored locally as `data/uci-occupancy-detection/occupancy_training.csv`. It contains 8,143 consecutive one-minute observations from 2015-02-04 17:51 through 2015-02-10 09:33.

- Signals: `Temperature`, `Humidity`, `Light`, `CO2`
- Conditioning metadata: `Occupancy`
- Excluded: timestamp/index and derived `HumidityRatio`
- Preprocessing: `train_only`
- Split: timestamp cutoff `2015-02-09T00:00:00Z`
- Train/test rows: 6,129 / 2,014
- Window: 120 minutes; effective training stride: 1
- Training windows: 6,010

The signal `StandardScaler` was fitted only on the 6,129 train rows. `Occupancy` is numeric metadata and is passed through as conditioning; it is not scaled and is excluded from generative metric averages. The v2 checkpoint stores the fitted scaler and preprocessing mode.

### Training and synthesis

The complete protocol is `configs/experiments/uci_occupancy_full.json`:

- batch size 256, 300 epochs, Adam learning rate `1e-4`, seed 42;
- 200 diffusion steps with beta endpoints `1e-4` and `0.02`;
- S4 backbone, four residual layers, 64 residual and skip channels;
- diffusion embedding 32/64/64;
- S4 `lmax=128`, state size 64, bidirectional, layer normalization, dropout 0;
- synthesis profile C, stride 30, batch size 64, five trials;
- 26 rows of leading synthesis context and 65 synthesis windows covering all 2,014 test rows selected by C.

Training writes structured checkpoint v2. Synthesis infers the dataset and restores the train-only scaler from that checkpoint. The full CPU fallback run is valid but substantially slower than an accelerated CUDA environment.

### Reproduction commands

Always preflight first:

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --dry-run
```

Full runs are intentionally explicit:

```bash
python -u -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_full.json

python -u -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning \
  --experiment-config configs/experiments/uci_occupancy_full.json \
  --output-dir generated/uci-occupancy-reproduction
```

Evaluation of five completed trial CSVs:

```bash
python -u -m scripts.evaluation.evaluate_uci_occupancy_full \
  --checkpoint-path saved_models/UCIOccupancyDetection/full_experiment_v2.pth \
  --input-dir generated/uci-occupancy-reproduction/UCIOccupancyDetection/C \
  --output-dir evaluation_results/uci-occupancy-reproduction
```

Use `--dry-run` on the evaluator to check checkpoint version, fitted preprocessing restoration, signal columns, trial count, file shape, and no-refit behavior without computing the campaign.

## Empirical conditioned baseline

The non-parametric diagnostic draws signal blocks exclusively from the train split. It partitions the exact test `Occupancy` sequence into constant-state runs, samples compatible train runs, and concatenates blocks up to 120 rows until the 2,014-row target is filled. It uses no real test signal values. Five seeded trials make the comparison with WaveStitch homogeneous.

The baseline is deliberately simple: it asks how much of the test can be explained using only the train signal regimes and the available binary conditioning. Its improvement over WaveStitch, combined with its persistent Humidity/CO2 mismatch, supports the `DATA/CONDITIONING LIMIT DOMINANT` diagnostic rather than proving that block bootstrap is a superior generator.

## Sampler ablations

Two one-trial ablations reused exactly the same final UCI checkpoint, dataset, preprocessing, seed, conditioning, architecture, window, stride, and diffusion schedule.

- Variant A, `--disable-gradient-correction`: disables only the fixed stitching-gradient update; legacy reverse noise remains unchanged.
- Variant B, `--sqrt-posterior-variance`: retains gradient correction and changes only reverse-noise amplitude from posterior variance to its square root.

The flags are mutually exclusive. Default synthesis enables gradient correction and uses legacy posterior-variance amplitude, preserving upstream/generalized behavior. Variant A is an ablation only. Variant B is experimental and non-default. Because each variant has one trial on one dataset, the result is diagnostic rather than a general sampler recommendation.
