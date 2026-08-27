# Versioned result summaries

This directory contains small, machine-readable summaries selected from completed experiments. Raw datasets, model checkpoints, generated time series, logs, PID/status files, and complete runtime evaluation directories are intentionally excluded.

## Provenance

- `metrotraffic/reference_metrics.csv` was transcribed without numerical rounding from this project's successful five-trial technical reference evaluation for profiles C, M, and F. The exact upstream-authors checkpoint was unavailable.
- `uci_occupancy/wavestitch_*` is copied from the completed five-trial profile-C evaluation using structured checkpoint v2 and checkpoint-restored `train_only` preprocessing.
- `uci_occupancy/empirical_*` and `comparison_*` are copied from the five-trial train-only conditioned block-bootstrap diagnostic.
- `uci_occupancy/sampler_*` is copied from the isolated one-trial sampler ablation; default trial 0 is the already completed WaveStitch trial, not a rerun.

All UCI generative metrics select only `Temperature`, `Humidity`, `Light`, and `CO2`. `Occupancy` is conditioning metadata and is excluded from metric means. Metric definitions and interpretation are documented in `docs/EXPERIMENTS.md` and `docs/RESULTS.md`.

These tables are research records of this independent project, not results of the original WaveStitch authors.
