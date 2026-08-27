# Limitations

This implementation makes its current scope explicit.

## Data model

- Panel or grouped time series are not supported. A configured dataset is treated as one continuous ordered series; entity boundaries would currently create invalid cross-entity windows.
- Multi-interval and interlaced train/test splits are not supported. Available splits are value complement, one chronological ratio boundary, or one timestamp cutoff.
- Duplicate complete temporal-order keys are rejected. Repeated timestamps are only valid when additional configured ordering columns make the composite key unique.
- Flat-CSV non-numeric metadata currently needs cyclic/category encoding; general-purpose categorical embeddings and unknown buckets are not implemented.

## Windowing and synthesis

- Synthesis stride greater than the window size is unsupported and rejected because it would leave uncovered gaps.
- Config-driven synthesis masks are implemented in the generalized pipeline synthesis entry point; historical upstream synthesis scripts retain their original interfaces.
- A continuous diffusion output does not enforce physical support. It can produce negative values for non-negative signals.
- Continuous diffusion does not represent point masses directly. Zero-inflated signals such as UCI Occupancy `Light` may lose their exact-zero mass even when broad occupancy relationships are learned.

## Preprocessing and distribution shift

- `train_only` avoids leakage but can expose genuine train-to-test distribution shifts that were partly masked by fitting preprocessing on the complete dataset. Values far outside the train regime remain far outside it after transformation.
- A metadata category unseen during train fitting raises an error. No unknown bucket is silently introduced because it would change conditioning semantics.
- `upstream_legacy` intentionally fits preprocessing on all rows. It exists for reference reproduction and should not be interpreted as leakage-free evaluation.

## Modeling and conditioning

- Conditioning variables must contain enough information to identify the regime being generated. Binary `Occupancy` did not explain the Humidity and CO2 regime shift in the UCI experiment.
- Model hyperparameters are dataset-dependent. The MetroTraffic protocol and the UCI protocol are references, not universal defaults or evidence that one setting generalizes to all domains.
- The sampler ablations were evaluated for one trial on one dataset. `--sqrt-posterior-variance` is experimental and `--disable-gradient-correction` is diagnostic only.

## Performance and reproducibility

- The pure PyTorch Cauchy fallback supports CPU/macOS but is much slower than a compatible CUDA extension or PyKeOps GPU path. Full S4 experiments can take many hours on CPU.
- Exact runtime and floating-point results depend on PyTorch, hardware, acceleration kernel, and platform. Seeds are recorded, but cross-device bitwise identity is not promised.
- Raw UCI datasets, checkpoints, generated trial CSVs, logs, and runtime state are not versioned. Reproduction requires obtaining the source dataset and rerunning the declared protocol.
- The precise checkpoint used by the upstream WaveStitch authors was unavailable, so MetroTraffic is a technical reference validation rather than exact checkpoint reproduction.

## Research-data boundary

The author's original bachelor-thesis energy/HPC data are proprietary or otherwise non-public. They are not included, and this repository does not reconstruct or simulate them as if they were the original research-centre measurements.
