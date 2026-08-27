# Roadmap

The initial generalization milestone is complete: configurable flat-CSV datasets, split modes, leakage-free preprocessing, structured v2 checkpoints, experiment configurations, generic training/synthesis, validation, evaluation, and regression tests are implemented.

Potential future work is intentionally limited to capabilities justified by new research questions:

- panel/grouped time-series support with entity-safe windows and splits;
- multi-interval and interlaced temporal evaluation protocols;
- bounded, transformed, or mixed discrete/continuous output models;
- richer time-varying conditioning for regime-shifted datasets;
- portable acceleration and systematic dataset-specific hyperparameter selection.

These items are not commitments and are not required to reproduce the results currently reported.
