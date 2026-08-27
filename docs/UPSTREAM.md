# Upstream provenance and project boundary

This repository is an independent follow-up to **WaveStitch: Flexible and Fast Conditional Time Series Generation With Diffusion Models**.

- Original repository: <https://github.com/adis98/HierarchicalTS>
- Imported base commit: `dcb1e98eb3bc31f4fa1c0ce3bfef4dcd8e473e47`
- Upstream license: MIT
- Retained license file: `LICENSE`, unchanged
- Retained upstream appendix: `docs/papers/WaveStitch_Appendix.pdf`, unchanged

This is not an official repository of the original authors. Results in this repository were generated during this independent project's validation and must not be attributed to the upstream authors.

## Preserved behavior

The WaveStitch loss, diffusion schedule, SSSD/S4 backbone, conditioning semantics, stitching objective, window construction, and default pipeline sampler are preserved wherever possible. MetroTraffic can use `preprocessing_mode: "upstream_legacy"`, which fits preprocessing on the complete dataset before the split and retains the original numerical layout, row counts, window count, and C/M/F masks.

The default reverse-noise amplitude also remains the legacy upstream behavior. The optional `--sqrt-posterior-variance` switch is a non-default experimental DDPM-style noise-amplitude ablation. The optional `--disable-gradient-correction` switch exists only to isolate the effect of stitching correction. Neither switch changes the default.

The registered S4 buffers `B`, `P`, and `w` are stored contiguously and without overlap so modern PyTorch can serialize them. Their numerical values and model equations are unchanged; save/reload behavior is covered by `tests/test_s4_state_dict_roundtrip.py`.

## Generalized additions

This follow-up adds:

- typed dataset and experiment JSON configurations;
- a generic flat-CSV loader and temporal/schema validation;
- configurable chronological and value-based splits;
- config-driven synthesis conditions;
- leakage-free train-only preprocessing with explicit unseen-category failure;
- structured checkpoint schemas, including v2 fitted preprocessing state;
- checkpoint-driven reconstruction and compatibility checks;
- window, context, mask, and stride preflight validation;
- generic training/synthesis paths for configured datasets;
- UCI Occupancy evaluation and diagnostics;
- empirical conditioned baseline and isolated sampler ablations;
- regression and reproducibility tests.

The legacy upstream research scripts remain in the tree for provenance and comparison. Not every historical entry point has been generalized; the supported generalized path is documented in the main README and `docs/EXPERIMENTS.md`.

## Results boundary

The MetroTraffic tables in `results/` are this project's technical reproduction/reference validation. The precise checkpoint used by the original authors was not available, so exact checkpoint-level equivalence is not claimed. The UCI Occupancy experiment, empirical baseline, failure analysis, and sampler ablations are entirely this project's results.

The original MIT terms remain applicable to the upstream-derived source. The retained `LICENSE` contains the authoritative notice.
