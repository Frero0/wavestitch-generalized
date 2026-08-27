# WaveStitch checkpoint format

New checkpoints produced by `scripts/training/training_wavestitch.py` use structured schema v2:

```text
checkpoint_format: "wavestitch"
format_version: 2
state_dict: complete PyTorch model state
dataset_id: configured dataset ID
model:
  in_dim, out_dim
  signal_columns, metadata_columns, encoded_metadata_columns
  model_columns, signal_indices, metadata_indices
  architecture: SSSD/S4 constructor arguments
training:
  window_size, effective_stride, requested_stride
  learning_rate, batch_size, seed, optimizer_steps
diffusion:
  timesteps, beta_0, beta_T
encoding:
  proportional_cyclic_encoding
  cyclic_columns, encoded_metadata_columns
preprocessing:
  mode: train_only | upstream_legacy
  scaler: columns, mean, scale, variance, feature names/count, sample count
  encoders: fitted categories and angles for every cyclic metadata column
dataset_config: serializable DatasetConfig snapshot
```

The DatasetConfig snapshot contains configuration and the CSV path reference,
but never embeds CSV rows or other dataset contents. Compatibility validation
checks dataset identity, semantic configuration, dimensions, exact signal
order, original and encoded metadata, complete model-column order, indices, and
encoding mode before loading weights.

Structured synthesis restores the scaler and cyclic encoders from the
`preprocessing` section before transforming the current dataset. It never fits
either object again. The preprocessing mode is also recorded in the
DatasetConfig snapshot; a checkpoint/config mode mismatch is rejected. A test
category absent from the checkpoint encoder state is rejected explicitly.

## Version behavior

- A raw upstream checkpoint is an unwrapped state-dict mapping without the
  `checkpoint_format` marker. It remains on the legacy loading path and has no
  structured metadata or preprocessing state.
- Structured v1 is the previous envelope schema. It contains model,
  architecture, diffusion, encoding, and dataset metadata, but no fitted
  preprocessing section. It is recognized as v1 and is never interpreted as
  v2. When leakage-free synthesis requests preprocessing state, loading fails
  explicitly and instructs the caller to provide a structured v2 checkpoint.
- Structured v2 adds the complete fitted `preprocessing` section shown above.
  All new training runs write v2, and leakage-free synthesis requires v2.

A v2 envelope with a missing or malformed `preprocessing` section is treated
as a corrupted v2 checkpoint, not downgraded to v1. Other structured version
numbers are unsupported.

For a structured checkpoint, the main synthesis entry point infers the dataset
when `-dataset` is omitted and restores architecture, diffusion schedule,
encoding mode, and training window size. Synthesis-specific choices remain CLI
parameters: `-synth_mask`, synthesis `-stride`, batch size, number of trials,
output directory, and optional smoke window limit.

Legacy upstream checkpoints remain unwrapped state-dict mappings. They are
detected by the absence of the `checkpoint_format: "wavestitch"` marker and are
loaded unchanged. Because they contain no metadata, their dataset and matching
architecture/diffusion arguments must still be supplied through the legacy CLI
path.
