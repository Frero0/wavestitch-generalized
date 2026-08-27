# Experiment configuration

The main WaveStitch training and pipeline-synthesis entry points accept a
typed JSON experiment configuration. No external configuration dependency is
used. Resolution always follows:

```text
explicit CLI override > experiment config > documented default
```

Unspecified CLI options are deliberately suppressed by `argparse`, so they do
not accidentally replace values read from JSON.

## Schema

```json
{
  "dataset_id": "UCIOccupancyDetection",
  "training": {
    "window_size": 8,
    "stride": 1,
    "batch_size": 2,
    "epochs": 1,
    "learning_rate": 0.0001,
    "seed": 42,
    "max_steps": 2
  },
  "diffusion": {
    "timesteps": 4,
    "beta_0": 0.0001,
    "beta_T": 0.02
  },
  "architecture": {
    "backbone": "S4",
    "hdim": 64,
    "layers": 4,
    "num_res_layers": 1,
    "res_channels": 4,
    "skip_channels": 4,
    "diff_step_embed_in": 8,
    "diff_step_embed_mid": 8,
    "diff_step_embed_out": 8,
    "s4_lmax": 8,
    "s4_dstate": 8,
    "s4_dropout": 0.0,
    "s4_bidirectional": true,
    "s4_layernorm": true,
    "proportional_cyclic_encoding": false
  },
  "artifacts": {
    "checkpoint_path": "saved_models/example/model.pth",
    "output_dir": "generated"
  },
  "synthesis": {
    "stride": 1,
    "profile": "C",
    "trials": 1,
    "batch_size": 2,
    "max_windows": 1
  }
}
```

`dataset_id` is required. Every other field is optional and inherits its
default. Integers representing sizes, counts, strides, epochs, or timesteps
must be positive; `seed` is non-negative; learning rate is positive;
`0 < beta_0 < beta_T < 1`; S4 dropout is in `[0, 1)`; and synthesis profile is
one of `C`, `M`, or `F`. `max_steps` and `max_windows` may be omitted for an
unbounded run.

The defaults preserve the former main-script CLI values: window 32, requested
training stride 1, training/synthesis batch size 1024, 1000 epochs, learning
rate `1e-4`, seed 42, 200 diffusion steps, betas `1e-4` and `0.02`, S4 with 64
residual/skip channels, four residual layers, and five synthesis trials. The
default synthesis stride is 1 and no synthesis profile is assumed. Training
continues to use the validated upstream **effective stride 1**; a different
requested training stride is recorded and warned about but does not change
the numerical windowing behavior.

Repository examples are provided in
`configs/experiments/metrotraffic_upstream.json` and
`configs/experiments/uci_occupancy_smoke.json`. The final UCI protocol is kept
separately in `configs/experiments/uci_occupancy_full.json`. The MetroTraffic
file selects profile C; use `--synthesis-profile M` or `F` as an explicit
override for the other configured upstream conditions.

## Commands

Training from JSON:

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_smoke.json
```

An explicit CLI override wins over the file:

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_smoke.json \
  --epochs 2 --max-steps 4
```

Synthesis reads model dimensions, architecture, encoding, window size, and
diffusion parameters from a structured checkpoint:

```bash
python -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning \
  --experiment-config configs/experiments/uci_occupancy_smoke.json
```

An experiment or CLI value explicitly supplied for a checkpoint-owned model
field must equal the structured checkpoint value. A mismatch is an error; the
experiment file cannot silently reconfigure the model. Legacy state-dict-only
checkpoints remain supported by supplying the dataset and former architecture
arguments through the legacy CLI.

Dry-run performs dataset loading, preprocessing layout resolution, train/test
and window validation, synthesis-context and mask validation, and (when a
checkpoint is supplied to synthesis) structured-checkpoint compatibility. It
prints the resolved device, dimensions, columns, row/window counts, optimizer,
diffusion setup, and artifact destinations, then exits before model execution:

```bash
python -m scripts.training.training_wavestitch \
  --experiment-config configs/experiments/uci_occupancy_smoke.json \
  --dry-run
```

## Sampler ablation flags

The generalized pipeline-synthesis CLI exposes two mutually exclusive
diagnostic flags:

- `--disable-gradient-correction` disables only the stitching-gradient update;
- `--sqrt-posterior-variance` retains gradient correction but uses the
  DDPM-style square-root posterior-variance noise amplitude.

Neither flag is read from the experiment JSON because both are deliberately
run-specific ablations. With neither flag present, the upstream-compatible
default remains unchanged: gradient correction is enabled and reverse noise
uses the legacy posterior-variance amplitude. The first flag is not a
recommended improvement; the second remains experimental and non-default.
