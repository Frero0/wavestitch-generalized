# Dataset configuration

Dataset configuration is defined by JSON files in `configs/datasets/`.
`Preprocessor` consumes the configuration for the generalized training and
pipeline-synthesis entry points. `MetroTraffic` preserves upstream behavior
through its explicit legacy loader and preprocessing mode. Datasets without a
JSON configuration continue through their historical dataset-specific
branches. Config-driven `C/M/F` masks are integrated in
`scripts/synthesis/synthesis_wavestitch_pipeline_strided_preconditioning.py`; other historical
synthesis entry points retain their upstream interfaces.

The schema is:

```json
{
  "dataset_id": "DatasetName",
  "csv_path": "relative/or/absolute/path.csv",
  "loader": "legacy",
  "preprocessing_mode": "train_only",
  "timestamp_column": "timestamp",
  "signal_columns": ["value_a", "value_b"],
  "metadata_columns": ["year", "group"],
  "cyclic_columns": ["year"],
  "dtype_overrides": {"group": "object"},
  "temporal_order": ["year", "timestamp"],
  "split": {
    "mode": "column_values",
    "column": "year",
    "test_values": [2024]
  },
  "synthesis_conditions": {
    "C": {},
    "M": {"year": 2024},
    "F": {"year": 2024, "group": "north"}
  }
}
```

`loader` accepts `legacy` or `flat_csv`. `preprocessing_mode` is required and
accepts exactly `train_only` or `upstream_legacy`. `timestamp_column` and
`dtype_overrides` are optional. `synthesis_conditions` is also optional; the
other fields are required.

## Preprocessing modes

`train_only` is the recommended mode for generalized datasets. Processing is
ordered as follows: load and validate the complete source; compute the
configured split while retaining source indices; fit cyclic metadata encoders
and the signal-column `StandardScaler` on train rows only; transform train and
test with those fitted objects; concatenate them back into the original index
and row order. Non-cyclic numeric metadata is passed through unchanged. No test
row contributes to scaler statistics, category discovery, or proportional
cyclic angles.

If a cyclic metadata category occurs in transform data but was absent during
train fit, preprocessing raises `MetadataCategoryError`. There is no unknown
bucket and no silent coercion because either would change conditioning
semantics.

`upstream_legacy` reproduces the original WaveStitch order: cyclic encoders and
the scaler are fit on the complete dataset before the configured split is
applied. Use it only for upstream comparison and reproduction. In particular,
`MetroTraffic.json` declares this mode to retain its historical values,
dimensions, windows, and masks.

## Synthesis conditions

`synthesis_conditions` may contain only the profiles `C`, `M`, and `F`. Each
profile maps metadata columns to non-null JSON scalar values. Its mask is the
intersection of the test indices already computed by `Preprocessor` with all
declared equality conditions:

```text
test split AND column_1 == value_1 AND ... AND column_n == value_n
```

An empty profile such as `"C": {}` therefore selects the complete test split.
Profiles may be omitted when they have no meaningful definition for a dataset;
requesting an omitted profile raises an explicit error. Condition columns must
be listed in `metadata_columns`. The mask evaluator does not recalculate the
split and does not infer calendar hierarchies from timestamps.

## Train/test split modes

All split modes preserve the existing row order and never shuffle. A split is
rejected when it would produce an empty train or test set.

`column_values` preserves the original behavior: matching values go to the
test set and the complement goes to train.

```json
"split": {
  "mode": "column_values",
  "column": "year",
  "test_values": [2024]
}
```

`ratio` creates a chronological positional split. For `train_ratio: 0.8`, the
first `floor(number_of_rows * 0.8)` rows are train and the remaining final rows
are test. The ratio must be finite and strictly between 0 and 1.

```json
"split": {
  "mode": "ratio",
  "train_ratio": 0.8
}
```

`timestamp` uses the dataset's configured `timestamp_column`. Its boundary is
left-closed for test: rows with `timestamp < cutoff` are train, while rows with
`timestamp >= cutoff` are test. The cutoff is parsed as UTC using the same
timestamp rules as the CSV loader.

```json
"split": {
  "mode": "timestamp",
  "cutoff": "2024-01-01T00:00:00Z"
}
```

Column arrays retain their declared order. Relative CSV paths are interpreted
from the project root. Loading a configuration validates its schema but does
not require the CSV to be installed; call `resolve_csv_path(must_exist=True)`
when source availability must also be checked.

`configs/datasets/MetroTraffic.json` supplies `Preprocessor` with its CSV path,
timestamp, signal and metadata columns, cyclic columns, temporal-order
metadata, and 2018 test split. Its loader remains `legacy`: only these declared
parameters are config-driven, while scaling and cyclic-encoding algorithms
remain the upstream implementations in `wavestitch/data_utils.py`.

`configs/datasets/UCIOccupancyDetection.json` is the complete generalized
example used by the final experiment. It declares four signals, `Occupancy`
as numeric conditioning metadata, a UTC timestamp cutoff, profile C, and
`train_only` preprocessing. The raw UCI file is not included; acquisition and
placement are documented in the main README and `docs/EXPERIMENTS.md`.

## Minimal custom flat CSV

Given an already ordered CSV such as:

```csv
timestamp,load,temperature,year,site
2023-01-01T00:00:00Z,10.0,5.0,2023,A
2023-01-01T00:00:00Z,12.0,6.0,2023,B
2024-01-01T00:00:00Z,18.0,9.0,2024,A
2024-01-01T00:00:00Z,20.0,10.0,2024,B
```

create `configs/datasets/MyDataset.json`:

```json
{
  "dataset_id": "MyDataset",
  "csv_path": "data/my_dataset.csv",
  "loader": "flat_csv",
  "preprocessing_mode": "train_only",
  "timestamp_column": "timestamp",
  "signal_columns": ["load", "temperature"],
  "metadata_columns": ["year", "site"],
  "cyclic_columns": ["year", "site"],
  "dtype_overrides": {"site": "string"},
  "temporal_order": ["timestamp", "site"],
  "split": {
    "mode": "ratio",
    "train_ratio": 0.8
  },
  "synthesis_conditions": {
    "C": {},
    "F": {"site": "A"}
  }
}
```

`Preprocessor("MyDataset", ...)` then parses the timestamp as UTC, validates
that rows are already ordered by the composite `temporal_order` key, and keeps
the configured signal and metadata columns in their declared order. The
timestamp is available as `preprocessor.timestamps` but is not included in the
model columns.

Duplicate timestamps are allowed when another `temporal_order` component makes
the composite key unique, as with different sites above. Duplicate complete
keys, invalid timestamps, missing columns, unsorted rows, and non-numeric
signals are rejected. Non-numeric metadata must currently be included in
`cyclic_columns`. In `train_only`, non-cyclic numeric metadata is not scaled;
only declared signal columns are passed to `StandardScaler`.
