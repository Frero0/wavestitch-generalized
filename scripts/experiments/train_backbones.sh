#!/usr/bin/env bash

set -euo pipefail
python_bin="${PYTHON_BIN:-python3}"

options_dataset=("AustraliaTourism" "MetroTraffic" "BeijingAirQuality" "RossmanSales" "PanamaEnergy")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  "$python_bin" -m scripts.training.training_wavestitch_ordinal -d "$dataset" -epochs 300
  "$python_bin" -m scripts.training.training_wavestitch_onehot -d "$dataset" -epochs 300
  "$python_bin" -m scripts.training.training_wavestitch -d "$dataset" -epochs 300
  "$python_bin" -m scripts.training.training_timegan -d "$dataset" -epochs 300
done
