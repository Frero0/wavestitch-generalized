#!/usr/bin/env bash

set -euo pipefail
python_bin="${PYTHON_BIN:-python3}"

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
options_encoding=("std")
options_dataset=("AustraliaTourism" "MetroTraffic" "BeijingAirQuality" "RossmanSales" "PanamaEnergy")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for synth_mask in "${options_synth_mask[@]}"
  do
#    "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride 1
    "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride 8
#    "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride 16
#    "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride 32
  done
done
