#!/usr/bin/env bash

set -euo pipefail
python_bin="${PYTHON_BIN:-python3}"

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
options_encoding=("onehot" "ordinal")
options_dataset=("AustraliaTourism" "MetroTraffic" "RossmanSales" "PanamaEnergy" "BeijingAirQuality")
options_stride=(8)
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for encoding in "${options_encoding[@]}"
  do
    for synth_mask in "${options_synth_mask[@]}"
    do
      for s in "${options_stride[@]}"
      do
        if [[ "$encoding" == "std" ]]; then
          "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride "$s"
        elif [[ "$encoding" == "prop" ]]; then
          "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning -d "$dataset" -synth_mask "$synth_mask" -stride "$s" -propCycEnc True
        elif [[ "$encoding" == "onehot" ]]; then
          "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning_onehot -d "$dataset" -synth_mask "$synth_mask" -stride "$s"
        elif [[ "$encoding" == "ordinal" ]]; then
          "$python_bin" -m scripts.synthesis.synthesis_wavestitch_pipeline_strided_preconditioning_ordinal -d "$dataset" -synth_mask "$synth_mask" -stride "$s"
        fi
      done
    done
  done
done
