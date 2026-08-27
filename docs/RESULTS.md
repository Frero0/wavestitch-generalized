# Scientific results

All values in this document are produced by this independent project. Lower MSE, ACD, and xcorrD values are better. Standard deviations use population aggregation over the stated trials. Exact machine-readable values are in `results/`.

## MetroTraffic technical reproduction

| Profile | MSE | ACD | xCorr |
|---|---:|---:|---:|
| C | 0.533067 ± 0.008743 | 0.145402 ± 0.009127 | 0.098706 ± 0.003757 |
| M | 0.323540 ± 0.032583 | 0.113842 ± 0.013006 | 0.121345 ± 0.005321 |
| F | 0.153312 ± 0.008988 | 0.045260 ± 0.001747 | 0.053775 ± 0.009166 |

Each result aggregates five trials. This validates that the retained legacy path can reproduce the expected reference behavior technically. The precise upstream-authors checkpoint was unavailable, so these are not presented as an exact reproduction of their checkpoint or as their reported results.

## UCI Occupancy full experiment

| Metric | Mean | Standard deviation | Trials |
|---|---:|---:|---:|
| Standardized MSE | 3.709588 | 0.056216 | 5 |
| ACD | 0.348028 | 0.034311 | 5 |
| xcorrD | 0.408812 | 0.064319 | 5 |

The pipeline succeeded end to end with checkpoint v2 and restored train-only preprocessing, but output quality was insufficient:

- Temperature, Humidity, and CO2 variances collapsed relative to the test set.
- Humidity was strongly underestimated and remained near the train regime.
- The high CO2 tail was largely absent.
- About 45.65% of synthetic Light values were negative, although the real rate is zero.
- The real Light series has 69.36% exact zeros; all five diffusion trials had 0% exact zeros.
- The `Occupancy`–Light relationship was preserved well, but `Occupancy` relationships with Temperature, Humidity, and CO2 were degraded or inverted.

The chronological train-only split exposed substantial train-to-test distribution shift, especially for Humidity and CO2. This is a property that full-dataset preprocessing can partially obscure but does not cause.

## Empirical conditioned baseline

| Metric | WaveStitch | Empirical baseline | Baseline minus WaveStitch |
|---|---:|---:|---:|
| Standardized MSE | 3.709588 ± 0.056216 | 3.206921 ± 0.487819 | -0.502667 |
| ACD | 0.348028 ± 0.034311 | 0.265747 ± 0.065407 | -0.082281 |
| xcorrD | 0.408812 ± 0.064319 | 0.374113 ± 0.087879 | -0.034699 |

The baseline uses only train signals and the exact test `Occupancy` sequence. It improved the mean of every primary metric, generated no negative Light or CO2 values, and recovered a 52.13% Light zero mass. Yet Humidity and CO2 remained tied to train regimes and could not match the shifted test distribution. Temperature and Light conditioning relationships improved differently from Humidity/CO2, showing that binary occupancy does not uniquely determine the environmental state.

Diagnostic verdict: **`DATA/CONDITIONING LIMIT DOMINANT`**.

## Sampler ablation

| Method | MSE | ACD | xcorrD | Temperature std | Humidity std | CO2 std | Light < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default trial 0 | 3.677873 | 0.396310 | 0.374654 | 0.221047 | 0.915597 | 57.944396 | 40.27% |
| No gradient correction | 3.701941 | 0.382315 | 0.386086 | 0.216149 | 0.849791 | 53.373966 | 41.06% |
| Sqrt posterior variance | 3.764678 | 0.384518 | 0.399147 | 0.286269 | 1.488011 | 58.862692 | 21.50% |

For context, real test standard deviations are 0.842975 for Temperature, 3.683683 for Humidity, and 487.261873 for CO2. Disabling gradient correction had no relevant benefit. Sqrt posterior variance increased Temperature and Humidity dispersion and roughly halved the negative-Light rate versus default trial 0, but slightly worsened MSE/xcorrD and did not recreate any exact Light zeros. Neither ablation addressed the dominant data/conditioning limitation.

Diagnostic verdict: **`SAMPLER CONTRIBUTION MODERATE`**.

The legacy reverse-noise amplitude contributes moderately to under-dispersion, but the evidence is a single trial on one dataset. It does not justify changing the upstream-compatible default.
