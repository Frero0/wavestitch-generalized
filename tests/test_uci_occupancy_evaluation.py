import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.evaluate_uci_occupancy_full import (
    EvaluationValidationError,
    acd,
    standardized_mse,
    xcorr_distance,
)


def test_wave_stitch_metrics_identical_signal_frames_are_zero():
    values = np.arange(480, dtype=float).reshape(120, 4)
    real = pd.DataFrame(values, columns=["Temperature", "Humidity", "Light", "CO2"])

    assert standardized_mse(real, real.copy()) == 0.0
    assert acd(real, real.copy()) == pytest.approx(0.0, abs=1e-15)
    assert xcorr_distance(real, real.copy()) == pytest.approx(0.0, abs=1e-15)


def test_standardized_mse_uses_only_explicitly_passed_signals():
    real = pd.DataFrame({"Temperature": [0.0, 1.0], "Humidity": [1.0, 2.0]})
    synthetic = pd.DataFrame({"Temperature": [1.0, 2.0], "Humidity": [2.0, 3.0]})

    assert standardized_mse(real, synthetic) == 1.0


def test_acd_rejects_all_constant_signals():
    real = pd.DataFrame(np.ones((120, 4)))

    with pytest.raises(EvaluationValidationError, match="all signals are constant"):
        acd(real, real.copy())


def test_xcorr_rejects_constant_signal():
    real = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})

    with pytest.raises(EvaluationValidationError, match="constant"):
        xcorr_distance(real, real.copy())
