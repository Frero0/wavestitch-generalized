import numpy as np
import pandas as pd

from scripts.evaluation.empirical_conditioned_baseline import conditioned_run_block_bootstrap


def _train_frame():
    occupancy = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    return pd.DataFrame(
        {
            "Temperature": np.arange(10, dtype=float),
            "Humidity": np.arange(10, dtype=float) + 100,
            "Light": np.arange(10, dtype=float) + 200,
            "CO2": np.arange(10, dtype=float) + 300,
            "Occupancy": occupancy,
        },
        index=np.arange(20, 30),
    )


def test_bootstrap_preserves_exact_condition_and_uses_compatible_train_rows():
    train = _train_frame()
    condition = pd.Series([0, 0, 1, 1, 1, 0, 0])

    generated, provenance = conditioned_run_block_bootstrap(
        train, condition, block_size=2, seed=42
    )

    np.testing.assert_array_equal(generated["Occupancy"], condition)
    assert len(generated) == len(condition)
    assert sum(block["length"] for block in provenance) == len(condition)
    assert all(20 <= block["source_original_index_start"] <= 29 for block in provenance)
    assert all(20 <= block["source_original_index_end"] <= 29 for block in provenance)

    for block in provenance:
        source = train.loc[
            block["source_original_index_start"]:block["source_original_index_end"]
        ]
        assert source["Occupancy"].eq(block["occupancy"]).all()


def test_bootstrap_is_reproducible_for_a_seed():
    train = _train_frame()
    condition = pd.Series([0, 0, 1, 1, 0, 0])

    first, first_provenance = conditioned_run_block_bootstrap(
        train, condition, block_size=2, seed=7
    )
    second, second_provenance = conditioned_run_block_bootstrap(
        train, condition, block_size=2, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_provenance == second_provenance
