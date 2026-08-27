import numpy as np
import pytest

from data_utils import Preprocessor
from training_utils import resolve_model_columns


def test_uci_occupancy_model_layout_comes_from_dataset_config():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)

    signal_indices, metadata_indices = resolve_model_columns(
        preprocessor.df_cleaned, preprocessor
    )

    np.testing.assert_array_equal(signal_indices, np.arange(4))
    np.testing.assert_array_equal(metadata_indices, np.array([4]))
    assert preprocessor.df_cleaned.columns[signal_indices].tolist() == list(
        preprocessor.dataset_config.signal_columns
    )
    assert preprocessor.df_cleaned.columns[metadata_indices].tolist() == ["Occupancy"]
    assert len(preprocessor.df_cleaned.columns) == 5
    assert len(signal_indices) == 4


def test_configured_model_layout_rejects_undeclared_model_column():
    preprocessor = Preprocessor("UCIOccupancyDetection", False)
    frame = preprocessor.df_cleaned.assign(undeclared=0.0)

    with pytest.raises(ValueError, match="not declared as signal or encoded metadata"):
        resolve_model_columns(frame, preprocessor)
