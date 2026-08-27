import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from wavestitch.dataset_config import DatasetConfigNotFoundError, load_dataset_config_by_id

pd.set_option('future.no_silent_downcasting', True)

datasets = {"WebTraffic": "WebTrafficLAcity/lacity.org-website-traffic.csv",
            "RossmanSales": "RossmanSales/train.csv",
            "AustraliaTourism": "QuarterlyTourismAustralia/tourism.csv",
            "MetroTraffic": "MetroInterstateTrafficVolume/Metro_Interstate_Traffic_Volume.csv/Metro_Interstate_Traffic_Volume.csv",
            "BeijingAirQuality": "BeijingAirQuality/beijing+multi+site+air+quality+data",
            "PanamaEnergy": "PanamaEnergy/continuous dataset.csv"}


def _plain_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


class FlatCSVValidationError(ValueError):
    """Raised when a flat CSV is incompatible with its dataset configuration."""


class DatasetSplitError(ValueError):
    """Raised when a configured split is invalid for the loaded dataset."""


class MetadataCategoryError(ValueError):
    """Raised when transform data contains metadata absent from encoder fit data."""


class PreprocessingStateError(ValueError):
    """Raised when serialized preprocessing state is missing or incompatible."""


class CyclicEncoder:

    def __init__(self, name, df, propCycEnc):
        self.column_name = name
        self.categories = df[name].unique()
        counts = df[name].value_counts(dropna=False)
        total_counts = counts.sum()
        angles = (counts / total_counts) * 2 * np.pi
        cumulative_angles = angles.cumsum() - (angles / 2)
        temp = counts.index.values
        """
        counts = df[name].value_counts(dropna=False)

        # Step 2: Calculate the proportional angles (in radians)
        total_counts = counts.sum()
        angles = (counts / total_counts) * 2 * np.pi  # Proportional angles in radians

        # Step 3: Calculate the cumulative angle positions
        cumulative_angles = angles.cumsum() - (angles / 2)
        """
        self.categories = temp
        if propCycEnc:
            self.angles = cumulative_angles
        else:
            self.angles = np.array(list(range(len(self.categories)))) * (2 * np.pi) / len(self.categories)
        self.mapper = dict(zip(self.categories, self.angles))
        self.mapper_sine = dict(zip(self.categories, np.sin(self.angles)))
        self.mapper_cosine = dict(zip(self.categories, np.cos(self.angles)))
        self.angles_to_cat = dict(zip(self.angles, self.categories))

    def encode(self, df):
        df_copy = df.copy()
        unseen_mask = ~df_copy[self.column_name].isin(self.categories)
        if unseen_mask.any():
            unseen = df_copy.loc[unseen_mask, self.column_name].drop_duplicates().tolist()
            raise MetadataCategoryError(
                "Metadata column {!r} contains category value(s) not seen while fitting "
                "the train encoder: {}.".format(self.column_name, unseen)
            )
        # Keep the upstream replacement operation exactly; the explicit check
        # above only makes unknown-category behavior safe and deterministic.
        df_copy[self.column_name + "_sine"] = df_copy[self.column_name].replace(self.mapper_sine).astype(float)
        df_copy[self.column_name + "_cos"] = df_copy[self.column_name].replace(self.mapper_cosine).astype(float)
        df_copy.drop(columns=[self.column_name], inplace=True)
        return df_copy

    def to_state(self):
        return {
            "column_name": self.column_name,
            "categories": [_plain_scalar(value) for value in self.categories],
            "angles": [float(value) for value in np.asarray(self.angles)],
        }

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict):
            raise PreprocessingStateError("Cyclic encoder state must be a mapping.")
        required = {"column_name", "categories", "angles"}
        missing = sorted(required - set(state))
        if missing:
            raise PreprocessingStateError(
                "Cyclic encoder state is missing field(s): {}.".format(
                    ", ".join(missing)
                )
            )
        categories = np.asarray(state["categories"])
        angles = np.asarray(state["angles"], dtype=float)
        if len(categories) == 0 or len(categories) != len(angles):
            raise PreprocessingStateError(
                "Cyclic encoder categories and angles must have equal non-zero length."
            )
        encoder = cls.__new__(cls)
        encoder.column_name = state["column_name"]
        encoder.categories = categories
        encoder.angles = angles
        encoder.mapper = dict(zip(categories, angles))
        encoder.mapper_sine = dict(zip(categories, np.sin(angles)))
        encoder.mapper_cosine = dict(zip(categories, np.cos(angles)))
        encoder.angles_to_cat = dict(zip(angles, categories))
        return encoder

    def decode(self, df):
        df_copy = df.copy()
        df_copy[self.column_name + "_sine"] = np.clip(df_copy[self.column_name + "_sine"], -1, 1)
        df_copy[self.column_name + "_cos"] = np.clip(df_copy[self.column_name + "_cos"], -1, 1)
        df_copy[self.column_name + "_angle"] = np.nan
        condition1 = np.logical_and(df_copy[self.column_name + "_sine"] >= 0, df_copy[self.column_name + "_cos"] > 0)
        condition2 = np.logical_and(df_copy[self.column_name + "_sine"] > 0, df_copy[self.column_name + "_cos"] <= 0)
        condition3 = np.logical_and(df_copy[self.column_name + "_sine"] <= 0, df_copy[self.column_name + "_cos"] < 0)
        condition4 = np.logical_and(df_copy[self.column_name + "_sine"] < 0, df_copy[self.column_name + "_cos"] >= 0)

        df_copy.loc[condition1, self.column_name + "_angle"] = (np.arcsin(df_copy[self.column_name + "_sine"].values)[
                                                                    condition1.values] +
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition1.values]) / 2

        df_copy.loc[condition2, self.column_name + "_angle"] = (np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition2.values] +
                                                                np.pi -
                                                                np.arcsin(df_copy[self.column_name + "_sine"].values)[
                                                                    condition2.values]) / 2
        df_copy.loc[condition3, self.column_name + "_angle"] = (2 * np.pi -
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition3.values] +
                                                                np.pi - np.arcsin(
                    df_copy[self.column_name + "_sine"].values)[condition3.values]) / 2
        df_copy.loc[condition4, self.column_name + "_angle"] = (4 * np.pi -
                                                                np.arccos(df_copy[self.column_name + "_cos"].values)[
                                                                    condition4.values] + np.arcsin(
                    df_copy[self.column_name + "_sine"].values)[condition4.values]) / 2

        df_copy[self.column_name + "_angle"] = df_copy[self.column_name + "_angle"] % (2 * np.pi)
        df_copy[self.column_name + '_threshold_angle'] = df_copy[self.column_name + "_angle"].apply(
            lambda x: self.nearest_threshold(x, self.angles))
        df_copy[self.column_name] = df_copy[self.column_name + '_threshold_angle'].replace(self.angles_to_cat)
        df_copy.drop(columns=[self.column_name + '_sine', self.column_name + '_cos', self.column_name + '_angle',
                              self.column_name + '_threshold_angle'], inplace=True)
        return df_copy

    @staticmethod
    def nearest_threshold(x, thresholds):
        return min(thresholds, key=lambda t: abs(t - x))


class Preprocessor:
    def __init__(self, name, propCycEnc, preprocessing_state=None):
        self.pce = propCycEnc
        self.cols_to_scale = None
        self.cyclic_encoded_columns = None
        self.encoders = {}
        self.hierarchical_features_uncyclic = []
        self.hierarchical_features_cyclic = []
        self.dataset_config = self._load_dataset_config(name)
        self.signal_columns = []
        self.metadata_columns = []
        self.timestamp_column = None
        self.temporal_order = []
        self.timestamps = None
        self.preprocessing_mode = "upstream_legacy"
        if self.dataset_config is not None:
            self.signal_columns = list(self.dataset_config.signal_columns)
            self.metadata_columns = list(self.dataset_config.metadata_columns)
            self.timestamp_column = self.dataset_config.timestamp_column
            self.temporal_order = list(self.dataset_config.temporal_order)
            self.hierarchical_features_uncyclic = list(self.metadata_columns)
            self.cyclic_encoded_columns = list(self.dataset_config.cyclic_columns)
            self.preprocessing_mode = self.dataset_config.preprocessing_mode
        self.scaler = StandardScaler()
        self.df_orig = self.fetchDataset(name, False)
        self.column_dtypes = self.df_orig.dtypes.to_dict()
        self.train_indices = None
        self.test_indices = None
        if preprocessing_state is not None:
            self._restore_preprocessing_state(preprocessing_state)
            self._determine_split(name)
            self.df_cleaned = self.cleanDataset(name, self.df_orig, fit=False)
        elif self.preprocessing_mode == "train_only":
            self._determine_split(name)
            train = self.cleanDataset(
                name, self.df_orig.loc[self.train_indices], fit=True
            )
            test = self.cleanDataset(
                name, self.df_orig.loc[self.test_indices], fit=False
            )
            self.df_cleaned = pd.concat([train, test]).loc[self.df_orig.index]
        else:
            # Exact upstream order: fit/transform the complete frame, then split.
            self.df_cleaned = self.cleanDataset(name, self.df_orig, fit=True)
            self._determine_split(name)
        self._set_encoded_metadata_layout()

    def _determine_split(self, name):
        if self.dataset_config is not None:
            self._apply_configured_split()
        elif name == "MetroTraffic":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2018])].to_list()
            self.train_indices = self.df_orig.index[self.df_orig['year'] != 2018].to_list()
        elif name == "BeijingAirQuality":
            temp = self.df_orig['year'].isin([2017])
            self.test_indices = temp.loc[temp].index.to_list()
            temp_c = ~temp
            self.train_indices = temp_c.loc[temp_c].index.to_list()
        elif name == "AustraliaTourism":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2016])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2016])].to_list()
        elif name == "RossmanSales":
            self.test_indices = self.df_orig.index[self.df_orig['Year'].isin([2015])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['Year'].isin([2015])].to_list()
        elif name == "PanamaEnergy":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2020])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2020])].to_list()

    def _set_encoded_metadata_layout(self):
        self.hierarchical_features_cyclic = []
        for column in self.hierarchical_features_uncyclic:
            if self.dataset_config is not None and self.dataset_config.loader == 'flat_csv':
                if column in self.cyclic_encoded_columns:
                    self.hierarchical_features_cyclic.extend(
                        [column + '_sine', column + '_cos']
                    )
                else:
                    self.hierarchical_features_cyclic.append(column)
            else:
                self.hierarchical_features_cyclic.extend(
                    [column + '_sine', column + '_cos']
                )

    def preprocessing_state_dict(self):
        if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
            raise PreprocessingStateError("Scaler has not been fitted.")
        n_samples_seen = self.scaler.n_samples_seen_
        if isinstance(n_samples_seen, np.ndarray):
            n_samples_seen = n_samples_seen.tolist()
        else:
            n_samples_seen = _plain_scalar(n_samples_seen)
        return {
            "mode": self.preprocessing_mode,
            "scaler": {
                "columns": list(self.cols_to_scale),
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(self.scaler.n_features_in_),
                "feature_names_in": list(
                    getattr(self.scaler, "feature_names_in_", self.cols_to_scale)
                ),
                "n_samples_seen": n_samples_seen,
            },
            "encoders": {
                column: encoder.to_state()
                for column, encoder in self.encoders.items()
            },
        }

    def _restore_preprocessing_state(self, state):
        if not isinstance(state, dict):
            raise PreprocessingStateError("Preprocessing state must be a mapping.")
        if state.get("mode") != self.preprocessing_mode:
            raise PreprocessingStateError(
                "Checkpoint preprocessing mode {!r} is incompatible with dataset "
                "configuration mode {!r}.".format(
                    state.get("mode"), self.preprocessing_mode
                )
            )
        scaler_state = state.get("scaler")
        encoder_states = state.get("encoders")
        if not isinstance(scaler_state, dict) or not isinstance(encoder_states, dict):
            raise PreprocessingStateError(
                "Preprocessing state requires scaler and encoders mappings."
            )
        required_scaler = {
            "columns", "mean", "scale", "var", "n_features_in",
            "feature_names_in", "n_samples_seen"
        }
        missing = sorted(required_scaler - set(scaler_state))
        if missing:
            raise PreprocessingStateError(
                "Scaler state is missing field(s): {}.".format(", ".join(missing))
            )
        expected_columns = (
            list(self.signal_columns)
            if self.preprocessing_mode == "train_only"
            else list(scaler_state["columns"])
        )
        if list(scaler_state["columns"]) != expected_columns:
            raise PreprocessingStateError(
                "Scaler columns are incompatible with preprocessing mode {!r}: {}.".format(
                    self.preprocessing_mode, scaler_state["columns"]
                )
            )
        feature_count = len(scaler_state["columns"])
        if (
            scaler_state["n_features_in"] != feature_count
            or list(scaler_state["feature_names_in"]) != list(scaler_state["columns"])
            or any(
                len(scaler_state[field]) != feature_count
                for field in ("mean", "scale", "var")
            )
        ):
            raise PreprocessingStateError(
                "Scaler state dimensions or feature names are inconsistent."
            )
        if set(encoder_states) != set(self.cyclic_encoded_columns):
            raise PreprocessingStateError(
                "Checkpoint encoder columns {} do not match configured cyclic columns {}."
                .format(sorted(encoder_states), sorted(self.cyclic_encoded_columns))
            )
        self.cols_to_scale = list(scaler_state["columns"])
        self.scaler.mean_ = np.asarray(scaler_state["mean"], dtype=float)
        self.scaler.scale_ = np.asarray(scaler_state["scale"], dtype=float)
        self.scaler.var_ = np.asarray(scaler_state["var"], dtype=float)
        self.scaler.n_features_in_ = int(scaler_state["n_features_in"])
        self.scaler.feature_names_in_ = np.asarray(
            scaler_state["feature_names_in"], dtype=object
        )
        self.scaler.n_samples_seen_ = np.asarray(scaler_state["n_samples_seen"])
        self.encoders = {
            column: CyclicEncoder.from_state(encoder_states[column])
            for column in self.cyclic_encoded_columns
        }
        for column, encoder in self.encoders.items():
            if encoder.column_name != column:
                raise PreprocessingStateError(
                    "Encoder state key {!r} contains column_name {!r}.".format(
                        column, encoder.column_name
                    )
                )

    @staticmethod
    def _load_dataset_config(name):
        try:
            return load_dataset_config_by_id(name)
        except DatasetConfigNotFoundError:
            return None

    def _dataset_path(self, name):
        if self.dataset_config is not None:
            return self.dataset_config.resolve_csv_path()
        return datasets[name]

    def _apply_configured_split(self):
        split = self.dataset_config.split

        if split.mode == 'column_values':
            if split.column == self.timestamp_column and self.timestamps is not None:
                split_values = pd.to_datetime(
                    list(split.test_values), errors='coerce', utc=True, format='mixed'
                )
                if split_values.isna().any():
                    raise DatasetSplitError(
                        "column_values split contains an invalid timestamp value."
                    )
                test_mask = self.timestamps.isin(split_values)
            else:
                test_mask = self.df_orig[split.column].isin(split.test_values)
            test_indices = self.df_orig.index[test_mask].to_list()
            train_indices = self.df_orig.index[~test_mask].to_list()

        elif split.mode == 'ratio':
            boundary = int(len(self.df_orig) * split.train_ratio)
            train_indices = self.df_orig.index[:boundary].to_list()
            test_indices = self.df_orig.index[boundary:].to_list()

        elif split.mode == 'timestamp':
            if self.timestamps is None:
                raise DatasetSplitError(
                    "timestamp split requires parsed timestamps from the configured timestamp_column."
                )
            cutoff = pd.to_datetime(
                [split.cutoff], errors='coerce', utc=True, format='mixed'
            )
            if cutoff.isna().any():
                raise DatasetSplitError(
                    "Invalid timestamp split cutoff: {!r}.".format(split.cutoff)
                )
            cutoff_value = cutoff[0]
            train_mask = self.timestamps < cutoff_value
            test_mask = self.timestamps >= cutoff_value
            train_indices = self.df_orig.index[train_mask].to_list()
            test_indices = self.df_orig.index[test_mask].to_list()

        else:
            raise DatasetSplitError(
                "Unsupported runtime split mode: {!r}.".format(split.mode)
            )

        if not train_indices or not test_indices:
            raise DatasetSplitError(
                "Split mode {!r} produced an empty {} set (train={}, test={}).".format(
                    split.mode,
                    'train' if not train_indices else 'test',
                    len(train_indices),
                    len(test_indices),
                )
            )

        self.train_indices = train_indices
        self.test_indices = test_indices

    def _fetch_flat_csv(self, dataset_path):
        if self.timestamp_column is None:
            raise FlatCSVValidationError(
                "flat_csv requires a configured timestamp_column."
            )
        if not self.temporal_order:
            raise FlatCSVValidationError(
                "flat_csv requires at least one temporal_order column."
            )
        if self.timestamp_column not in self.temporal_order:
            raise FlatCSVValidationError(
                "temporal_order must include timestamp_column {!r}.".format(
                    self.timestamp_column
                )
            )
        if self.timestamp_column in self.signal_columns + self.metadata_columns:
            raise FlatCSVValidationError(
                "timestamp_column must be separate from signal_columns and metadata_columns."
            )

        read_csv_kwargs = {}
        if self.dataset_config.dtype_overrides:
            read_csv_kwargs['dtype'] = dict(self.dataset_config.dtype_overrides)
        try:
            df = pd.read_csv(dataset_path, **read_csv_kwargs)
        except (TypeError, ValueError) as exc:
            raise FlatCSVValidationError(
                "Could not read flat CSV {!s} with the configured dtype_overrides: {}".format(
                    dataset_path, exc
                )
            ) from exc

        required_columns = set(
            [self.timestamp_column]
            + self.signal_columns
            + self.metadata_columns
            + self.temporal_order
            + list(self.dataset_config.dtype_overrides)
        )
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise FlatCSVValidationError(
                "flat_csv is missing required column(s): {}.".format(
                    ", ".join(missing_columns)
                )
            )

        parsed_timestamps = pd.to_datetime(
            df[self.timestamp_column], errors='coerce', utc=True, format='mixed'
        )
        invalid_timestamps = parsed_timestamps.isna()
        if invalid_timestamps.any():
            invalid_rows = df.index[invalid_timestamps].to_list()
            examples = df.loc[invalid_timestamps, self.timestamp_column].head(3).to_list()
            raise FlatCSVValidationError(
                "Invalid timestamp value(s) in column {!r} at row(s) {}. Examples: {}.".format(
                    self.timestamp_column, invalid_rows, examples
                )
            )
        df[self.timestamp_column] = parsed_timestamps

        null_temporal_keys = df[self.temporal_order].isna().any(axis=1)
        if null_temporal_keys.any():
            rows = df.index[null_temporal_keys].to_list()
            raise FlatCSVValidationError(
                "temporal_order column(s) contain missing values at row(s) {}.".format(
                    rows
                )
            )

        duplicate_keys = df.duplicated(subset=self.temporal_order, keep=False)
        if duplicate_keys.any():
            rows = df.index[duplicate_keys].to_list()
            raise FlatCSVValidationError(
                "Duplicate temporal key(s) for temporal_order {} at row(s) {}.".format(
                    self.temporal_order, rows
                )
            )

        sorted_indices = df.sort_values(
            by=self.temporal_order, kind='stable'
        ).index.to_numpy()
        if not np.array_equal(sorted_indices, df.index.to_numpy()):
            raise FlatCSVValidationError(
                "CSV rows are not ordered by temporal_order {}. Row order is preserved; "
                "sort the source CSV or change the configuration.".format(
                    self.temporal_order
                )
            )

        non_numeric_signals = [
            column
            for column in self.signal_columns
            if not is_numeric_dtype(df[column].dtype) or is_bool_dtype(df[column].dtype)
        ]
        if non_numeric_signals:
            raise FlatCSVValidationError(
                "flat_csv signal column(s) must be numeric: {}.".format(
                    ", ".join(non_numeric_signals)
                )
            )

        non_cyclic_metadata = [
            column
            for column in self.metadata_columns
            if column not in self.cyclic_encoded_columns
        ]
        unsupported_metadata = [
            column
            for column in non_cyclic_metadata
            if not is_numeric_dtype(df[column].dtype) or is_bool_dtype(df[column].dtype)
        ]
        if unsupported_metadata:
            raise FlatCSVValidationError(
                "Non-numeric metadata must be listed in cyclic_columns: {}.".format(
                    ", ".join(unsupported_metadata)
                )
            )

        self.timestamps = df[self.timestamp_column].copy()
        self.hierarchical_features_uncyclic = list(self.metadata_columns)
        return df[self.signal_columns + self.metadata_columns].copy()

    def fetchDataset(self, name, return_cleaned):
        if self.dataset_config is not None and self.dataset_config.loader == 'flat_csv':
            df = self._fetch_flat_csv(self._dataset_path(name))
        elif name != "BeijingAirQuality":
            dataset_path = self._dataset_path(name)
            if name == "RossmanSales":
                df = pd.read_csv(dataset_path, dtype={'StateHoliday': 'object'})
            else:
                read_csv_kwargs = {}
                if self.dataset_config is not None and self.dataset_config.dtype_overrides:
                    read_csv_kwargs['dtype'] = dict(self.dataset_config.dtype_overrides)
                df = pd.read_csv(dataset_path, **read_csv_kwargs)
            if name == "MetroTraffic":
                if self.dataset_config is not None:
                    timestamp_column = self.timestamp_column
                    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                    datetime_values = df[timestamp_column].dt
                    temporal_components = {
                        'year': datetime_values.year,
                        'month': datetime_values.month,
                        'day': datetime_values.day,
                        'hour': datetime_values.hour,
                    }
                    for column in self.metadata_columns:
                        df[column] = temporal_components[column]
                    df = df[self.signal_columns + self.metadata_columns]
                    self.hierarchical_features_uncyclic = list(self.metadata_columns)
                else:
                    df['date_time'] = pd.to_datetime(df['date_time'])
                    df['year'] = df['date_time'].dt.year
                    df['month'] = df['date_time'].dt.month
                    df['day'] = df['date_time'].dt.day
                    df['hour'] = df['date_time'].dt.hour
                    df.drop(columns=['date_time', 'weather_main', 'weather_description', 'holiday'], inplace=True)
                    self.hierarchical_features_uncyclic = ['year', 'month', 'day', 'hour']
            elif name == "AustraliaTourism":
                df['date_time'] = pd.to_datetime(df['Quarter'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time', 'day', 'hour', 'Quarter', 'Unnamed: 0'], inplace=True)
                df = df.sort_values(by=['year', 'month', 'State', 'Region', 'Purpose']).reset_index(drop=True)
                self.hierarchical_features_uncyclic = ['year', 'month', 'State', 'Region', 'Purpose']
            elif name == "RossmanSales":
                store_ids = df['Store'].unique()[:10]
                df = df[(df['Store'].isin(store_ids)) & (df['Open'] == 1)]

                # Step 2: Plot sales data for each StoreID with different colors
                # df = filtered_df.copy()
                df['Datetime'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Datetime'].dt.year
                df['Month'] = df['Datetime'].dt.month
                df['Day'] = df['Datetime'].dt.day
                df.drop(columns=['Datetime', 'Promo', 'Open'], inplace=True)
                df = df.sort_values(by=['Year', 'Month', 'Day', 'Store'], ignore_index=True)
                df = df[['Year', 'Month', 'Day', 'Store', 'Sales', 'Customers']]
                self.hierarchical_features_uncyclic = ['Year', 'Month', 'Day', 'Store']
            elif name == "PanamaEnergy":
                df = df.drop(columns=['nat_demand', 'Holiday_ID', 'holiday', 'school'])

                # Create a multi-index by city and weather parameter
                # We melt the dataframe to unpivot the city-specific columns and create a 'city' column
                df = pd.melt(df,
                             id_vars=['datetime'],
                             value_vars=['T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
                                         'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san',
                                         'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav'],
                             var_name='variable',
                             value_name='value')

                # Split 'variable' column into 'city' and 'parameter'
                df['city'] = df['variable'].str.split('_').str[-1]
                df['parameter'] = df['variable'].str.split('_').str[0]

                # Pivot the dataframe to get the parameters as columns and city as a column
                df = df.pivot_table(index=['datetime', 'city'],
                                    columns='parameter',
                                    values='value').reset_index()

                # Rearranging columns (optional)
                df['date'] = pd.to_datetime(df['datetime'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['hour'] = df['date'].dt.hour
                df.drop(columns=['date', 'datetime'], inplace=True)
                df = df[['year', 'month', 'day', 'hour', 'city', 'T2M', 'TQL', 'W2M', 'QV2M']]
                df = df.sort_values(by=['year', 'month', 'day', 'hour', 'city'], ignore_index=True)
                self.hierarchical_features_uncyclic = ['year', 'month', 'day', 'hour', 'city']
        else:
            dfs = []
            csvs = os.listdir(datasets[name])
            csvs.sort()
            for file in csvs[:6]:
                dfs.append(pd.read_csv(datasets[name] + "/" + file))
            df = pd.concat(dfs, ignore_index=True)
            df.drop(columns=['No', 'wd'], inplace=True)  # redundant
            self.hierarchical_features_uncyclic = ['year', 'station', 'month', 'day', 'hour']
            df = df.sort_values(by=self.hierarchical_features_uncyclic).reset_index(drop=True)

        if return_cleaned:
            df_cleaned = self.cleanDataset(name, df)
            if self.dataset_config is not None and self.dataset_config.loader == 'flat_csv':
                for col in self.hierarchical_features_uncyclic:
                    if col in self.cyclic_encoded_columns:
                        self.hierarchical_features_cyclic.append(col + '_sine')
                        self.hierarchical_features_cyclic.append(col + '_cos')
                    else:
                        self.hierarchical_features_cyclic.append(col)
            else:
                for col in self.hierarchical_features_uncyclic:
                    self.hierarchical_features_cyclic.append(col + '_sine')
                    self.hierarchical_features_cyclic.append(col + '_cos')
            return df_cleaned

        else:
            return df

    def cleanDataset(self, name, df, fit=None):
        """Beijing Air Quality has some missing values for the sensor data"""
        if fit is None:
            fit = not (hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'))
        df_clean = df.copy()
        if name == "BeijingAirQuality":
            for column in df_clean.columns:
                if df_clean[column].dtype != 'object':
                    df_clean[column] = df_clean[column].interpolate()
            if self.cyclic_encoded_columns is None:
                self.cyclic_encoded_columns = ['year', 'month', 'day', 'hour', 'station']

        elif name == 'MetroTraffic':
            if self.cyclic_encoded_columns is None:
                self.cyclic_encoded_columns = ['year', 'month', 'day', 'hour']
        elif name == "AustraliaTourism":
            if self.cyclic_encoded_columns is None:
                self.cyclic_encoded_columns = ['State', 'Region', 'Purpose', 'year', 'month']
        elif name == "RossmanSales":
            if self.cyclic_encoded_columns is None:
                self.cyclic_encoded_columns = ['Year', 'Month', 'Day', 'Store']
        elif name == "PanamaEnergy":
            if self.cyclic_encoded_columns is None:
                self.cyclic_encoded_columns = ['year', 'month', 'day', 'hour', 'city']

        df_cyclic = self.cyclicEncode(
            df_clean, fit=fit
        )  # returns the dataframe with cyclic encoding applied

        if self.cols_to_scale is None:
            if self.preprocessing_mode == 'train_only':
                self.cols_to_scale = list(self.signal_columns)
            else:
                self.cols_to_scale = [col for col in df_cyclic.columns if
                                      col not in self.cyclic_encoded_columns and '_sine' not in col and '_cos' not in col]

        if fit:
            df_cyclic[self.cols_to_scale] = self.scaler.fit_transform(df_cyclic[self.cols_to_scale])
        else:
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                raise PreprocessingStateError(
                    "Cannot transform before the scaler has been fitted or restored."
                )
            df_cyclic[self.cols_to_scale] = self.scaler.transform(df_cyclic[self.cols_to_scale])
        return df_cyclic

    def cyclicEncode(self, df, fit=None):
        if fit is None:
            fit = not self.encoders
        df_copy = df.copy()
        for column in self.cyclic_encoded_columns:
            if fit and column not in self.encoders:
                self.encoders[column] = CyclicEncoder(column, df_copy, self.pce)
            if column not in self.encoders:
                raise PreprocessingStateError(
                    "Cannot transform cyclic metadata column {!r} before its encoder "
                    "has been fitted or restored.".format(column)
                )
            df_copy = self.encoders[column].encode(df_copy)
        return df_copy

    def cyclicDecode(self, df):
        df_copy = df.copy()
        for column in self.cyclic_encoded_columns:
            if column + '_sine' not in df_copy.columns:
                continue
            else:
                df_copy = self.encoders[column].decode(df_copy)
                df_copy[column] = df_copy[column].astype(self.column_dtypes[column])

        return df_copy

    def decode(self, dataframe=None, rescale=False):  # without rescaling only the cyclic part is decoded
        df_mod = dataframe.copy()
        for column in self.cyclic_encoded_columns:
            df_mod = self.encoders[column].decode(df_mod)
        if rescale:
            df_mod[self.cols_to_scale] = self.scaler.inverse_transform(df_mod[self.cols_to_scale])

        for col in df_mod.columns:
            try:
                df_mod[col] = df_mod[col].astype(self.column_dtypes[col])
            except Exception as e:
                print()
        return df_mod

    def scale(self, df):
        df_scaled = df.copy()
        df_scaled[self.cols_to_scale] = self.scaler.transform(df_scaled[self.cols_to_scale])
        return df_scaled

    def rescale(self, df):
        df_rescaled = df.copy()
        df_rescaled[self.cols_to_scale] = self.scaler.inverse_transform(df_rescaled[self.cols_to_scale])
        return df_rescaled


class PreprocessorOrdinal:
    def __init__(self, name):
        self.cols_to_scale = None
        self.encoded_columns = None
        self.encoder = None
        self.hierarchical_features = []
        self.scaler = StandardScaler()
        self.df_orig = self.fetchDataset(name, False)
        self.column_dtypes = self.df_orig.dtypes.to_dict()
        self.cats_with_nans = None
        self.df_cleaned = self.fetchDataset(name, True)
        self.train_indices = None
        self.test_indices = None
        if name == "MetroTraffic":
            self.test_indices = self.df_orig.index[self.df_orig['year'] == 2018].to_list()
            self.train_indices = self.df_orig.index[self.df_orig['year'] != 2018].to_list()
        elif name == "AustraliaTourism":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2016])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2016])].to_list()
        elif name == "BeijingAirQuality":
            temp = self.df_orig['year'].isin([2017])
            self.test_indices = temp.loc[temp].index.to_list()
            temp_c = ~temp
            self.train_indices = temp_c.loc[temp_c].index.to_list()
        elif name == "RossmanSales":
            self.test_indices = self.df_orig.index[self.df_orig['Year'].isin([2015])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['Year'].isin([2015])].to_list()
        elif name == "PanamaEnergy":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2020])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2020])].to_list()

    def fetchDataset(self, name, return_cleaned):
        if name != "BeijingAirQuality":
            if name == "RossmanSales":
                df = pd.read_csv(datasets[name], dtype={'StateHoliday': 'object'})
            else:
                df = pd.read_csv(datasets[name])
            if name == "MetroTraffic":
                df['date_time'] = pd.to_datetime(df['date_time'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time', 'weather_main', 'weather_description', 'holiday'], inplace=True)
                self.hierarchical_features = ['year', 'month', 'day', 'hour']
            elif name == "AustraliaTourism":
                df['date_time'] = pd.to_datetime(df['Quarter'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time', 'day', 'hour', 'Quarter', 'Unnamed: 0'], inplace=True)
                df = df.sort_values(by=['year', 'month', 'State', 'Region', 'Purpose']).reset_index(drop=True)
                self.hierarchical_features = ['year', 'month', 'State', 'Region', 'Purpose']
            elif name == "RossmanSales":
                store_ids = df['Store'].unique()[:10]
                df = df[(df['Store'].isin(store_ids)) & (df['Open'] == 1)]

                df['Datetime'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Datetime'].dt.year
                df['Month'] = df['Datetime'].dt.month
                df['Day'] = df['Datetime'].dt.day
                df.drop(columns=['Datetime', 'Promo', 'Open'], inplace=True)
                df = df.sort_values(by=['Year', 'Month', 'Day', 'Store'], ignore_index=True)
                df = df[['Year', 'Month', 'Day', 'Store', 'Sales', 'Customers']]
                self.hierarchical_features = ['Year', 'Month', 'Day', 'Store']
            elif name == "PanamaEnergy":
                df = df.drop(columns=['nat_demand', 'Holiday_ID', 'holiday', 'school'])

                # Create a multi-index by city and weather parameter
                # We melt the dataframe to unpivot the city-specific columns and create a 'city' column
                df = pd.melt(df,
                             id_vars=['datetime'],
                             value_vars=['T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
                                         'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san',
                                         'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav'],
                             var_name='variable',
                             value_name='value')

                # Split 'variable' column into 'city' and 'parameter'
                df['city'] = df['variable'].str.split('_').str[-1]
                df['parameter'] = df['variable'].str.split('_').str[0]

                # Pivot the dataframe to get the parameters as columns and city as a column
                df = df.pivot_table(index=['datetime', 'city'],
                                    columns='parameter',
                                    values='value').reset_index()

                # Rearranging columns (optional)
                df['date'] = pd.to_datetime(df['datetime'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['hour'] = df['date'].dt.hour
                df.drop(columns=['date', 'datetime'], inplace=True)
                df = df[['year', 'month', 'day', 'hour', 'city', 'T2M', 'TQL', 'W2M', 'QV2M']]
                df = df.sort_values(by=['year', 'month', 'day', 'hour', 'city'], ignore_index=True)
                self.hierarchical_features = ['year', 'month', 'day', 'hour', 'city']

        else:
            dfs = []
            csvs = os.listdir(datasets[name])
            csvs.sort()
            for file in csvs[:6]:
                dfs.append(pd.read_csv(datasets[name] + "/" + file))
            df = pd.concat(dfs, ignore_index=True)
            df.drop(columns=['No', 'wd'], inplace=True)  # redundant
            self.hierarchical_features = ['year', 'station', 'month', 'day', 'hour']
            df = df.sort_values(by=self.hierarchical_features).reset_index(drop=True)
        if return_cleaned:
            df_cleaned = self.cleanDataset(name, df)
            return df_cleaned

        else:
            return df

    def cleanDataset(self, name, df):
        """Beijing Air Quality has some missing values for the sensor data"""
        df_clean = df.copy()
        if name == "BeijingAirQuality":
            for column in df_clean.columns:
                if df_clean[column].dtype != 'object':
                    df_clean[column] = df_clean[column].interpolate()
            if self.encoded_columns is None:
                self.encoded_columns = ['year', 'month', 'day', 'hour', 'station']

        elif name == 'MetroTraffic':
            if self.encoded_columns is None:
                self.encoded_columns = ['year', 'month', 'day', 'hour']
        elif name == "AustraliaTourism":
            if self.encoded_columns is None:
                self.encoded_columns = ['State', 'Region', 'Purpose', 'year', 'month']
        elif name == "RossmanSales":
            if self.encoded_columns is None:
                self.encoded_columns = ['Year', 'Month', 'Day', 'Store']
        elif name == "PanamaEnergy":
            if self.encoded_columns is None:
                self.encoded_columns = ['year', 'month', 'day', 'hour', 'city']

        df_encoded = self.ordinalEncode(df_clean)  # returns the dataframe with cyclic encoding applied

        if self.cols_to_scale is None:
            self.cols_to_scale = [col for col in df_encoded.columns]

        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            df_encoded[self.cols_to_scale] = self.scaler.transform(df_encoded[self.cols_to_scale])
        else:
            df_encoded[self.cols_to_scale] = self.scaler.fit_transform(df_encoded[self.cols_to_scale])
        return df_encoded

    def ordinalEncode(self, df):
        df_copy = df.copy()
        if self.encoder is None:
            self.encoder = OrdinalEncoder().set_params(encoded_missing_value=-1)
            self.encoder.fit(df_copy[self.encoded_columns].values)
        df_copy[self.encoded_columns] = self.encoder.transform(df_copy[self.encoded_columns].values)
        if self.cats_with_nans is None:
            self.cats_with_nans = (df_copy == -1).any().to_dict()
        return df_copy

    def ordinalDecode(self, df):
        df_copy = df.copy()
        df_copy[self.encoded_columns] = self.encoder.inverse_transform(df_copy[self.encoded_columns].values)
        return df_copy

    def decode(self, dataframe=None, rescale=False, resolve=False):  # without rescaling only the cyclic part is decoded
        df_mod = dataframe.copy()
        if rescale:
            df_mod[self.cols_to_scale] = self.scaler.inverse_transform(df_mod[self.cols_to_scale])
        if resolve:
            df_mod[self.encoded_columns] = self.threshold_vals(df_mod, self.encoded_columns)
        df_mod[self.encoded_columns] = self.encoder.inverse_transform(df_mod[self.encoded_columns])
        for col in df_mod.columns:
            df_mod[col] = df_mod[col].astype(self.column_dtypes[col])
        return df_mod

    def scale(self, df):
        df_scaled = df.copy()
        df_scaled[self.cols_to_scale] = self.scaler.transform(df_scaled[self.cols_to_scale])
        return df_scaled

    def rescale(self, df):
        df_rescaled = df.copy()
        df_rescaled[self.cols_to_scale] = self.scaler.inverse_transform(df_rescaled[self.cols_to_scale])
        return df_rescaled

    def threshold_vals(self, df, encoded_columns):
        num_categories = []
        lowers = []
        for i in range(len(encoded_columns)):
            cats = len(self.encoder.categories_[i])
            if self.cats_with_nans[encoded_columns[i]]:
                cats -= 1
                lowers.append(-1)
            else:
                lowers.append(0)
            num_categories.append(cats)
        df_copy = df[encoded_columns]
        df_copy = df_copy.round()
        df_copy = df_copy.clip(lower=lowers, upper=[n - 1 for n in num_categories])
        return df_copy


def resolve_dummies(row):
    first_one = row.idxmax()  # Get the index of the first maximum (1 in this case)
    row[:] = 0.0  # Reset all values to 0
    row[first_one] = 1.0  # Set the first 1's column to 1
    return row


class PreprocessorOneHot:
    def __init__(self, name):
        self.cols_to_scale = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.hierarchical_features = []
        self.hierarchical_features_onehot = []
        self.onehot_encoded_columns = []
        self.onehot_column_names = []
        self.df_orig = self.fetchDataset(name, False)
        self.column_dtypes = self.df_orig.dtypes.to_dict()
        self.df_cleaned = self.fetchDataset(name, True)
        self.one_hot_mapper = {}
        for col in self.onehot_encoded_columns:
            feats = []
            for nm in self.onehot_column_names:
                if nm.startswith(col):
                    feats.append(nm)
            self.one_hot_mapper[col] = feats
        self.train_indices = None
        self.test_indices = None
        if name == "MetroTraffic":
            self.test_indices = self.df_orig.index[self.df_orig['year'] == 2018].to_list()
            self.train_indices = self.df_orig.index[self.df_orig['year'] != 2018].to_list()
        elif name == "AustraliaTourism":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2016])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2016])].to_list()
        elif name == "BeijingAirQuality":
            temp = self.df_orig['year'].isin([2017])
            self.test_indices = temp.loc[temp].index.to_list()
            temp_c = ~temp
            self.train_indices = temp_c.loc[temp_c].index.to_list()
        elif name == "RossmanSales":
            self.test_indices = self.df_orig.index[self.df_orig['Year'].isin([2015])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['Year'].isin([2015])].to_list()
        elif name == "PanamaEnergy":
            self.test_indices = self.df_orig.index[self.df_orig['year'].isin([2020])].to_list()
            self.train_indices = self.df_orig.index[~self.df_orig['year'].isin([2020])].to_list()

    def fetchDataset(self, name, return_cleaned):
        if name != "BeijingAirQuality":
            if name == "RossmanSales":
                df = pd.read_csv(datasets[name], dtype={'StateHoliday': 'object'})
            else:
                df = pd.read_csv(datasets[name])
            if name == "MetroTraffic":
                df['date_time'] = pd.to_datetime(df['date_time'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time', 'weather_main', 'weather_description', 'holiday'], inplace=True)
                self.hierarchical_features = ['year', 'month', 'day', 'hour']
            elif name == "AustraliaTourism":
                df['date_time'] = pd.to_datetime(df['Quarter'])
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df.drop(columns=['date_time', 'day', 'hour', 'Quarter', 'Unnamed: 0'], inplace=True)
                df = df.sort_values(by=['year', 'month', 'State', 'Region', 'Purpose']).reset_index(drop=True)
                self.hierarchical_features = ['year', 'month', 'State', 'Region', 'Purpose']
            elif name == "RossmanSales":
                store_ids = df['Store'].unique()[:10]
                df = df[(df['Store'].isin(store_ids)) & (df['Open'] == 1)]

                df['Datetime'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Datetime'].dt.year
                df['Month'] = df['Datetime'].dt.month
                df['Day'] = df['Datetime'].dt.day
                df.drop(columns=['Datetime', 'Promo', 'Open'], inplace=True)
                df = df.sort_values(by=['Year', 'Month', 'Day', 'Store'], ignore_index=True)
                df = df[['Year', 'Month', 'Day', 'Store', 'Sales', 'Customers']]
                self.hierarchical_features = ['Year', 'Month', 'Day', 'Store']
            elif name == "PanamaEnergy":
                df = df.drop(columns=['nat_demand', 'Holiday_ID', 'holiday', 'school'])

                # Create a multi-index by city and weather parameter
                # We melt the dataframe to unpivot the city-specific columns and create a 'city' column
                df = pd.melt(df,
                             id_vars=['datetime'],
                             value_vars=['T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
                                         'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san',
                                         'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav'],
                             var_name='variable',
                             value_name='value')

                # Split 'variable' column into 'city' and 'parameter'
                df['city'] = df['variable'].str.split('_').str[-1]
                df['parameter'] = df['variable'].str.split('_').str[0]

                # Pivot the dataframe to get the parameters as columns and city as a column
                df = df.pivot_table(index=['datetime', 'city'],
                                    columns='parameter',
                                    values='value').reset_index()

                # Rearranging columns (optional)
                df['date'] = pd.to_datetime(df['datetime'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['hour'] = df['date'].dt.hour
                df.drop(columns=['date', 'datetime'], inplace=True)
                df = df[['year', 'month', 'day', 'hour', 'city', 'T2M', 'TQL', 'W2M', 'QV2M']]
                df = df.sort_values(by=['year', 'month', 'day', 'hour', 'city'], ignore_index=True)
                self.hierarchical_features = ['year', 'month', 'day', 'hour', 'city']

        else:
            dfs = []
            csvs = os.listdir(datasets[name])
            csvs.sort()
            for file in csvs[:6]:
                dfs.append(pd.read_csv(datasets[name] + "/" + file))
            df = pd.concat(dfs, ignore_index=True)
            df.drop(columns=['No', 'wd'], inplace=True)  # redundant
            self.hierarchical_features = ['year', 'station', 'month', 'day', 'hour']
            df = df.sort_values(by=self.hierarchical_features).reset_index(drop=True)
        if return_cleaned:
            df_cleaned = self.cleanDataset(name, df)
            return df_cleaned

        else:
            return df

    def cleanDataset(self, name, df):
        """Beijing Air Quality has some missing values for the sensor data"""
        df_clean = df.copy()
        if name == "BeijingAirQuality":
            for column in df_clean.columns:
                if df_clean[column].dtype != 'object':
                    df_clean[column] = df_clean[column].interpolate()
            if len(self.onehot_encoded_columns) == 0:
                self.onehot_encoded_columns = ['year', 'month', 'day', 'hour', 'station']

        elif name == 'MetroTraffic':
            if len(self.onehot_encoded_columns) == 0:
                self.onehot_encoded_columns = ['year', 'month', 'day', 'hour']
        elif name == "AustraliaTourism":
            if len(self.onehot_encoded_columns) == 0:
                self.onehot_encoded_columns = ['State', 'Region', 'Purpose', 'year', 'month']
        elif name == "RossmanSales":
            if len(self.onehot_encoded_columns) == 0:
                self.onehot_encoded_columns = ['Year', 'Month', 'Day', 'Store']
        elif name == "PanamaEnergy":
            if len(self.onehot_encoded_columns) == 0:
                self.onehot_encoded_columns = ['year', 'month', 'day', 'hour', 'city']

        df_onehot = self.onehotEncode(df_clean)  # returns the dataframe with cyclic encoding applied

        for feature in self.hierarchical_features:
            if feature in self.onehot_encoded_columns:
                self.hierarchical_features_onehot.extend(self.encoders[feature])
            else:
                self.hierarchical_features_onehot.append(feature)

        if self.cols_to_scale is None:
            self.cols_to_scale = [col for col in df_clean.columns if
                                  col not in self.onehot_encoded_columns]
            df_onehot[self.cols_to_scale] = self.scaler.fit_transform(df[self.cols_to_scale])
        else:
            df_onehot[self.cols_to_scale] = self.scaler.transform(df[self.cols_to_scale])
        return df_onehot

    def onehotEncode(self, df):
        df_copy = df.copy()
        df_copy = pd.get_dummies(df_copy, columns=self.onehot_encoded_columns, dummy_na=True)
        for col in self.onehot_encoded_columns:
            if not df[col].isna().any():
                name = f'{col}_nan'
                df_copy = df_copy.drop(columns=[name])
        if len(self.onehot_column_names) == 0:  # if it's the first time
            self.onehot_column_names = [name for name in df_copy.columns if name not in df.columns]
            for column in self.onehot_encoded_columns:
                names = []
                for ohcs in self.onehot_column_names:
                    if ohcs.startswith(column):
                        names.append(ohcs)
                self.encoders[column] = names
        return df_copy

    def onehotDecode(self, df, resolve):
        df_copy = df.copy()
        for column in self.encoders.keys():
            df_select = df_copy[self.encoders[column]]
            sep_str = f'{column}_'
            if resolve:
                df_select = df_select.apply(resolve_dummies, axis=1)
            category = pd.from_dummies(df_select, sep=sep_str)
            if self.column_dtypes[column] != 'object':
                category = category.apply(pd.to_numeric)
            df_copy[column] = category.astype(self.column_dtypes[column])
            df_copy = df_copy.drop(columns=self.encoders[column])

        df_copy = df_copy[self.df_orig.columns]
        return df_copy

    def decode(self, dataframe=None, rescale=False, resolve=False):  # without rescaling only the cyclic part is decoded
        df_mod = dataframe.copy()
        df_mod = self.onehotDecode(df_mod, resolve)
        if rescale:
            df_mod[self.cols_to_scale] = self.scaler.inverse_transform(df_mod[self.cols_to_scale])

        for col in df_mod.columns:
            df_mod[col] = df_mod[col].astype(self.column_dtypes[col])
        df_mod = df_mod[self.df_orig.columns]
        return df_mod

    def scale(self, df):
        df_scaled = df.copy()
        df_scaled[self.cols_to_scale] = self.scaler.transform(df_scaled[self.cols_to_scale])
        return df_scaled

    def rescale(self, df):
        df_rescaled = df.copy()
        df_rescaled[self.cols_to_scale] = self.scaler.inverse_transform(df_rescaled[self.cols_to_scale])
        return df_rescaled
