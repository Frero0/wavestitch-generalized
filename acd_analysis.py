import pandas as pd
import numpy as np
from data_utils import Preprocessor
import os
import warnings
lags = 100


def env_list(name, default):
    value = os.environ.get(name)
    return default if not value else [item.strip() for item in value.split(',') if item.strip()]


if __name__ == "__main__":
    datasets = env_list(
        "WAVESTITCH_EVAL_DATASETS",
        ["MetroTraffic", "RossmanSales", "BeijingAirQuality", "AustraliaTourism", "PanamaEnergy"])
    methods = env_list(
        "WAVESTITCH_EVAL_ACD_METHODS",
        ['algo-8', 'algo-16', 'algo-32', 'algo-1', "timegan", "timeweaver", "tsdiff-0.5", "sssd",
         'timeautodiff'])
    levels = env_list("WAVESTITCH_EVAL_LEVELS", ['C', 'M', 'F'])
    n_trials = int(os.environ.get("WAVESTITCH_EVAL_N_TRIALS", "5"))
    wavestitch_suffix = os.environ.get(
        "WAVESTITCH_EVAL_WAVESTITCH_SUFFIX", "cycStd_grad_simplecoeff")
    acdtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. ACD', 'Std. ACD'])
    for dataset in datasets:
        preprocessor = Preprocessor(dataset, False)
        for method in methods:
            for mask in levels:
                real = pd.read_csv(f'generated/{dataset}/{mask}/real.csv')
                non_hier_cols = [col for col in real.columns if
                                 col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']

                filt = real[non_hier_cols].values
                stds = np.std(filt, axis=0)
                boolmask = stds == 0
                stds[boolmask] = 1.0
                filt_centered = (filt - np.mean(filt, axis=0))/stds
                autocorr_real = np.ones((len(non_hier_cols), lags))
                for lag in range(1, lags):
                    acf = np.mean(filt_centered[lag:, :] * filt_centered[:-lag, :], axis=0)
                    autocorr_real[:, lag] = acf

                MAES = []
                for trial in range(n_trials):
                    if method == "timegan":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timegan_trial_{trial}_cycStd.csv')
                    elif method == "timeweaver":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif "tsdiff" in method:
                        strength = method.split('-')[1]
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')
                    elif "sssd" in method:
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_sssd_signalconditioned_trial_{trial}.csv')
                    elif 'timeautodiff' in method:
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeautodiff_trial_{trial}.csv')
                    else:
                        stride = method.split('-')[1]
                        data = pd.read_csv(
                            f'generated/{dataset}/{mask}/synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_{wavestitch_suffix}.csv')
                    data = data[non_hier_cols].values
                    stds_meth = np.std(data, axis=0)
                    boolmask_meth = stds_meth == 0
                    stds_meth[boolmask_meth] = 1.0
                    data_centered = (data - np.mean(data, axis=0))/stds_meth
                    autocorr = np.ones((len(non_hier_cols), lags))
                    for lag in range(1, lags):
                        acf = np.mean(data_centered[lag:, :] * data_centered[:-lag, :], axis=0)
                        autocorr[:, lag] = acf

                    undefmask = boolmask_meth | boolmask
                    diff = np.abs(autocorr_real - autocorr)
                    complement = undefmask==False
                    diff_filtered = diff[complement, :]
                    MAE = np.mean(diff_filtered)
                    MAES.append(MAE)
                arr = np.array(MAES)
                avg = np.mean(arr)
                std = np.std(arr)
                if "algo" in method:
                    tech = 'wavestitch-'+method.split('-')[1]
                else:
                    tech = method
                row = {"Dataset": dataset, "Method": tech, "Level": mask, 'Avg. ACD': avg, 'Std. ACD': std}
                acdtable.loc[len(acdtable)] = row
    final_path = os.environ.get(
        "WAVESTITCH_EVAL_ACD_OUTPUT",
        "experiments/acdtable/acdtable_wavestitch_grad_revision.csv")
    output_dir = os.path.dirname(final_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    acdtable.to_csv(final_path)
