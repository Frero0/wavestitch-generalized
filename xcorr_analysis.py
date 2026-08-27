import pandas as pd
from matplotlib import pyplot as plt
from data_utils import Preprocessor
import numpy as np
import os


def env_list(name, default):
    value = os.environ.get(name)
    return default if not value else [item.strip() for item in value.split(',') if item.strip()]


if __name__ == "__main__":
    datasets = env_list(
        "WAVESTITCH_EVAL_DATASETS",
        ["AustraliaTourism", "MetroTraffic", "RossmanSales", "BeijingAirQuality", "PanamaEnergy"])
    methods = env_list(
        "WAVESTITCH_EVAL_XCORR_METHODS",
        ["timegan", "timeweaver", "tsdiff-0.0", "tsdiff-0.5", 'tsdiff-1.0', 'tsdiff-2.0', 'algo-1',
         'algo-8', 'algo-16', 'algo-32', 'sssd', 'timeautodiff'])
    levels = env_list("WAVESTITCH_EVAL_LEVELS", ['C', 'M', 'F'])
    n_trials = int(os.environ.get("WAVESTITCH_EVAL_N_TRIALS", "5"))
    wavestitch_suffix = os.environ.get(
        "WAVESTITCH_EVAL_WAVESTITCH_SUFFIX", "cycStd_grad_simplecoeff")
    xcorrdtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. xcorrD', 'Std. xcorrD'])
    for dataset in datasets:
        preprocessor = Preprocessor(dataset, False)
        for method in methods:
            for mask in levels:
                real = pd.read_csv(f'generated/{dataset}/{mask}/real.csv')
                non_hier_cols = [col for col in real.columns if
                                 col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']

                filt = real[non_hier_cols]
                xcorr_real = filt.corr()
                MAES = []
                for trial in range(n_trials):
                    if method == "timegan":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timegan_trial_{trial}_cycStd.csv')
                    elif method == "timeweaver":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif "tsdiff" in method:
                        strength = method.split('-')[1]
                        data = pd.read_csv(
                            f'generated/{dataset}/{mask}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')
                    elif "sssd" in method:
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_sssd_signalconditioned_trial_{trial}.csv')
                    elif 'timeautodiff' in method:
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeautodiff_trial_{trial}.csv')
                    else:
                        stride = method.split('-')[1]
                        data = pd.read_csv(
                            f'generated/{dataset}/{mask}/synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_{wavestitch_suffix}.csv')
                    data = data[non_hier_cols]
                    xcorr_data = data.corr()
                    diff = (xcorr_real - xcorr_data).abs().mean().mean()
                    MAES.append(diff)
                arr = np.array(MAES)
                avg = np.mean(arr)
                std = np.std(arr)
                if "algo" in method:
                    tech = 'wavestitch-' + method.split('-')[1]
                else:
                    tech = method
                row = {"Dataset": dataset, "Method": tech, "Level": mask, 'Avg. xcorrD': avg, 'Std. xcorrD': std}
                xcorrdtable.loc[len(xcorrdtable)] = row
    final_path = os.environ.get(
        "WAVESTITCH_EVAL_XCORR_OUTPUT",
        "experiments/xcorrdtable/xcorrdtable_wavestitch_grad_revision.csv")
    output_dir = os.path.dirname(final_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    xcorrdtable.to_csv(final_path)
