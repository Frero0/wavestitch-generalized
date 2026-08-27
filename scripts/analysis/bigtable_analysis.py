from wavestitch.data_utils import Preprocessor
import pandas as pd
import numpy as np
import os


def env_list(name, default):
    value = os.environ.get(name)
    return default if not value else [item.strip() for item in value.split(',') if item.strip()]


if __name__ == "__main__":
    datasets = env_list(
        "WAVESTITCH_EVAL_DATASETS",
        ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"])
    levels = env_list("WAVESTITCH_EVAL_LEVELS", ['C', 'M', 'F'])
    methods = env_list(
        "WAVESTITCH_EVAL_MSE_METHODS",
        ["TimeGAN", "SSSD", "TimeAutoDiff", "TimeWeaver", "TSDiff-0", "TSDiff-0.5", "TSDiff-1.0",
         "TSDiff-2.0", "Pipe-1", "Pipe-8", "Pipe-16", "Pipe-32"])
    n_trials = int(os.environ.get("WAVESTITCH_EVAL_N_TRIALS", "5"))
    wavestitch_suffix = os.environ.get(
        "WAVESTITCH_EVAL_WAVESTITCH_SUFFIX", "cycStd_grad_simplecoeff")
    bigtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. MSE', 'Std. MSE'])
    for dataset in datasets:
        preprocessor = Preprocessor(dataset, False)
        for level in levels:
            for method in methods:
                df_real = pd.read_csv(f"generated/{dataset}/{level}/real.csv").drop(columns=['Unnamed: 0'])
                df_real_cleaned = preprocessor.cleanDataset(dataset, df_real)
                non_hier_cols = [col for col in df_real_cleaned.columns if
                                 col not in preprocessor.hierarchical_features_cyclic]
                df_real_cleaned_selected = df_real_cleaned[non_hier_cols]
                mses = []
                for trial in range(n_trials):
                    df_synth = None
                    if "TSDiff" in method:
                        strength = float(method.split('-')[1])
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')

                    elif "Pipe" in method:
                        stride = int(method.split('-')[1])
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_{wavestitch_suffix}.csv')
                    elif method == "TimeWeaver":
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif method == "TimeGAN":
                        df_synth = pd.read_csv(f'generated/{dataset}/{level}/synth_timegan_trial_{trial}_cycStd.csv')
                    elif method == "SSSD":
                        df_synth = pd.read_csv(f'generated/{dataset}/{level}/synth_sssd_signalconditioned_trial_{trial}.csv')
                    elif method == 'TimeAutoDiff':
                        df_synth = pd.read_csv(f'generated/{dataset}/{level}/synth_timeautodiff_trial_{trial}.csv')

                    df_synth = df_synth.drop(columns=['Unnamed: 0'])

                    df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                    df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                    MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                    mses.append(MSE)

                mses = np.array(mses)
                AVG_MSE = np.mean(mses)
                STD = np.std(mses)
                row = {'Dataset': dataset, 'Method': method, 'Level': level, 'Avg. MSE': AVG_MSE,
                       'Std. MSE': STD}

                bigtable.loc[len(bigtable)] = row

            final_path = os.environ.get(
                "WAVESTITCH_EVAL_MSE_OUTPUT",
                "experiments/bigtable/bigtable_wavestitch_grad_revision.csv")
            output_dir = os.path.dirname(final_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            bigtable.to_csv(final_path)
