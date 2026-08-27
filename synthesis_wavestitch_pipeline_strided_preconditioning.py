import argparse
from dataclasses import dataclass
import torch
from checkpoint_utils import (
    CheckpointCompatibilityError,
    apply_structured_checkpoint_args,
    checkpoint_preprocessing_state,
    checkpoint_state_dict,
    is_structured_checkpoint,
    validate_structured_checkpoint,
)
from data_utils import Preprocessor
from experiment_config import (
    ExperimentConfigError,
    add_common_cli_arguments,
    add_synthesis_cli_arguments,
    load_experiment_config,
    resolve_experiment,
    validate_checkpoint_experiment_conflicts,
)
from experiment_runtime import build_dry_run_report, print_dry_run_report
from training_utils import (
    MyDataset,
    fetchDiffusionConfig,
    fetchModel,
    resolve_model_columns,
)
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metadataMask
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from copy import deepcopy
import torch.nn.functional as F
from window_validation import (
    validate_synthesis_context_mask,
    validate_synthesis_windowing,
)


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


def create_pipelined_noise(test_batch, args):
    sampled = torch.normal(0, 1, (args.stride * (test_batch.shape[0] - 1) + args.window_size, test_batch.shape[2]))
    sampled_noise = sampled.unfold(0, args.window_size, args.stride).transpose(1, 2)
    return sampled_noise


@dataclass(frozen=True)
class SamplerAblationOptions:
    gradient_correction_enabled: bool = True
    sqrt_posterior_variance: bool = False


def resolve_sampler_ablation_options(
    *, disable_gradient_correction=False, sqrt_posterior_variance=False
):
    """Resolve isolated sampler ablations while preserving the upstream default."""

    if disable_gradient_correction and sqrt_posterior_variance:
        raise ValueError(
            "Sampler ablations are mutually exclusive; change only one variable per run."
        )
    return SamplerAblationOptions(
        gradient_correction_enabled=not disable_gradient_correction,
        sqrt_posterior_variance=sqrt_posterior_variance,
    )


def reverse_noise_term(posterior_variance, noise, options):
    """Return reverse-process noise with legacy or DDPM-standard amplitude."""

    amplitude = (
        torch.sqrt(posterior_variance)
        if options.sqrt_posterior_variance
        else posterior_variance
    )
    return amplitude * noise


def stitching_gradient_step(gradient, signal_indices, options):
    """Return the current fixed correction, or exactly zero for Variant A."""

    selected = gradient[:, :, signal_indices]
    if not options.gradient_correction_enabled:
        return torch.zeros_like(selected)
    return -0.1 * selected


def build_synthesis_preprocessor(dataset, proportional_cyclic_encoding, checkpoint):
    """Build preprocessing from checkpoint fit state for structured synthesis."""

    preprocessing_state = (
        checkpoint_preprocessing_state(checkpoint)
        if is_structured_checkpoint(checkpoint)
        else None
    )
    return Preprocessor(
        dataset,
        proportional_cyclic_encoding,
        preprocessing_state=preprocessing_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_common_cli_arguments(parser)
    add_synthesis_cli_arguments(parser)
    sampler_group = parser.add_mutually_exclusive_group()
    sampler_group.add_argument(
        "--disable-gradient-correction", action="store_true"
    )
    sampler_group.add_argument(
        "--sqrt-posterior-variance", action="store_true"
    )
    parsed = vars(parser.parse_args())
    disable_gradient_correction = parsed.pop(
        "disable_gradient_correction", False
    )
    sqrt_posterior_variance = parsed.pop("sqrt_posterior_variance", False)
    sampler_options = resolve_sampler_ablation_options(
        disable_gradient_correction=disable_gradient_correction,
        sqrt_posterior_variance=sqrt_posterior_variance,
    )
    config_path = parsed.pop("experiment_config", None)
    dry_run = parsed.pop("dry_run", False)
    try:
        experiment_config = (
            load_experiment_config(config_path) if config_path else None
        )
        resolved_experiment = resolve_experiment(
            experiment_config,
            parsed,
            require_synthesis_profile=True,
            allow_missing_dataset=True,
        )
    except (ExperimentConfigError, FileNotFoundError) as exc:
        parser.error(str(exc))

    checkpoint_path = resolved_experiment.values["checkpoint_path"]
    if checkpoint_path is None:
        parser.error(
            "checkpoint path is required when the dataset cannot be inferred"
        )
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if is_structured_checkpoint(loaded_checkpoint):
        checkpoint_dataset = loaded_checkpoint.get('dataset_id')
        if resolved_experiment.values["dataset"] is None:
            resolved_experiment = resolve_experiment(
                experiment_config,
                parsed,
                require_synthesis_profile=True,
                inferred_dataset=checkpoint_dataset,
            )
        validate_checkpoint_experiment_conflicts(
            loaded_checkpoint, resolved_experiment
        )
    elif resolved_experiment.values["dataset"] is None:
        parser.error('-dataset is required when loading a legacy checkpoint')

    args = resolved_experiment.namespace("synthesis")
    args.dry_run = dry_run
    if is_structured_checkpoint(loaded_checkpoint):
        apply_structured_checkpoint_args(args, loaded_checkpoint)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dry_run:
        print_dry_run_report(
            build_dry_run_report(
                args,
                resolved_experiment,
                phase="synthesis",
                checkpoint=loaded_checkpoint,
            )
        )
        raise SystemExit(0)
    preprocessor = build_synthesis_preprocessor(
        dataset, args.propCycEnc, loaded_checkpoint
    )
    df = preprocessor.df_cleaned

    windowing_plan = validate_synthesis_windowing(
        preprocessor,
        window_size=args.window_size,
        synthesis_stride=args.stride,
    )
    test_df = df.loc[list(windowing_plan.candidate_indices)]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = test_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]

    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
    rows_to_synth = metadataMask(
        metadata,
        args.synth_mask,
        args.dataset,
        dataset_config=preprocessor.dataset_config,
        test_indices=preprocessor.test_indices,
    )
    validate_synthesis_context_mask(rows_to_synth, windowing_plan)
    real_df = test_df_with_hierarchy[rows_to_synth]
    df_synth = test_df.copy()
    if args.max_windows is not None:
        full_masks = from_numpy(rows_to_synth.values).unfold(
            0, args.window_size, args.stride
        )
        synthesized_windows = torch.any(full_masks, dim=1).nonzero().flatten()
        if len(synthesized_windows) == 0:
            raise ValueError('No synthesis window contains a configured True mask value.')
        first_window = int(synthesized_windows[0])
        available_windows = len(full_masks) - first_window
        selected_windows = min(args.max_windows, available_windows)
        first_row = first_window * args.stride
        selected_rows = args.window_size + (selected_windows - 1) * args.stride
        row_slice = slice(first_row, first_row + selected_rows)
        df_synth = df_synth.iloc[row_slice]
        test_df_with_hierarchy = test_df_with_hierarchy.iloc[row_slice]
        rows_to_synth = rows_to_synth.iloc[row_slice]
        real_df = test_df_with_hierarchy[rows_to_synth]
    """Approach 1: Pipeline"""
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values

    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.stride).transpose(1, 2)
    masks = m_vals_tensor.unfold(0, args.window_size, args.stride)
    signal_column_indices, hierarchical_column_indices = resolve_model_columns(
        df_synth, preprocessor
    )
    if is_structured_checkpoint(loaded_checkpoint):
        in_dim, out_dim = validate_structured_checkpoint(
            loaded_checkpoint,
            dataset_id=dataset,
            frame=df_synth,
            preprocessor=preprocessor,
            signal_indices=signal_column_indices,
            metadata_indices=hierarchical_column_indices,
        )
    else:
        in_dim = len(df_synth.columns)
        out_dim = len(signal_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)
    non_hier_cols = signal_column_indices
    saved_params = checkpoint_state_dict(loaded_checkpoint)
    model.load_state_dict(saved_params, strict=True)
    model.eval()
    num_ops = 0  # start measuring the number of compute steps for the whole generation time
    exec_times = []
    # mses = []
    for trial in range(args.n_trials):
        start = timer()
        with torch.no_grad():
            synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
            for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
                test_batch = test_batch.to(device)
                mask_batch = mask_batch.to(device)
                x = create_pipelined_noise(test_batch, args).to(device)
                x.requires_grad_()
                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                print(f'batch: {idx} of {len(test_dataloader)}')
                mask_expanded = torch.zeros_like(test_batch, dtype=bool)
                for channel in non_hier_cols:
                    mask_expanded[:, :, channel] = mask_batch

                for step in range(diffusion_config['T'] - 1, -1, -1):
                    print(f"backward step: {step}")
                    times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
                    alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                    alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                    alpha_t = diffusion_config['alphas'][step].to(device)
                    beta_t = diffusion_config['betas'][step].to(device)
                    sampled_noise = create_pipelined_noise(test_batch, args).to(device)
                    conditional_fwd = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * sampled_noise
                    if step == diffusion_config['T'] - 1:
                        x[~mask_expanded] = conditional_fwd[~mask_expanded]
                    x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                    with torch.enable_grad():
                        epsilon_pred = model(x, times).permute((0, 2, 1))
                        # grad_xt = torch.autograd.grad(epsilon_pred, x, grad_outputs=torch.ones_like(epsilon_pred), retain_graph=True)[0]
                        if step > 0:
                            posterior_variance = beta_t * (
                                (1 - alpha_bar_t_1) / (1 - alpha_bar_t)
                            )
                            vari = reverse_noise_term(
                                posterior_variance,
                                torch.normal(0, 1, size=epsilon_pred.shape).to(device),
                                sampler_options,
                            )
                        else:
                            vari = 0.0

                        normal_denoising = create_pipelined_noise(test_batch, args).to(device)
                        normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
                                (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                        normal_denoising[:, :, non_hier_cols] += vari
                        masked_binary = mask_batch.int()

                        # x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                        # x[~mask_expanded] = test_batch[~mask_expanded]
                        rolled_x = normal_denoising.roll(1, 0)
                        rolled_x[0, args.stride:, :] = normal_denoising[0, :(args.window_size - args.stride), :]
                        # loss2 = 0.0
                        """MSE LOSS"""
                        loss1 = torch.sum((normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - rolled_x[:, args.stride:args.window_size, non_hier_cols])**2, dim=(1, 2))
                        """MAE LOSS"""
                        # loss1 = torch.sum(torch.abs(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - rolled_x[:, args.stride:args.window_size, non_hier_cols]), dim=(1, 2))
                        """COSINE SIMILARITY"""
                        # dot = torch.sum(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] * rolled_x[:, args.stride:args.window_size, non_hier_cols], dim=1)
                        # unorm = torch.norm(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols], p=2, dim=1)
                        # vnorm = torch.norm(rolled_x[:, args.stride:args.window_size, non_hier_cols], p=2, dim=1)
                        # cosinesim = dot/(unorm*vnorm + 1e-8)
                        # loss1 = 1 - cosinesim.mean(dim=1)

                        """TEMPORAL CORRELATION"""
                        # umean = torch.mean(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols], dim=1, keepdim=True)
                        # vmean = torch.mean(rolled_x[:, args.stride:args.window_size, non_hier_cols], dim=1, keepdim=True)
                        # ucentred = normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - umean
                        # vcentred = rolled_x[:, args.stride:args.window_size, non_hier_cols] - vmean
                        # num = torch.sum(ucentred*vcentred, dim=1)
                        # den = torch.sqrt(torch.sum(ucentred**2, dim=1) * torch.sum(vcentred**2, dim=1) + 1e-8)
                        # score = num/den
                        # loss1 = 1 - score.mean(dim=1)

                        loss2 = torch.sum(~mask_expanded[:, :, non_hier_cols] * ((x[:, :, non_hier_cols]-torch.sqrt(1-alpha_bar_t)*epsilon_pred)/(torch.sqrt(alpha_bar_t)) - test_batch[:, :, non_hier_cols])**2, dim=(1, 2))
                        # loss1 = 0.0
                        loss = loss1 + loss2
                        # print(torch.sum(loss.cpu()))
                        grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

                    x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                    eps = stitching_gradient_step(
                        grad, non_hier_cols, sampler_options
                    )
                    x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + eps
                    # x[:, :, non_hier_cols] += eps/sqrt(alpha_t) - (eps*beta_t/(sqrt(alpha_t)*sqrt(1-alpha_bar_t))) * grad_xt[:, :, non_hier_cols]
                    # x[1:, : (args.window_size - args.stride), :] = rolled_x[1:, args.stride: args.window_size, :]
                    # x[~mask_expanded] = test_batch[~mask_expanded]
                    # if step == 0:
                    #     pass
                        # x[~mask_expanded] = test_batch[~mask_expanded]
                        # x[1:, : (args.window_size - args.stride), :] = rolled_x[1:, args.stride: args.window_size, :]
                    if trial == 0:
                        num_ops += 1

                x[~mask_expanded] = test_batch[~mask_expanded]
                first_sample = x[0]
                # mse = torch.mean((x[mask_expanded] - test_batch[mask_expanded])**2).cpu()
                # print(f'MSE: {mse}')
                # plt.plot(x[0, :, non_hier_cols].cpu())
                # plt.plot(test_batch[0, :, non_hier_cols].cpu())
                # plt.title(f'{mse}')
                # plt.show()
                # exit()
                last_timesteps = x[1:, (args.window_size - args.stride):, :]
                if idx == 0:
                    last_timesteps = last_timesteps.reshape(-1, last_timesteps.shape[2])
                    generated = torch.cat((first_sample, last_timesteps), dim=0)
                else:
                    generated = x[:, (args.window_size - args.stride):, :]
                    generated = generated.reshape(-1, generated.shape[2])
                synth_tensor = torch.cat((synth_tensor, generated), dim=0)

        end = timer()
        diff = end - start
        exec_times.append(diff)
        df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
        real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
        real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
        synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)
        rows_to_synth_reset = rows_to_synth.reset_index(drop=True)
        synth_df_reconverted_selected = synth_df_reconverted[rows_to_synth_reset]
        synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
        synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
    #     real_df_cleaned = preprocessor.cleanDataset(args.dataset, real_df_reconverted)
    #     synth_df_cleaned = preprocessor.cleanDataset(args.dataset, synth_df_reconverted_selected)
    #     nhc = [c for c in real_df_cleaned.columns if
    #                      c not in preprocessor.hierarchical_features_cyclic]
    #     MSE = ((synth_df_cleaned[nhc] - real_df_cleaned[nhc]) ** 2).mean().mean()
    #     mses.append(MSE)
    #     print(trial)
    # print(np.mean(np.array(mses)))
        path = os.path.join(args.output_dir, args.dataset, args.synth_mask)
        if not os.path.exists(path):
            os.makedirs(path)

        real_path = os.path.join(path, 'real.csv')
        if not os.path.exists(real_path):
            real_df_reconverted.to_csv(real_path)
        synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
        if args.propCycEnc:
            synth_df_reconverted_selected.to_csv(
                os.path.join(path, f'synth_wavestitch_pipeline_stride_{args.stride}_trial_{trial}_cycProp_grad_correction.csv'))
            if trial == 0:
                with open(os.path.join(path, f'denoiser_calls_pipeline_stride_{args.stride}_cycProp_grad_correction.txt'), 'w') as file:
                    file.write(str(num_ops))
        else:
            synth_df_reconverted_selected.to_csv(
                os.path.join(path, f'synth_wavestitch_pipeline_stride_{args.stride}_trial_{trial}_cycStd_grad_correction.csv'))
            if trial == 0:
                with open(os.path.join(path, f'denoiser_calls_pipeline_stride_{args.stride}_cycStd_grad_correction.txt'), 'w') as file:
                    file.write(str(num_ops))

    denoiser_log = os.path.join(
        path,
        f'denoiser_calls_pipeline_stride_{args.stride}_cycStd_grad_correction.txt',
    )
    with open(denoiser_log, 'a') as file:
        arr_time = np.array(exec_times)
        file.write('\n' + str(np.mean(arr_time)) + '\n')
        file.write(str(np.std(arr_time)))
