import argparse

import torch

from wavestitch.checkpoint_utils import build_structured_checkpoint
from wavestitch.data_utils import Preprocessor
from wavestitch.experiment_config import (
    ExperimentConfigError,
    add_common_cli_arguments,
    add_training_cli_arguments,
    load_experiment_config,
    resolve_experiment,
)
from wavestitch.experiment_runtime import build_dry_run_report, print_dry_run_report
from wavestitch.training_utils import (
    MyDataset,
    fetchDiffusionConfig,
    fetchModel,
    resolve_model_columns,
)
from wavestitch.window_validation import validate_training_windowing
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_common_cli_arguments(parser)
    add_training_cli_arguments(parser)
    parsed = vars(parser.parse_args())
    config_path = parsed.pop("experiment_config", None)
    dry_run = parsed.pop("dry_run", False)
    try:
        experiment_config = (
            load_experiment_config(config_path) if config_path else None
        )
        resolved_experiment = resolve_experiment(experiment_config, parsed)
    except (ExperimentConfigError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args = resolved_experiment.namespace("training")
    args.dry_run = dry_run
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dry_run:
        print_dry_run_report(
            build_dry_run_report(
                args,
                resolved_experiment,
                phase="training",
            )
        )
        raise SystemExit(0)
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned
    windowing_plan = validate_training_windowing(
        preprocessor,
        window_size=args.window_size,
        requested_training_stride=args.stride,
        effective_training_stride=1,
    )
    print(
        'training windowing: effective_stride={} (upstream), requested_stride={}, '
        'windows={}'.format(
            windowing_plan.training_stride,
            args.stride,
            windowing_plan.training_window_count,
        )
    )
    training_df = df.loc[list(windowing_plan.train_indices)]
    signal_column_indices, hierarchical_column_indices = resolve_model_columns(
        training_df, preprocessor
    )
    # training_samples = []
    d_vals_tensor = from_numpy(training_df.values)
    training_samples = d_vals_tensor.unfold(
        0, args.window_size, windowing_plan.training_stride
    ).transpose(1, 2)
    # masks = m_vals_tensor.unfold(0, args.window_size, 1)
    # for i in range(0, len(training_df) - args.window_size + 1, args.stride):
    #     window = training_df.iloc[i:i + args.window_size].values
    #     training_samples.append(window)
    in_dim = len(training_df.columns)
    out_dim = len(signal_column_indices)
    training_dataset = MyDataset(training_samples.float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    non_hier_cols = signal_column_indices
    """TRAINING"""
    optimizer_steps = 0
    stop_training = False
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)
            sigmas = normal(0, 1, size=batch.shape).to(device)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars'].to(device)
            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
            batch_noised = batch_noised.to(device)
            timesteps = timesteps.reshape((-1, 1))
            # timesteps = timesteps.to(device)
            sigmas_predicted = model(batch_noised, timesteps)
            optimizer.zero_grad()
            sigmas_permuted = sigmas[:, :, non_hier_cols].permute((0, 2, 1))
            sigmas_permuted = sigmas_permuted.to(device)
            loss = criterion(sigmas_predicted, sigmas_permuted)
            loss.backward()
            total_loss += loss
            optimizer.step()
            optimizer_steps += 1
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                stop_training = True
                break
        print(f'epoch: {epoch}, loss: {total_loss}')
        if stop_training:
            break
    if args.checkpoint_path is not None:
        filepath = args.checkpoint_path
    else:
        path = f'saved_models/{args.dataset}/'
        if args.propCycEnc:
            filename = "model_prop.pth"
        else:
            filename = "model.pth"
        filepath = os.path.join(path, filename)

    checkpoint_directory = os.path.dirname(filepath)
    if checkpoint_directory:
        os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint = build_structured_checkpoint(
        model=model,
        dataset_id=dataset,
        frame=training_df,
        preprocessor=preprocessor,
        signal_indices=signal_column_indices,
        metadata_indices=hierarchical_column_indices,
        args=args,
        effective_training_stride=windowing_plan.training_stride,
        optimizer_steps=optimizer_steps,
    )
    torch.save(checkpoint, filepath)
    print(
        'saved checkpoint: {} (optimizer_steps={}, in_dim={}, out_dim={}, '
        'signal_indices={}, metadata_indices={})'.format(
            filepath,
            optimizer_steps,
            in_dim,
            out_dim,
            signal_column_indices.tolist(),
            hierarchical_column_indices.tolist(),
        )
    )
