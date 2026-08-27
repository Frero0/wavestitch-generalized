import pytest
import torch

from synthesis_wavestitch_pipeline_strided_preconditioning import (
    resolve_sampler_ablation_options,
    reverse_noise_term,
    stitching_gradient_step,
)


def test_default_preserves_legacy_gradient_and_variance_amplitude():
    options = resolve_sampler_ablation_options()
    variance = torch.tensor([0.04])
    noise = torch.tensor([2.0])
    gradient = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    signal_indices = torch.tensor([0, 2])

    assert options.gradient_correction_enabled is True
    assert options.sqrt_posterior_variance is False
    torch.testing.assert_close(
        reverse_noise_term(variance, noise, options), variance * noise
    )
    torch.testing.assert_close(
        stitching_gradient_step(gradient, signal_indices, options),
        -0.1 * gradient[:, :, signal_indices],
    )


def test_variant_a_changes_only_gradient_correction():
    default = resolve_sampler_ablation_options()
    variant = resolve_sampler_ablation_options(disable_gradient_correction=True)
    variance = torch.tensor([0.04])
    noise = torch.tensor([2.0])
    gradient = torch.ones((1, 2, 3))
    signal_indices = torch.tensor([0, 1])

    assert variant.gradient_correction_enabled is False
    assert variant.sqrt_posterior_variance == default.sqrt_posterior_variance
    torch.testing.assert_close(
        reverse_noise_term(variance, noise, variant),
        reverse_noise_term(variance, noise, default),
    )
    torch.testing.assert_close(
        stitching_gradient_step(gradient, signal_indices, variant),
        torch.zeros((1, 2, 2)),
    )


def test_variant_b_changes_only_reverse_noise_amplitude():
    default = resolve_sampler_ablation_options()
    variant = resolve_sampler_ablation_options(sqrt_posterior_variance=True)
    variance = torch.tensor([0.04])
    noise = torch.tensor([2.0])
    gradient = torch.ones((1, 2, 3))
    signal_indices = torch.tensor([0, 1])

    assert variant.gradient_correction_enabled == default.gradient_correction_enabled
    assert variant.sqrt_posterior_variance is True
    torch.testing.assert_close(
        reverse_noise_term(variance, noise, variant),
        torch.sqrt(variance) * noise,
    )
    torch.testing.assert_close(
        stitching_gradient_step(gradient, signal_indices, variant),
        stitching_gradient_step(gradient, signal_indices, default),
    )


def test_variants_cannot_be_combined():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_sampler_ablation_options(
            disable_gradient_correction=True,
            sqrt_posterior_variance=True,
        )
