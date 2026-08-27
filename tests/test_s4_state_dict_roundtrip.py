import io

import torch

from wavestitch.models.sssd_s4_imputer import SSSDS4Imputer


def _make_model():
    return SSSDS4Imputer(
        in_channels=3,
        res_channels=4,
        skip_channels=4,
        out_channels=2,
        num_res_layers=1,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
        s4_lmax=8,
        s4_d_state=8,
        s4_dropout=0.0,
        s4_bidirectional=True,
        s4_layernorm=True,
    ).eval()


def test_s4_state_dict_save_load_roundtrip():
    torch.manual_seed(1234)
    source = _make_model()
    inputs = torch.randn(2, 8, 3)
    timesteps = torch.tensor([[3], [7]])

    with torch.no_grad():
        expected = source(inputs, timesteps)

    checkpoint = io.BytesIO()
    torch.save(source.state_dict(), checkpoint)
    checkpoint.seek(0)

    torch.manual_seed(5678)
    restored = _make_model()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)

    with torch.no_grad():
        actual = restored(inputs, timesteps)

    torch.testing.assert_close(actual, expected)

    for name, buffer in restored.named_buffers():
        if name.endswith((".B", ".P", ".w")):
            assert buffer.is_contiguous(), name
            assert torch._debug_has_internal_overlap(buffer) == 0, name
