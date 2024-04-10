import pytest
import torch

from espnet2.enh.layers.ncsnpp import NCSNpp


@pytest.mark.parametrize("scale_by_sigma", [True, False])
@pytest.mark.parametrize("nonlinearity", ["elu", "relu", "lrelu", "swish"])
@pytest.mark.parametrize("fir", [True])
@pytest.mark.parametrize("skip_rescale", [True, False])
@pytest.mark.parametrize("progressive_input", ["none", "input_skip", "residual"])
@pytest.mark.parametrize(
    "resblock_type, resamp_with_conv", [("biggan", True), ("ddpm", False)]
)
def test_ncsnpp_forward_backward_complex(
    scale_by_sigma,
    nonlinearity,
    resamp_with_conv,
    fir,
    skip_rescale,
    resblock_type,
    progressive_input,
):
    model = NCSNpp(
        scale_by_sigma=scale_by_sigma,
        nonlinearity=nonlinearity,
        nf=16,
        ch_mult=(
            1,
            2,
        ),
        num_res_blocks=2,
        attn_resolutions=(1,),
        resamp_with_conv=resamp_with_conv,
        conditional=True,
        fir=fir,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=skip_rescale,
        resblock_type=resblock_type,
        progressive_input=progressive_input,
        init_scale=0.0,
        fourier_scale=16,
        image_size=32,
        dropout=0.0,
        centered=True,
    )

    real = torch.rand(2, 2, 32, 32)
    imag = torch.rand(2, 2, 32, 32)
    x = real + 1j * imag
    t = torch.rand(2)
    rtv = model(x, t)
    rtv.abs().mean().backward()
