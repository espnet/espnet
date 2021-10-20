# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for WaveNet modules."""

import pytest
import torch

from espnet2.gan_tts.wavenet import WaveNet


def make_wavenet_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=4,
        stacks=1,
        base_dilation=2,
        residual_channels=4,
        aux_channels=-1,
        gate_channels=8,
        skip_channels=8,
        global_channels=-1,
        dropout_rate=0.0,
        bias=True,
        use_weight_norm=True,
        use_first_conv=True,
        use_last_conv=False,
        scale_residual=False,
        scale_skip_connect=False,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "model_dict",
    [
        ({}),
        ({"use_first_conv": False}),
        ({"use_last_conv": True}),
        ({"global_channels": 3}),
        ({"aux_channels": 3}),
        ({"scale_residual": True}),
        ({"scale_skip_connect": True}),
    ],
)
def test_wavenet_forward(model_dict):
    batch_size = 2
    batch_length = 128
    args = make_wavenet_args(**model_dict)
    if args["use_first_conv"]:
        y = torch.randn(batch_size, 1, batch_length)
    else:
        y = torch.randn(batch_size, args["residual_channels"], batch_length)
    c, g = None, None
    if args["aux_channels"] > 0:
        c = torch.randn(batch_size, args["aux_channels"], batch_length)
    if args["global_channels"] > 0:
        g = torch.randn(batch_size, args["global_channels"], 1)
    model = WaveNet(**args)
    out = model(y, c=c, g=g)
    if args["use_last_conv"]:
        out.size(1) == args["out_channels"]
    else:
        out.size(1) == args["skip_channels"]
