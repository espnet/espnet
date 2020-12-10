import torch
import os

import espnet2.utils.plot_sinc_filters as psf
from espnet2.layers.sinc_conv import SincConv


def test_plot_sinc_filters_help():
    args = ["-h"]
    try:
        psf.main(args)
    except SystemExit:
        pass


def test_plot_sinc_filters_plot_filters():
    # random path to avoid errors from existing files
    random_number = int(torch.rand(1) * 100000)
    output_path = f"./test_plot_sinc_filters_{random_number}"
    os.mkdir(output_path)
    model_path = output_path + "/test.model.pth"
    # We need a mock model. - One could also initialize a full E2E model.
    filters = SincConv(
        in_channels=1, out_channels=128, kernel_size=101, stride=1, fs=16000
    )
    model = {"preencoder.filters.f": filters.f}
    model = {"model": model}
    torch.save(model, model_path)
    # WARNING. plotting with --all gives full test coverage,
    # but also plotting all filters takes ~ 2 minutes
    # for speedup, remove the --all in the list below
    args = ["--all", "--sample_rate", "16000", model_path, output_path]
    psf.main(args)
