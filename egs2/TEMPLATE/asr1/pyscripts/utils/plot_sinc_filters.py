#!/usr/bin/env python3
#  2020, Technische Universität München; Nicolas Lindae, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Visualize Sinc convolution filters.

Description:
    This program loads a pretrained Sinc convolution of an ESPnet2 ASR model and
    plots filters, as well as the bandpass frequencies. The learned filter values
    are automatically read out from a trained model file (`*.pth`). Plots are
    saved to the specified output directory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate.")
    parser.add_argument(
        "--all", action="store_true", help="Plot every filter in its own plot."
    )
    parser.add_argument(
        "--filetype", type=str, default="png", help="Filetype (svg, png)."
    )
    parser.add_argument(
        "--filter-key",
        type=str,
        default="preencoder.filters.f",
        help="Name of the torch module the Sinc filter parameters are stored"
        " within the model file.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="mel",
        choices=["mel", "bark"],
        help="Filter bank initialization values.",
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model file (*.pth)."
    )
    parser.add_argument(
        "out_folder",
        type=Path,
        nargs="?",
        default=Path("plot_sinc_filters").absolute(),
        help="Output folder to save the plots in.",
    )
    return parser


def convert_parameter_to_frequencies(f1, f2, sample_rate, sorted):
    """Convert parameters to frequencies.

    Parameters f1 and f2 denote frequencies normed to the sampling frequency.

    Args:
        f1: Lower frequency relative to sample rate.
        f2: Higher frequency  relative to sample rate.
        sample_rate: Sample rate.
        sorted: Sort filters by their center frequency.

    Returns:
        f_mins: Absolute lower frequency.
        f_maxs: Absolute higher frequency.
        f_mins: Absolute center frequency.
    """
    f_mins = np.abs(f1) * sample_rate
    f_maxs = (np.abs(f1) + np.abs(f2 - f1)) * sample_rate
    f_mins = np.clip(f_mins, 0, sample_rate / 2)
    f_maxs = np.clip(f_maxs, 0, sample_rate / 2)
    f_mids = (f_maxs + f_mins) / 2
    if sorted:
        order = np.argsort(f_mids)
        f_mins, f_mids, f_maxs = f_mins[order], f_mids[order], f_maxs[order]
    return f_mins, f_maxs, f_mids


def plot_filtergraph(
    filters: torch.Tensor,
    sample_rate: int,
    img_path: str,
    sorted: bool = True,
    logscale: bool = False,
    scale: str = "mel",
):
    """Plot the Sinc filter bandpass frequencies.

    Args:
        filters: Filter parameters.
        sample_rate: Sample rate of signal.
        img_path: Output plot file.
        sorted: Sort bandpasses by center frequency.
        logscale: Set Y axis to logarithmic scale.
    """
    if scale == "mel":
        from espnet2.layers.sinc_conv import MelScale

        f = MelScale.bank(128, sample_rate).detach().cpu().numpy()
    elif scale == "bark":
        from espnet2.layers.sinc_conv import BarkScale

        f = BarkScale.bank(128, sample_rate).detach().cpu().numpy()
    else:
        raise NotImplementedError

    f = f / sample_rate
    f1, f2 = f[:, 0], f[:, 1]
    f_mins, f_maxs, f_mids = convert_parameter_to_frequencies(
        filters[:, 0], filters[:, 1], sample_rate, sorted
    )
    mel_mins, mel_maxs, mel_mids = convert_parameter_to_frequencies(
        f1, f2, sample_rate, sorted
    )

    x = np.arange(len(f_mids))
    plt.clf()
    if logscale:
        plt.yscale("log")
    plt.xlabel("filter index")
    plt.ylabel("f [Hz]")
    ax = plt.gca()
    ax.plot(x, mel_mins, color="blue", label="mel filters")
    ax.plot(x, mel_maxs, color="blue")
    ax.plot(x, mel_mids, "--", color="darkblue")
    ax.fill_between(x, mel_mins, mel_maxs, color="blue", alpha=0.3)
    ax.plot(x, f_mins, color="green", label="learned filters")
    ax.plot(x, f_maxs, color="green")
    ax.plot(x, f_mids, "--", color="darkgreen")
    ax.fill_between(x, f_mins, f_maxs, color="green", alpha=0.3)
    ax.legend(loc="upper left", prop={"size": 15})
    plt.savefig(img_path, bbox_inches="tight")
    print("Plotted %s" % img_path)


def plot_filter_kernels(filters: torch.Tensor, sample_rate: int, args):
    """Plot the Sinc filter kernels.

    Args:
        filters (torch.Tensor): Filter parameters.
        sample_rate (int): Sample rate of Signal.
        args (dict): Dictionary with output options.
    """
    from espnet2.layers.sinc_conv import SincConv

    print(
        "When plotting filter kernels, make sure the script has the"
        " correct SincConv settings (currently hard-coded)."
    )
    convs = SincConv(1, 128, 101)

    # unlearned
    convs._create_filters(convs.f.device)
    pre_kernels = convs.sinc_filters.detach().numpy()

    pre_filters = convs.f.detach().numpy()
    f_mins = np.abs(pre_filters[:, 0])
    f_maxs = np.abs(pre_filters[:, 0]) + np.abs(pre_filters[:, 1] - pre_filters[:, 0])
    F_mins, F_maxs = f_mins * sample_rate, f_maxs * sample_rate
    pre_F_mins, pre_F_maxs = np.round(F_mins).astype(np.int), np.round(F_maxs).astype(
        np.int
    )

    # learned
    convs.f = torch.nn.Parameter(torch.Tensor(filters))
    convs._create_filters(convs.f.device)
    kernels = convs.sinc_filters.detach().numpy()

    f_mins = np.abs(filters[:, 0])
    f_maxs = np.abs(filters[:, 0]) + np.abs(filters[:, 1] - filters[:, 0])
    F_mins, F_maxs = f_mins * sample_rate, f_maxs * sample_rate
    F_mins, F_maxs = np.round(F_mins).astype(np.int), np.round(F_maxs).astype(np.int)
    F_mins, F_maxs = np.clip(F_mins, 0, sample_rate / 2.0), np.clip(
        F_maxs, 0, sample_rate / 2.0
    )

    x_f = np.linspace(0.0, np.max(F_maxs), int(np.max(F_maxs)) + 1)
    x = np.arange(kernels.shape[2])
    if args.all:
        for i in range(len(kernels)):
            pre_kernel = pre_kernels[i][0]
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.plot(x, pre_kernel)
            img_name = "filter_pre_kernel_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = str(args.out_folder / img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            kernel = kernels[i][0]
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.plot(x, kernel)
            img_name = "filter_kernel_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = str(args.out_folder / img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            plt.clf()
            plt.xlabel("kernel index")
            plt.plot(x, kernel)
            plt.plot(x, pre_kernel, "--", alpha=0.5)
            img_name = "filter_kernel_both_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = str(args.out_folder / img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            y = np.zeros_like(x_f)
            y[F_mins[i] : F_maxs[i]] = 1.0
            plt.clf()
            plt.plot(x_f, y)
            img_name = "filter_freq_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = str(args.out_folder / img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            pre_y = np.zeros_like(x_f)
            pre_y[pre_F_mins[i] : pre_F_maxs[i]] = 1.0
            plt.clf()
            plt.plot(x_f, y)
            plt.plot(x_f, pre_y)
            img_name = "filter_freq_both_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = args.out_folder / img_name
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

    plt.clf()
    filters = [32, 71, 113, 126]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey="row")

    axs[0, 0].plot(x, kernels[filters[0]][0])
    axs[0, 0].plot(x, pre_kernels[filters[0]][0], "--", alpha=0.5)
    axs[0, 1].plot(x, kernels[filters[1]][0])
    axs[0, 1].plot(x, pre_kernels[filters[1]][0], "--", alpha=0.5)
    axs[1, 0].plot(x, kernels[filters[2]][0])
    axs[1, 0].plot(x, pre_kernels[filters[2]][0], "--", alpha=0.5)
    axs[1, 1].plot(x, kernels[filters[3]][0])
    axs[1, 1].plot(x, pre_kernels[filters[3]][0], "--", alpha=0.5)

    img_name = "filter_kernel_ensemble2.%s" % (args.filetype)
    img_path = str(args.out_folder / img_name)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close(fig)
    print("Plotted %s" % img_path)


def plot_filters(indices, filename, F_mins, F_maxs, output_folder):
    """Plot filters bandwidths.

    Args:
        indices: Sorted indices of filters.
        filename: Output filename (png or svg).
        F_mins: Minimum frequencies.
        F_maxs: Maximum frequencies.
        output_folder: Output folder.
    """
    x = np.linspace(0, np.max(F_maxs), np.max(F_maxs) + 1)
    plt.clf()
    height = 1
    for i in indices:
        y = np.zeros_like(x)
        y[F_mins[i] : F_maxs[i]] = height
        height += 1
        plt.plot(x, y)
    img_path = str(output_folder / filename)
    plt.savefig(img_path, bbox_inches="tight")
    print("Plotted %s" % img_path)


def main(argv):
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args(argv)
    model_path = args.model_path
    sample_rate = args.sample_rate

    model = torch.load(model_path, map_location="cpu")
    if "model" in model:  # snapshots vs. model.acc.best
        model = model["model"]
    if args.filter_key not in model:
        raise ValueError(
            f"The loaded model file does not contain the learned"
            f" filters in {args.filter_key}"
        )
    filters = model[args.filter_key]
    if not filters.type() == "torch.FloatTensor":
        raise TypeError("The loaded filter values are not of type torch.FloatTensor")
    filters = filters.detach().cpu().numpy()
    f_mins = np.abs(filters[:, 0])
    f_maxs = np.abs(filters[:, 0]) + np.abs(filters[:, 1] - filters[:, 0])
    F_mins, F_maxs = f_mins * sample_rate, f_maxs * sample_rate
    F_mins, F_maxs = np.round(F_mins).astype(np.int), np.round(F_maxs).astype(np.int)

    # Create output folder if it does not yet exist
    args.out_folder.mkdir(parents=True, exist_ok=True)

    plot_filter_kernels(filters, sample_rate, args)

    plot_filters(
        range(len(F_mins)),
        "filters.%s" % args.filetype,
        F_mins,
        F_maxs,
        args.out_folder,
    )
    plot_filters(
        np.argsort(F_maxs - F_mins),
        "filters_len_sort.%s" % args.filetype,
        F_mins,
        F_maxs,
        args.out_folder,
    )
    plot_filters(
        np.argsort(F_mins),
        "filters_min_sort.%s" % args.filetype,
        F_mins,
        F_maxs,
        args.out_folder,
    )

    img_path = str(args.out_folder / f"filtergraph.{args.filetype}")
    plot_filtergraph(
        filters, sample_rate=sample_rate, img_path=img_path, scale=args.scale
    )
    img_path = str(args.out_folder / f"filtergraph_unsorted.{args.filetype}")
    plot_filtergraph(
        filters,
        sample_rate=sample_rate,
        img_path=img_path,
        sorted=False,
        scale=args.scale,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
