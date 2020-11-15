#!/usr/bin/env python3
#  2020, Technische Universität München; Nicolas Lindae, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Visualize Sinc convolution filters.

This program loads a pretrained Sinc convolution of an ESPnet2 ASR model and
plots filters, as well as the bandpass frequencies.

Plots are saved to the specified output directory.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description="create plots of the sinc filters from a trained network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate")
    parser.add_argument(
        "--all", action="store_true", help="Plot every filter in its own plot"
    )
    parser.add_argument(
        "--filetype", type=str, default="png", help="Filetype (e.g. svg or png)"
    )
    parser.add_argument(
        "model_path", type=str, help="Torch checkpoint of the trained ASR net"
    )
    parser.add_argument(
        "out_folder",
        type=str,
        nargs="?",
        default=os.getcwd(),
        help="Output folder to save the plots in",
    )
    return parser


def plot_filtergraph(filters, sample_rate, img_path, sorted=True, logscale=False):
    """Plot the Sinc filter bandpass frequencies.

    :param filters: Filter parameters
    :param sample_rate: Sample rate of signal
    :param img_path: Output plot file
    :param sorted: Sort bandpasses by center frequency.
    :param logscale: Set Y axis to logarithmic scale.
    :return:
    """

    def convert(f1, f2):
        f_mins = np.abs(f1) * sample_rate
        f_maxs = (np.abs(f1) + np.abs(f2 - f1)) * sample_rate
        f_mins = np.clip(f_mins, 0, sample_rate / 2)
        f_maxs = np.clip(f_maxs, 0, sample_rate / 2)
        f_mids = (f_maxs + f_mins) / 2
        if sorted:
            order = np.argsort(f_mids)
            f_mins, f_mids, f_maxs = f_mins[order], f_mids[order], f_maxs[order]
        return f_mins, f_maxs, f_mids

    def mel(x):
        return 1125 * np.log(1 + x / 700)

    def hz(x):
        return 700 * (np.exp(x / 1125) - 1)

    # mel filters
    fs = np.linspace(mel(30), mel(sample_rate * 0.5), len(filters) + 2)
    fs = hz(fs) / sample_rate
    f1, f2 = fs[:-2], fs[2:]

    f_mins, f_maxs, f_mids = convert(filters[:, 0], filters[:, 1])
    mel_mins, mel_maxs, mel_mids = convert(f1, f2)

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


def plot_filter_kernels(filters, sample_rate, args):
    """Plot the Sinc filter kernels.

    :param filters: Filter parameters
    :param sample_rate: Sample rate of Signal
    :param args: Dictionary with output options.
    :return:
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
    F_mins, F_maxs = np.clip(F_mins, 0, 8000), np.clip(F_maxs, 0, 8000)

    x_f = np.linspace(0, np.max(F_maxs), np.max(F_maxs) + 1)
    x = np.arange(kernels.shape[2])
    if args.all:
        for i in range(len(kernels)):
            pre_kernel = pre_kernels[i][0]
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.plot(x, pre_kernel)
            img_name = "filter_pre_kernel_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = os.path.join(args.out_folder, img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            kernel = kernels[i][0]
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.plot(x, kernel)
            img_name = "filter_kernel_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = os.path.join(args.out_folder, img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            plt.clf()
            plt.xlabel("kernel index")
            plt.plot(x, kernel)
            plt.plot(x, pre_kernel, "--", alpha=0.5)
            img_name = "filter_kernel_both_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = os.path.join(args.out_folder, img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            y = np.zeros_like(x_f)
            y[F_mins[i] : F_maxs[i]] = 1.0
            plt.clf()
            plt.plot(x_f, y)
            img_name = "filter_freq_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = os.path.join(args.out_folder, img_name)
            plt.savefig(img_path, bbox_inches="tight")
            print("Plotted %s" % img_path)

            pre_y = np.zeros_like(x_f)
            pre_y[pre_F_mins[i] : pre_F_maxs[i]] = 1.0
            plt.clf()
            plt.plot(x_f, y)
            plt.plot(x_f, pre_y)
            img_name = "filter_freq_both_%s.%s" % (str(i).zfill(2), args.filetype)
            img_path = os.path.join(args.out_folder, img_name)
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
    img_path = os.path.join(args.out_folder, img_name)
    plt.savefig(img_path, bbox_inches="tight")
    print("Plotted %s" % img_path)


def main():
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args()
    model_path = args.model_path
    sample_rate = args.sample_rate

    model = torch.load(model_path, map_location="cpu")
    if "model" in model:  # snapshots vs. model.acc.best
        model = model["model"]
    filters = model["preencoder.filters.f"].detach().cpu().numpy()
    f_mins = np.abs(filters[:, 0])
    f_maxs = np.abs(filters[:, 0]) + np.abs(filters[:, 1] - filters[:, 0])
    F_mins, F_maxs = f_mins * sample_rate, f_maxs * sample_rate
    F_mins, F_maxs = np.round(F_mins).astype(np.int), np.round(F_maxs).astype(np.int)

    plot_filter_kernels(filters, sample_rate, args)

    def plot_filters(indices, filename):
        x = np.linspace(0, np.max(F_maxs), np.max(F_maxs) + 1)
        plt.clf()
        height = 1
        for i in indices:
            y = np.zeros_like(x)
            y[F_mins[i] : F_maxs[i]] = height
            height += 1
            plt.plot(x, y)
        img_path = os.path.join(args.out_folder, filename)
        plt.savefig(img_path, bbox_inches="tight")
        print("Plotted %s" % img_path)

    plot_filters(range(len(F_mins)), "filters.%s" % args.filetype)
    plot_filters(np.argsort(F_maxs - F_mins), "filters_len_sort.%s" % args.filetype)
    plot_filters(np.argsort(F_mins), "filters_min_sort.%s" % args.filetype)

    img_path = os.path.join(args.out_folder, "filtergraph.%s" % args.filetype)
    plot_filtergraph(filters, sample_rate, img_path=img_path)
    img_path = os.path.join(args.out_folder, "filtergraph_unsorted.%s" % args.filetype)
    plot_filtergraph(filters, sample_rate, img_path=img_path, sorted=False)


if __name__ == "__main__":
    main()
