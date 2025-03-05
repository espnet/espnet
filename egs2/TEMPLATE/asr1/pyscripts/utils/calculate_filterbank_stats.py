#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import torch
from typeguard import typechecked
from tqdm import tqdm

from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.utils import config_argparse
from espnet.utils.cli_utils import get_commandline_args
import torchaudio.compliance.kaldi as ta_kaldi


def compute_filterbank(
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Compute filterbank from raw audio."""
    assert (
        waveform.max() <= 1.0 and waveform.min() >= -1.0
    ), "waveform should be normalized to be within [-1,1]."
    " This is usually automatic if wav is in float32 format."

    waveform = waveform.unsqueeze(0) * 2**15  # float32 to int16
    fbank = ta_kaldi.fbank(
        waveform,
        num_mel_bins=128,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
    )
    return fbank


@typechecked
def compute_filterbank_statistics(
    output_file: str,
    input_wav_scp: str,
):
    """Compute filterbank features."""

    n = torch.tensor(0, dtype=torch.float32)
    s = torch.tensor(0.0, dtype=torch.float32)
    ss = torch.tensor(0.0, dtype=torch.float32)

    sound_scp_reader = SoundScpReader(input_wav_scp)
    mean = 0
    std = 0
    for i, key in enumerate(
        tqdm(sound_scp_reader, desc="Calculating filterbank stats")
    ):
        _, sound = sound_scp_reader[key]
        sound = torch.tensor(sound, dtype=torch.float32)
        fbanks = compute_filterbank(waveform=sound)
        fbanks = fbanks.flatten()
        n += fbanks.size(0)
        s += fbanks.sum()
        ss += (fbanks**2).sum()

    mean = s / n
    variance = (ss / n) - (mean**2)
    std = torch.sqrt(variance)
    mean, std = mean.item(), std.item()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Compute speech metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The file to save the computed metrics in.",
    )
    parser.add_argument(
        "--input_wav_scp", type=str, required=True, help="The input wav scp file."
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    compute_filterbank_statistics(**kwargs)


if __name__ == "__main__":
    main()
