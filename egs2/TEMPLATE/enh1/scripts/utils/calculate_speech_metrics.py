#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import List, Union

import numpy as np
import torch
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from typeguard import check_argument_types

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.utils import config_argparse
from espnet.utils.cli_utils import get_commandline_args


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
    metrics: List[str],
    frame_size: int = 512,
    frame_hop: int = 256,
):
    assert check_argument_types()
    for metric in metrics:
        assert metric in (
            "STOI",
            "ESTOI",
            "SNR",
            "SI_SNR",
            "SDR",
            "SAR",
            "SIR",
            "framewise-SNR",
        ), metric

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    assert len(ref_scp) == len(inf_scp), ref_scp
    num_spk = len(ref_scp)

    keys = [
        line.rstrip().split(maxsplit=1)[0] for line in open(key_file, encoding="utf-8")
    ]

    ref_readers = [SoundScpReader(f, dtype=dtype) for f in ref_scp]
    inf_readers = [SoundScpReader(f, dtype=dtype) for f in inf_scp]

    # get sample rate
    fs, _ = ref_readers[0][keys[0]]

    # check keys
    for inf_reader, ref_reader in zip(inf_readers, ref_readers):
        assert inf_reader.keys() == ref_reader.keys()

    stft = STFTEncoder(n_fft=frame_size, hop_length=frame_hop)

    do_bss_eval = "SDR" in metrics or "SAR" in metrics or "SIR" in metrics
    with DatadirWriter(output_dir) as writer:
        for key in keys:
            ref_audios = [ref_reader[key][1] for ref_reader in ref_readers]
            inf_audios = [inf_reader[key][1] for inf_reader in inf_readers]
            ref = np.array(ref_audios)
            inf = np.array(inf_audios)
            if ref.ndim > inf.ndim:
                # multi-channel reference and single-channel output
                ref = ref[..., ref_channel]
                assert ref.shape == inf.shape, (ref.shape, inf.shape)
            elif ref.ndim < inf.ndim:
                # single-channel reference and multi-channel output
                raise ValueError(
                    "Reference must be multi-channel when the "
                    "network output is multi-channel."
                )
            elif ref.ndim == inf.ndim == 3:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]

            if do_bss_eval or num_spk > 1:
                sdr, sir, sar, perm = bss_eval_sources(
                    ref, inf, compute_permutation=True
                )
            else:
                perm = [0]

            ilens = torch.LongTensor([ref.shape[1]])
            # (num_spk, T, F)
            ref_spec, flens = stft(torch.from_numpy(ref), ilens)
            inf_spec, _ = stft(torch.from_numpy(inf), ilens)

            for i in range(num_spk):
                p = int(perm[i])
                for metric in metrics:
                    name = f"{metric}_spk{i + 1}"
                    if metric == "STOI":
                        writer[name][key] = str(
                            stoi(ref[i], inf[p], fs_sig=fs, extended=False)
                        )
                    elif metric == "ESTOI":
                        writer[name][key] = str(
                            stoi(ref[i], inf[p], fs_sig=fs, extended=True)
                        )
                    elif metric == "SNR":
                        si_snr_score = -float(
                            ESPnetEnhancementModel.snr_loss(
                                torch.from_numpy(ref[i][None, ...]),
                                torch.from_numpy(inf[p][None, ...]),
                            )
                        )
                        writer[name][key] = str(si_snr_score)
                    elif metric == "SI_SNR":
                        si_snr_score = -float(
                            ESPnetEnhancementModel.si_snr_loss(
                                torch.from_numpy(ref[i][None, ...]),
                                torch.from_numpy(inf[p][None, ...]),
                            )
                        )
                        writer[name][key] = str(si_snr_score)
                    elif metric == "SDR":
                        writer[name][key] = str(sdr[i])
                    elif metric == "SAR":
                        writer[name][key] = str(sar[i])
                    elif metric == "SIR":
                        writer[name][key] = str(sir[i])
                    elif metric == "framewise-SNR":
                        framewise_snr = -ESPnetEnhancementModel.snr_loss(
                            ref_spec[i].abs(), inf_spec[i].abs()
                        )
                        writer[name][key] = " ".join(map(str, framewise_snr.tolist()))
                    else:
                        raise ValueError("Unsupported metric: %s" % metric)
                    # save permutation assigned script file
                    writer[f"wav_spk{i + 1}"][key] = inf_readers[perm[i]].data[key]


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str)
    group.add_argument("--metrics", type=str, action="append")
    group.add_argument("--ref_channel", type=int, default=0)
    group.add_argument(
        "--frame_size",
        type=int,
        default=512,
        help="STFT frame size in samples, for calculating framewise-* metrics",
    )
    group.add_argument(
        "--frame_hop",
        type=int,
        default=256,
        help="STFT frame hop in samples, for calculating framewise-* metrics",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    scoring(**kwargs)


if __name__ == "__main__":
    main()
