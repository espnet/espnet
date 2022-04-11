#!/usr/bin/env python3
import argparse
import itertools
import logging
import os
import sys
from typing import List
from typing import Union

import kaldiio
from mir_eval.separation import bss_eval_sources
import numpy as np
from pystoi import stoi
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.train.dataset import AdapterForSoundScpReader
from espnet2.utils import config_argparse


si_snr_loss = SISNRLoss()


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
):
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    keys = [
        line.rstrip().split(maxsplit=1)[0] for line in open(key_file, encoding="utf-8")
    ]

    with open(ref_scp[0], "r") as f:
        line = f.readline()
        filename = os.path.basename(line.split()[1])
        if ".ark" in filename:
            ref_audio_format = "kaldi_ark"
        else:
            ref_audio_format = "sound"

    if ref_audio_format == "kaldi_ark":
        ref_readers = [AdapterForSoundScpReader(kaldiio.load_scp(f, max_cache_fd=0), dtype) for f in ref_scp]
        # get sample rate
        sample_rate, _ = ref_readers[0].loader[keys[0]]
    else:
        ref_readers = [SoundScpReader(f, dtype=dtype, normalize=True) for f in ref_scp]
        # get sample rate
        sample_rate, _ = ref_readers[0][keys[0]]

    with open(inf_scp[0], "r") as f:
        line = f.readline()
        filename = os.path.basename(line.split()[1])
        if ".ark" in filename:
            inf_audio_format = "kaldi_ark"
        else:
            inf_audio_format = "sound"

    if inf_audio_format == "kaldi_ark":
        inf_readers = [AdapterForSoundScpReader(kaldiio.load_scp(f, max_cache_fd=0), dtype, fname=f) for f in inf_scp]
    else:
        inf_readers = [SoundScpReader(f, dtype=dtype, normalize=True) for f in inf_scp]

    # check keys
    for inf_reader, ref_reader in itertools.product(inf_readers, ref_readers):
        assert inf_reader.keys() == ref_reader.keys()

    with DatadirWriter(output_dir) as writer:
        for key in keys:
            if ref_audio_format == "kaldi_ark":
                ref_audios = [ref_reader[key] for ref_reader in ref_readers]
            else:
                ref_audios = [ref_reader[key][1] for ref_reader in ref_readers]
            if inf_audio_format == "kaldi_ark":
                inf_audios = [inf_reader[key] for inf_reader in inf_readers]
            else:
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
                    "Reference must be multi-channel when the \
                    network output is multi-channel."
                )
            elif ref.ndim == inf.ndim == 3:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]

            all_combinations = itertools.combinations(range(len(inf_audios)), r=len(ref_audios))
            result = None
            for comb in all_combinations:
                tmp_inf = inf[list(comb)]
                if any([np.sum(tmp_inf[x]) == 0 for x in range(2)]):
                    continue
                sdr, sir, sar, perm = bss_eval_sources(ref, tmp_inf, compute_permutation=True)
                if result is None:
                    result = dict(
                        sdr=sdr, sir=sir, sar=sar, perm=[comb[perm[i]] for i in range(len(ref_audios))], comb=comb
                    )
                else:
                    if sum(sdr) > sum(result["sdr"]):
                        result.update(
                            dict(
                                sdr=sdr, sir=sir, sar=sar, perm=[comb[perm[i]] for i in range(len(ref_audios))], comb=comb
                            )
                        )
            if result is not None:
                sdr, sir, sar, perm = result["sdr"], result["sir"], result["sar"], result["perm"]
            else:
                sdr, sir, sar, perm = [0, 0], [0, 0], [0, 0], [0, 0]

            for i in range(len(ref)):
                stoi_score = stoi(ref[i], inf[int(perm[i])], fs_sig=sample_rate)
                si_snr_score = -float(
                    si_snr_loss(
                        torch.from_numpy(ref[i][None, ...]),
                        torch.from_numpy(inf[int(perm[i])][None, ...]),
                    )
                )
                writer[f"STOI_spk{i + 1}"][key] = str(stoi_score)
                writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr_score)
                writer[f"SDR_spk{i + 1}"][key] = str(sdr[i])
                writer[f"SAR_spk{i + 1}"][key] = str(sar[i])
                writer[f"SIR_spk{i + 1}"][key] = str(sir[i])
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
    group.add_argument("--ref_channel", type=int, default=0)

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
