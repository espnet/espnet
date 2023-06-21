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

from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args

si_snr_loss = SISNRLoss()


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
    flexible_numspk: bool,
):
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if not flexible_numspk:
        assert len(ref_scp) == len(inf_scp), ref_scp
    num_spk = len(ref_scp)

    keys = [
        line.rstrip().split(maxsplit=1)[0] for line in open(key_file, encoding="utf-8")
    ]

    ref_readers = [SoundScpReader(f, dtype=dtype, normalize=True) for f in ref_scp]
    inf_readers = [SoundScpReader(f, dtype=dtype, normalize=True) for f in inf_scp]

    # get sample rate
    sample_rate, _ = ref_readers[0][keys[0]]

    # check keys
    if not flexible_numspk:
        for inf_reader, ref_reader in zip(inf_readers, ref_readers):
            assert inf_reader.keys() == ref_reader.keys()

    with DatadirWriter(output_dir) as writer:
        for key in keys:
            if not flexible_numspk:
                ref_audios = [ref_reader[key][1] for ref_reader in ref_readers]
                inf_audios = [inf_reader[key][1] for inf_reader in inf_readers]
            else:
                ref_audios = [
                    ref_reader[key][1]
                    for ref_reader in ref_readers
                    if key in ref_reader.keys()
                ]
                inf_audios = [
                    inf_reader[key][1]
                    for inf_reader in inf_readers
                    if key in inf_reader.keys()
                ]
            ref = np.array(ref_audios)
            inf = np.array(inf_audios)
            if ref.ndim > inf.ndim:
                # multi-channel reference and single-channel output
                ref = ref[..., ref_channel]
            elif ref.ndim < inf.ndim:
                # single-channel reference and multi-channel output
                inf = inf[..., ref_channel]
            elif ref.ndim == inf.ndim == 3:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]
            if not flexible_numspk:
                assert ref.shape == inf.shape, (ref.shape, inf.shape)
            else:
                # epsilon value to avoid divergence
                # caused by zero-value, e.g., log(0)
                eps = 0.000001
                # if num_spk of ref > num_spk of inf
                if ref.shape[0] > inf.shape[0]:
                    p = np.full((ref.shape[0] - inf.shape[0], inf.shape[1]), eps)
                    inf = np.concatenate([inf, p])
                    num_spk = ref.shape[0]
                # if num_spk of ref < num_spk of inf
                elif ref.shape[0] < inf.shape[0]:
                    p = np.full((inf.shape[0] - ref.shape[0], ref.shape[1]), eps)
                    ref = np.concatenate([ref, p])
                    num_spk = inf.shape[0]
                else:
                    num_spk = ref.shape[0]

            sdr, sir, sar, perm = bss_eval_sources(ref, inf, compute_permutation=True)

            for i in range(num_spk):
                stoi_score = stoi(ref[i], inf[int(perm[i])], fs_sig=sample_rate)
                estoi_score = stoi(
                    ref[i], inf[int(perm[i])], fs_sig=sample_rate, extended=True
                )
                si_snr_score = -float(
                    si_snr_loss(
                        torch.from_numpy(ref[i][None, ...]),
                        torch.from_numpy(inf[int(perm[i])][None, ...]),
                    )
                )
                writer[f"STOI_spk{i + 1}"][key] = str(stoi_score * 100)  # in percentage
                writer[f"ESTOI_spk{i + 1}"][key] = str(estoi_score * 100)
                writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr_score)
                writer[f"SDR_spk{i + 1}"][key] = str(sdr[i])
                writer[f"SAR_spk{i + 1}"][key] = str(sar[i])
                writer[f"SIR_spk{i + 1}"][key] = str(sir[i])
                # save permutation assigned script file
                if i < len(ref_scp):
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
    group.add_argument("--flexible_numspk", type=str2bool, default=False)

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
