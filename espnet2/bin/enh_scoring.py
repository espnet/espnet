#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from typeguard import check_argument_types

from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.train.dataset import kaldi_loader
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args

si_snr_loss = SISNRLoss()


def get_readers(scps: List[str], dtype: str):
    # Determine the audio format (sound or kaldi_ark)
    with open(scps[0], "r") as f:
        line = f.readline()
        filename = Path(line.strip().split(maxsplit=1)[1]).name
    if re.fullmatch(r".*\.ark(:\d+)?", filename):
        # xxx.ark or xxx.ark:123
        readers = [kaldi_loader(f, float_dtype=dtype) for f in scps]
        audio_format = "kaldi_ark"
    else:
        readers = [SoundScpReader(f, dtype=dtype) for f in scps]
        audio_format = "sound"
    return readers, audio_format


def read_audio(reader, key, audio_format="sound"):
    if audio_format == "sound":
        return reader[key][1]
    elif audio_format == "kaldi_ark":
        return reader[key]
    else:
        raise ValueError(f"Unknown audio format: {audio_format}")


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
    flexible_numspk: bool,
    is_tse: bool,
    use_dnsmos: bool,
    dnsmos_args: Dict,
    use_pesq: bool,
):
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if use_dnsmos:
        if dnsmos_args["mode"] == "local":
            from espnet2.enh.layers.dnsmos import DNSMOS_local

            if not Path(dnsmos_args["primary_model"]).exists():
                raise ValueError(
                    f"The primary model '{dnsmos_args['primary_model']}' doesn't exist."
                    " You can download the model from https://github.com/microsoft/"
                    "DNS-Challenge/tree/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
                )
            if not Path(dnsmos_args["p808_model"]).exists():
                raise ValueError(
                    f"The P808 model '{dnsmos_args['p808_model']}' doesn't exist."
                    " You can download the model from https://github.com/microsoft/"
                    "DNS-Challenge/tree/master/DNSMOS/DNSMOS/model_v8.onnx"
                )
            dnsmos = DNSMOS_local(
                dnsmos_args["primary_model"], dnsmos_args["p808_model"]
            )
            logging.warning("Using local DNSMOS models for evaluation")

        elif dnsmos_args["mode"] == "web":
            from espnet2.enh.layers.dnsmos import DNSMOS_web

            if not dnsmos_args["auth_key"]:
                raise ValueError(
                    "Please specify the authentication key for access to the Web-API. "
                    "You can apply for the AUTH_KEY at https://github.com/microsoft/"
                    "DNS-Challenge/blob/master/DNSMOS/README.md#to-use-the-web-api"
                )
            dnsmos = DNSMOS_web(dnsmos_args["auth_key"])
            logging.warning("Using the DNSMOS Web-API for evaluation")
    else:
        dnsmos = None

    if use_pesq:
        try:
            from pesq import PesqError, pesq

            logging.warning("Using the PESQ package for evaluation")
        except ImportError:
            raise ImportError("Please install pesq and retry: pip install pesq")
    else:
        pesq = None

    if not flexible_numspk:
        assert len(ref_scp) == len(inf_scp), ref_scp
    num_spk = len(ref_scp)

    keys = [
        line.rstrip().split(maxsplit=1)[0] for line in open(key_file, encoding="utf-8")
    ]

    ref_readers, ref_audio_format = get_readers(ref_scp, dtype)
    inf_readers, inf_audio_format = get_readers(inf_scp, dtype)

    # get sample rate
    retval = ref_readers[0][keys[0]]
    if ref_audio_format == "kaldi_ark":
        sample_rate = ref_readers[0].rate
    elif ref_audio_format == "sound":
        sample_rate = retval[0]
    else:
        raise NotImplementedError(ref_audio_format)
    assert sample_rate is not None, (sample_rate, ref_audio_format)

    # check keys
    if not flexible_numspk:
        for inf_reader, ref_reader in zip(inf_readers, ref_readers):
            assert inf_reader.keys() == ref_reader.keys()

    with DatadirWriter(output_dir) as writer:
        for n, key in enumerate(keys):
            logging.info(f"[{n}] Scoring {key}")
            if not flexible_numspk:
                ref_audios = [
                    read_audio(ref_reader, key, audio_format=ref_audio_format)
                    for ref_reader in ref_readers
                ]
                inf_audios = [
                    read_audio(inf_reader, key, audio_format=inf_audio_format)
                    for inf_reader in inf_readers
                ]
            else:
                ref_audios = [
                    read_audio(ref_reader, key, audio_format=ref_audio_format)
                    for ref_reader in ref_readers
                    if key in ref_reader.keys()
                ]
                inf_audios = [
                    read_audio(inf_reader, key, audio_format=inf_audio_format)
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

            sdr, sir, sar, perm = bss_eval_sources(
                ref, inf, compute_permutation=not is_tse
            )

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
                if dnsmos:
                    dnsmos_score = dnsmos(inf[int(perm[i])], sample_rate)
                    writer[f"OVRL_spk{i + 1}"][key] = str(dnsmos_score["OVRL"])
                    writer[f"SIG_spk{i + 1}"][key] = str(dnsmos_score["SIG"])
                    writer[f"BAK_spk{i + 1}"][key] = str(dnsmos_score["BAK"])
                    writer[f"P808_MOS_spk{i + 1}"][key] = str(dnsmos_score["P808_MOS"])
                if pesq:
                    if sample_rate == 8000:
                        mode = "nb"
                    elif sample_rate == 16000:
                        mode = "wb"
                    else:
                        raise ValueError(
                            "sample rate must be 8000 or 16000 for PESQ evaluation, "
                            f"but got {sample_rate}"
                        )
                    pesq_score = pesq(
                        sample_rate,
                        ref[i],
                        inf[int(perm[i])],
                        mode=mode,
                        on_error=PesqError.RETURN_VALUES,
                    )
                    if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
                        logging.warning(
                            f"[PESQ] Error: No utterances detected for {key}. "
                            "Skipping this utterance."
                        )
                    else:
                        writer[f"PESQ_{mode.upper()}_spk{i + 1}"][key] = str(pesq_score)
                writer[f"STOI_spk{i + 1}"][key] = str(stoi_score * 100)  # in percentage
                writer[f"ESTOI_spk{i + 1}"][key] = str(estoi_score * 100)
                writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr_score)
                writer[f"SDR_spk{i + 1}"][key] = str(sdr[i])
                writer[f"SAR_spk{i + 1}"][key] = str(sar[i])
                writer[f"SIR_spk{i + 1}"][key] = str(sir[i])
                # save permutation assigned script file
                if i < len(ref_scp):
                    if inf_audio_format == "sound":
                        writer[f"wav_spk{i + 1}"][key] = inf_readers[perm[i]].data[key]
                    elif inf_audio_format == "kaldi_ark":
                        # NOTE: SegmentsExtractor is not supported
                        writer[f"wav_spk{i + 1}"][key] = inf_readers[
                            perm[i]
                        ].loader._dict[key]
                    else:
                        raise ValueError(f"Unknown audio format: {inf_audio_format}")


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
    group.add_argument("--is_tse", type=str2bool, default=False)

    group = parser.add_argument_group("DNSMOS related")
    group.add_argument("--use_dnsmos", type=str2bool, default=False)
    group.add_argument(
        "--dnsmos_mode",
        type=str,
        choices=("local", "web"),
        default="local",
        help="Use local DNSMOS model or web API for DNSMOS calculation",
    )
    group.add_argument(
        "--dnsmos_auth_key", type=str, default="", help="Required if dnsmsos_mode='web'"
    )
    group.add_argument(
        "--dnsmos_primary_model",
        type=str,
        default="./DNSMOS/sig_bak_ovr.onnx",
        help="Path to the primary DNSMOS model. Required if dnsmsos_mode='local'",
    )
    group.add_argument(
        "--dnsmos_p808_model",
        type=str,
        default="./DNSMOS/model_v8.onnx",
        help="Path to the p808 model. Required if dnsmsos_mode='local'",
    )

    group = parser.add_argument_group("PESQ related")
    group.add_argument(
        "--use_pesq",
        type=str2bool,
        default=False,
        help="Bebore setting this to True, please make sure that you or "
        "your institution have the license "
        "(check https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en) to report PESQ",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    dnsmos_args = {
        "mode": kwargs.pop("dnsmos_mode"),
        "auth_key": kwargs.pop("dnsmos_auth_key"),
        "primary_model": kwargs.pop("dnsmos_primary_model"),
        "p808_model": kwargs.pop("dnsmos_p808_model"),
    }
    kwargs["dnsmos_args"] = dnsmos_args
    scoring(**kwargs)


if __name__ == "__main__":
    main()
