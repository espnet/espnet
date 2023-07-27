#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import soundfile as sf
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.fileio.read_text import read_2columns_text
from espnet2.tasks.aai import AAITask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class AAI:
    def __init__(
        self,
        aai_train_config: Union[Path, str] = None,
        aai_model_file: Union[Path, str] = None,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
    ):
        """Initialize Text2Speech module."""
        assert check_argument_types()

        # setup model
        model, train_args = AAITask.build_model_from_file(
            aai_train_config, aai_model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.aai_model = model
        self.aai_train_args = train_args
        decode_conf = {}
        self.decode_conf = decode_conf

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run text-to-speech."""
        assert check_argument_types()

        # check inputs
        if speech is None:
            raise RuntimeError("Missing required argument: 'speech'")

        # prepare batch
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        batch = to_device(batch, device=self.device)

        output, lens = self.aai_model.encode(**batch)
        output = self.aai_model.decoder(output, lens)
        return output, lens

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Text2Speech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            Text2Speech: Text2Speech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return AAI(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    score: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    aai_train_config: Optional[str],
    aai_model_file: Optional[str],
    model_tag: Optional[str],
    minlenratio: float,
    maxlenratio: float,
    allow_variable_data_keys: bool,
    decoder_weight=1.0,
):
    """Run text-to-speech inference."""
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    aai_kwargs = dict(
        aai_train_config=aai_train_config,
        aai_model_file=aai_model_file,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    aai = AAI.from_pretrained(
        model_tag=model_tag,
        **aai_kwargs,
    )

    # 3. Build data-iterator

    loader = AAITask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=AAITask.build_preprocess_fn(aai.train_args, False),
        collate_fn=AAITask.build_collate_fn(aai.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 6. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "speech_shape").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)
    (output_dir / "probs").mkdir(parents=True, exist_ok=True)
    (output_dir / "durations").mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_rates").mkdir(parents=True, exist_ok=True)

    # Lazy load to avoid the backend error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with NpyScpWriter(
        output_dir / "pred_ema", output_dir / "pred_ema/pred.scp"
    ) as ema_writer:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            output, lengths = aai(**batch)
            assert len(output.shape) == 3
            for idx in range(len(output)):
                ema_writer[keys[idx]] = output[idx, : lengths[idx], :].cpu().numpy()


def score(args):
    p = Path(args.data_path_and_name_and_type[0][0])
    parent_folder = p.parents[0]
    refs = read_2columns_text(os.path.join(parent_folder, "text"))
    spks = read_2columns_text(os.path.join(parent_folder, "utt2spk"))
    sdublevel_cc = {}
    overall_cc = []
    for folder in os.listdir(args.output_dir):
        if not folder.startswith("output."):
            continue
        folder = os.path.join(args.output_dir, folder, "pred_ema")
        for fname in os.listdir(folder):
            fname = os.path.join(folder, fname)
            if not fname.endswith(".npy"):
                continue
            utt = Path(fname).stem
            spk = spks[utt]
            if spk not in sdublevel_cc:
                sdublevel_cc[spk] = []
            with open(fname, "rb") as f:
                hyp = np.load(f)
            ref = torch.load(refs[utt])
            min_len = min(ref.shape[0], hyp.shape[0])
            ref = ref[:min_len, :]
            hyp = hyp[:min_len, :]
            utt_cc = []
            for i in range(len(ref[0])):
                utt_cc.append(scipy.stats.pearsonr(ref[:, i], hyp[:, i])[0])
            sdublevel_cc[spk].append(utt_cc)
            overall_cc.append(utt_cc)
    overall_cc = round(np.mean(np.mean(overall_cc, axis=0)), 4)
    print(parent_folder, overall_cc)


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="TTS inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--aai_train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--aai_model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--decoder_weight",
        type=float,
        default=1.0,
    )
    group.add_argument(
        "--score",
        type=bool,
        default=False,
    )
    return parser


def main(cmd=None):
    """Run AAI model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    if args.score:
        score(args)
        return
    inference(**kwargs)


if __name__ == "__main__":
    main()
