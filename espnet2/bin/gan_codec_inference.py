#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_codec.dac import DAC
from espnet2.gan_codec.soundstream import SoundStream
from espnet2.tasks.gan_codec import GANCodecTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import float_or_none, str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class AudioCoding:
    """AudioCoding class.

    Examples:
        >>> TODO(jiatong)
    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        target_bandwidth: Union[Path, str, None] = None,
        dtype: str = "float32",
        device: Union[str, torch.device] = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
    ):
        """Initialize AudioCoding module."""

        # setup model
        model, train_args = GANCodecTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.codec = model.codec
        self.preprocess_fn = GANCodecTask.build_preprocess_fn(train_args, False)
        self.seed = seed
        self.always_fix_seed = always_fix_seed

        decode_conf = {}
        if isinstance(self.codec, SoundStream) or isinstance(self.codec, DAC):
            decode_conf.update(
                target_bw=target_bandwidth,
            )
        self.decode_conf = decode_conf

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        audio: Union[torch.Tensor, np.ndarray, None] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
        encode_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run Audio Coding."""

        assert audio is not None, "Audio is invalid, input a valid audio."

        batch = dict(audio=audio)
        batch = to_device(batch, self.device)
        # overwrite the decode configs if provided
        cfg = self.decode_conf
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)

        codes = self.model.encode(**batch, **cfg)
        output_dict = dict(codes=codes)

        if encode_only:
            return output_dict
        else:
            # TODO(jiatong): to consider multichannel cases
            if len(audio.shape) == 1:
                resyn_audio = self.model.decode(codes).view(-1)
            else:
                resyn_audio = self.model.decode(codes)
            output_dict.update(resyn_audio=resyn_audio)
            return output_dict

    @torch.no_grad()
    @typechecked
    def decode(
        self,
        codes: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Audio Coding decoding."""

        assert codes is not None, "Codes are invalid, input a valid one"

        batch = dict(codes=codes)
        batch = to_device(batch, self.device)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)

        resyn_audio = self.model.decode(codes)
        output_dict = dict(resyn_audio=resyn_audio)
        return output_dict

    @property
    def fs(self) -> Optional[int]:
        """Return sampling rate."""
        if hasattr(self.codec, "fs"):
            return self.codec.fs
        else:
            return None

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build AudioCoding instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            AudioCoding: AudioCoding instance.

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

        return AudioCoding(**kwargs)


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    target_bandwidth: Optional[float],
    encode_only: bool,
    always_fix_seed: bool,
    allow_variable_data_keys: bool,
):
    """Run speech coding inference."""
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
    audio_coding_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        target_bandwidth=target_bandwidth,
        dtype=dtype,
        device=device,
        seed=seed,
        always_fix_seed=always_fix_seed,
    )
    audio_coding = AudioCoding.from_pretrained(
        model_tag=model_tag,
        **audio_coding_kwargs,
    )

    # 3. Build data-iterator
    loader = GANCodecTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=GANCodecTask.build_preprocess_fn(audio_coding.train_args, False),
        collate_fn=GANCodecTask.build_collate_fn(audio_coding.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    output_dir_path = Path(output_dir)
    (output_dir_path / "codec").mkdir(parents=True, exist_ok=True)
    (output_dir_path / "wav").mkdir(parents=True, exist_ok=True)

    with NpyScpWriter(
        output_dir_path / "codec", output_dir_path / "codec/codec.scp"
    ) as codec_writer:
        for _, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            batch.update(encode_only=encode_only)

            start_time = time.perf_counter()
            output_dict = audio_coding(**batch)

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            if output_dict.get("resyn_audio") is not None:
                wav = output_dict["resyn_audio"]
                # Note(jiatong): Assume the wav is single channel here
                logging.info(
                    "inference speed = {:.1f} points / sec.".format(
                        int(wav.size(0)) / (time.perf_counter() - start_time)
                    )
                )
                logging.info(f"{key} (size:{insize}->{wav.size(0)})")
                sf.write(
                    f"{output_dir_path}/wav/{key}.wav",
                    output_dict["resyn_audio"].cpu().numpy(),
                    audio_coding.fs,
                    "PCM_16",
                )

            if output_dict.get("codes") is not None:
                codec_writer[key] = output_dict["codes"].cpu().numpy()

    # remove files if those are not included in output dict
    if output_dict.get("codes") is None:
        shutil.rmtree(output_dir_path / "codes")
    if output_dict.get("resyn_audio") is None:
        shutil.rmtree(output_dir_path / "wav")


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="Codec inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )
    group.add_argument(
        "--target_bandwidth",
        type=float_or_none,
        default=None,
        help="Target bandwidth for models supporting various bandwidth",
    )
    group.add_argument(
        "--encode_only",
        type=str2bool,
        default=False,
        help="Whether to only do encoding.",
    )
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
    )
    return parser


def main(cmd=None):
    """Run Codec model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
