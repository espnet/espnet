#!/usr/bin/env python3

"""TTS mode decoding."""

import logging
from pathlib import Path
import sys
import time
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import kaldiio
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.tts import TTSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class Text2Speech:
    """Text2speech class

    Examples:
        >>> text2speech = Text2Speech("config.yaml", "tts.pth")
        >>> fs, wav, outs, outs_denorm, probs, att_ws = text2speech("Hello World")

        Note that text cleaner, g2p and vocoder are also included in this class.

    """

    def __init__(
        self,
        train_config: Union[Path, str],
        model_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 0.0,
        vocoder_conf: dict = None,
    ):
        # 1. Build model
        model, train_args = TTSTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()

        # 2. Build converter from spectrogram to waveform
        if vocoder_conf is None:
            vocoder_conf = {}
        if model.feats_extract is not None:
            vocoder_conf.update(model.feats_extract.get_parameters())
        if (
            "n_fft" in vocoder_conf
            and "n_shift" in vocoder_conf
            and "fs" in vocoder_conf
        ):
            # Now supporting only griffin_lim vocoder
            spc2wav = Spectrogram2Waveform(**vocoder_conf)
            logging.info(f"Vocoder: {spc2wav}")
        else:
            spc2wav = None
            logging.warning("Vocoder is not used because vocoder_conf is insufficient")
        self.spc2wav = spc2wav

        self.tts = model.tts
        self.normalize = model.normalize
        self.train_args = train_args
        self.device = device
        self.threshold = threshold
        self.minlenratio = minlenratio
        self.maxlenratio = maxlenratio
        self.preprocess_fn = TTSTask.build_preprocess_fn(train_args, False)
        logging.info(f"Normalization:\n{self.normalize}")
        logging.info(f"TTS:\n{self.tts}")

    def __call__(self, data: Union[dict, torch.Tensor, np.ndarray, str]):
        assert check_argument_types()

        if isinstance(data, dict):
            # data comes from data loader

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {
                k: v.squeeze(0) for k, v in data.items() if not k.endswith("_lengths")
            }
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            batch = {"text": data}
        elif isinstance(data, str):
            batch = {"text": data}
            # Text to numpy array
            batch = self.preprocess_fn("<dummy>", batch)
        else:
            raise TypeError(f"dict, torch.Tensor, or np.ndarray: {type(data)}")
        batch = to_device(batch, self.device)

        # TODO(kamo): Now att_ws is not used.
        with torch.no_grad():
            outs, probs, att_ws = self.tts.inference(
                **batch,
                threshold=self.threshold,
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
            )

        insize = next(iter(batch.values())).size(0)
        logging.info(f"size: {insize}->{outs.size(0)}")
        if outs.size(0) == insize * self.maxlenratio:
            logging.warning("output length reaches maximum length.")

        if self.normalize is not None:
            outs_denorm = self.normalize.inverse(outs[None])[0][0]
        else:
            outs_denorm = outs

        if self.spc2wav is not None:
            wav = self.spc2wav(outs_denorm.cpu().numpy())
            fs = self.spc2wav.fs
        else:
            wav = None
            fs = None

        return fs, wav, outs, outs_denorm, probs, att_ws


@torch.no_grad()
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
    train_config: str,
    model_file: Optional[str],
    threshold: float,
    minlenratio: float,
    maxlenratio: float,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    allow_variable_data_keys: bool,
    vocoder_conf: dict,
):
    """Perform TTS model decoding."""
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

    # 2. Build text2speech
    text2speech = Text2Speech(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
        threshold=threshold,
        minlenratio=minlenratio,
        maxlenratio=maxlenratio,
        vocoder_conf=vocoder_conf,
    )

    # 3. Build data-iterator
    loader = TTSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=TTSTask.build_preprocess_fn(text2speech.train_args, False),
        collate_fn=TTSTask.build_collate_fn(text2speech.train_args),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )
    # 4. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)

    # FIXME(kamo): I think we shouldn't depend on kaldi-format any more.
    #  How about numpy or HDF5?
    #  >>> with NpyScpWriter() as f:
    with kaldiio.WriteHelper(
        "ark,scp:{o}.ark,{o}.scp".format(o=output_dir / "norm/feats")
    ) as f, kaldiio.WriteHelper(
        "ark,scp:{o}.ark,{o}.scp".format(o=output_dir / "denorm/feats")
    ) as g:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = to_device(batch, device)

            start_time = time.perf_counter()
            fs, wav, outs, outs_denorm, probs, att_ws = text2speech(batch)
            key = keys[0]
            logging.info(
                "{}: inference speed = {} msec / frame.".format(
                    key, (time.perf_counter() - start_time) / (int(outs.size(0)) * 1000)
                )
            )
            f[key] = outs.cpu().numpy()
            g[key] = outs_denorm.cpu().numpy()

            # TODO(kamo): Write scp
            if wav is not None:
                sf.write(f"{output_dir}/wav/{key}.wav", wav, fs, "PCM_16")


def get_parser():
    """Get argument parser."""
    parser = configargparse.ArgumentParser(
        description="TTS Decode",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

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
        "--threshold", type=float, default=0.5, help="Threshold value in decoding",
    )
    group.add_argument(
        "--use_att_constraint",
        type=str2bool,
        default=False,
        help="Whether to use attention constraint",
    )
    group.add_argument(
        "--backward_window",
        type=int,
        default=1,
        help="Backward window value in attention constraint",
    )
    group.add_argument(
        "--forward_window",
        type=int,
        default=3,
        help="Forward window value in attention constraint",
    )

    group = parser.add_argument_group(" Grriffin-Lim related")
    group.add_argument(
        "--vocoder_conf",
        action=NestedDictAction,
        default=get_default_kwargs(Spectrogram2Waveform),
        help="The configuration for Grriffin-Lim",
    )
    return parser


def main(cmd=None):
    """Run TTS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
