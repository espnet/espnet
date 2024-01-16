#!/usr/bin/env python3
import argparse
import logging
import sys
from distutils.version import LooseVersion
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.spk import SpeakerTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class Speech2Embedding:
    """Speech2Embedding class

    Examples:
        >>> import soundfile
        >>> speech2spkembed = Speech2Embedding("spk_config.yml", "spk.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2spkembed(audio)

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        batch_size: int = 1,
    ):
        assert check_argument_types()

        spk_model, spk_train_args = SpeakerTask.build_model_from_file(
            train_config, model_file, device
        )
        self.spk_model = spk_model.eval()
        self.spk_train_args = spk_train_args
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Inference

        Args:
            speech: Input speech data

        Returns:
            spk_embedding

        """

        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        logging.info("speech length: " + str(speech.size(1)))
        batch = {"speech": speech, "extract_embd": True}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward the model embedding extraction
        output = self.spk_model(**batch)

        return output

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Embedding instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Embedding instance.

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

        return Speech2Embedding(**kwargs)


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
):
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

    # 2. Build speech2embedding
    speech2embedding_kwargs = dict(
        batch_size=batch_size,
        dtype=dtype,
        train_config=train_config,
        model_file=model_file,
    )

    speech2embedding = Speech2Embedding.from_pretrained(
        model_tag=model_tag,
        **speech2embedding_kwargs,
    )

    # 3. Build data-iterator
    loader = SpeakerTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SpeakerTask.build_preprocess_fn(
            speech2embedding.spk_train_args, False
        ),
        collate_fn=SpeakerTask.build_colate_fn(speech2embedding.spk_train_args, False),
        inference=True,
    )

    # 4. Start for-loop
    with NpyScpWriter(output_dir / "embed", output_dir / "embed.scp") as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            result = speech2embedding(**batch)

            # Only supporting batch_size==1
            key = keys[0]

            writer[key] = result.cpu().numpy()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker Embedding Extraction",
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
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
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

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Speaker model training configuration",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Speaker model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
