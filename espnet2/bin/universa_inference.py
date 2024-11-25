#!/usr/bin/env python3

"""Inference script for ESPnet Universa model."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.universa import UniversaTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class UniversaInference:
    """Inference class for ESPnet Universa model."""

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
    ):
        """Initialize UniversaInference class."""

        # setup model
        model, train_args = UniversaTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.universa = model.universa
        self.frontend = model.frontend
        self.preprocess_fn = UniversaTask.build_preprocess_fn(train_args, False)
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        logging.info(f"Frontend: {model.frontend}")
        logging.info(f"Universa: {model.universa}")
    
    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        audio_lengths: Union[np.ndarray, torch.Tensor] = None,
        ref_audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_audio_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_text: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_text_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        "Run universa."

        # check the input type
        if self.model.use_ref_audio and ref_audio is None:
            logging.warning("Universa model pretrained with ref_audio is used.")
        if self.model.use_ref_text and ref_text is None:
            logging.warning("Universa model pretrained with ref_text is used.")
        if not self.model.use_ref_audio and ref_audio is not None:
            logging.warning("Universa model not pretrained with ref_audio is used.")
        if not self.model.use_ref_text and ref_text is not None:
            logging.warning("Universa model not pretrained with ref_text is used.")
        
        # prepare batch
        batch = dict(audio=audio, audio_lengths=audio_lengths)
        if ref_audio is not None:
            batch.update(ref_audio=ref_audio, ref_audio_lengths=ref_audio_lengths)
        if ref_text is not None:
            if isinstance(ref_text, str):
                ref_text = self.preprocess_fn("<dummy>", dict(text=ref_text))["text"]
            batch.update(ref_text=ref_text, ref_text_lengths=ref_text_lengths)
        batch = to_device(batch, device=self.device)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)
        
        output_dict = self.model.inference(**batch, **kwargs)
        return output_dict
    
    @property
    def use_ref_audio(self):
        return self.model.use_ref_audio
    
    @property
    def use_ref_text(self):
        return self.model.use_ref_text
    
    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build UniversaInference from pretrained model."""
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
        return UniversaInference(**kwargs)


@typechecked
def inference(
    output_dir: Union[Path, str],
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
    always_fix_seed: bool,
    allow_variable_data_keys: bool,
):
    """Run inference."""
    # setup logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # check the input type
    if model_file is None and model_tag is None:
        raise ValueError("model_file or model_tag must be specified.")
    if model_file is not None and model_tag is not None:
        raise ValueError("model_file and model_tag cannot be specified at the same time.")
    
    if ngpu == 0:
        device = "cpu"
    else:
        device = "cuda"

    # 1. set random seed
    set_all_random_seed(seed)
    
    # 2. setup UniversaInference (build model)
    universa_inference = UniversaInference.from_pretrained(
        model_tag=model_tag,
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        seed=seed,
        always_fix_seed=always_fix_seed,
        device=device,
    )

    # 3. setup data loader
    loader = UniversaTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        key_file=key_file,
        preprocess_fn=UniversaTask.build_preprocess_fn(universa_inference.train_args, False),
        collate_fn=UniversaTask.build_collate_fn(universa_inference.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True
    )

    # 4. start for-loop inference
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(k, str) for k in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            results = universa_inference(**batch)
            
            for i in range(_bs):
                key = keys[i]
                # NOTE(jiatong): assume the prediction target is 1-dimensional.
                metrics_info = {
                    k: float(v[i, 0]) for k, v in results.items()
                }
                writer["metric.scp"][key] = json.dumps(metrics_info)
    
def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Universa Inference",
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
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

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

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
    )
    return parser

def main(cmd=None):
    """Run Universa model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()

