#!/usr/bin/env python3

# Copyright 2022 Nanyang Technological University (Kwok Chin Yuen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import kaldiio
import numpy as np
import resampy
import torch
from typeguard import check_argument_types

from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.transform.spectrogram import logmelspectrogram
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper


def build_asr_model(
    enh_s2t_task: bool = False,
    quantize_asr_model: bool = False,
    quantize_modules: List[str] = ["Linear"],
    quantize_dtype: str = "qint8",
    asr_train_config: Union[Path, str] = None,
    asr_model_file: Union[Path, str] = None,
    device: str = "cpu",
    dtype: str = np.float32,
):
    task = ASRTask if not enh_s2t_task else EnhS2TTask

    if quantize_asr_model:
        if quantize_dtype == "float16" and torch.__version__ < LooseVersion("1.5.0"):
            raise ValueError(
                "float16 dtype for dynamic quantization is not supported with "
                "torch version < 1.5.0. Switch to qint8 dtype instead."
            )

    quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
    quantize_dtype = getattr(torch, quantize_dtype)

    # 1. Build ASR model
    asr_model, asr_train_args = task.build_model_from_file(
        asr_train_config, asr_model_file, device
    )
    if enh_s2t_task:
        asr_model.inherite_attributes(
            inherite_s2t_attrs=[
                "ctc",
                "decoder",
                "eos",
                "joint_network",
                "sos",
                "token_list",
                "use_transducer_decoder",
            ]
        )
    asr_model.to(dtype=getattr(torch, dtype)).eval()

    if quantize_asr_model:
        logging.info("Use quantized asr model for decoding.")

        asr_model = torch.quantization.quantize_dynamic(
            asr_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
        )

    decoder = asr_model.decoder

    return asr_model, asr_train_args


class Speech2Feat:
    def __init__(
        self,
        data_path_and_name_and_type: str,
        enh_s2t_task: bool = False,
        quantize_asr_model: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        device: str = "cpu",
        key_file: str = None,
        batch_size: int = 1,
        dtype: str = np.float32,
        num_workers: int = 1,
        allow_variable_data_keys: bool = False,
        ngpu: int = 0,
        inference: bool = False,
        output_modules: List[str] = ["encoder.encoders.*"],
        skip_pad_modules: List[str] = [],
    ):

        # 1. Build ASR model
        self.asr_model, self.asr_train_args = build_asr_model(
            enh_s2t_task=enh_s2t_task,
            quantize_asr_model=quantize_asr_model,
            quantize_modules=quantize_modules,
            quantize_dtype=quantize_dtype,
            asr_train_config=asr_train_config,
            asr_model_file=asr_model_file,
            device=device,
            dtype=dtype,
        )

        # Register output hook for modules
        self.register_hook(self.asr_model, key_paths=output_modules)

        # 2. Build data-iterator
        self.loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=ASRTask.build_preprocess_fn(self.asr_train_args, False),
            collate_fn=ASRTask.build_collate_fn(self.asr_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.skip_pad = [] # Name of features that should not be unpadded
        self.pad_dict = {} # Dimensions to pad for each padded tensor
        self.unpad_search_done = False
        self.feats_dict = None

    def register_hook(self, module, path="", key_paths=[]):
        re_path = lambda l: "^" + l.replace(".", r"\.").replace("*", r"\w+") + "$"
        if any([re.match(re_path(key_path), path) for key_path in key_paths]):
            module.register_forward_hook(self.save_outputs_hook(path))
            logging.info(f"Registered output hook for {path}")

        for n, p in module.named_children():
            new_path = ".".join(list(filter(len, path.split("."))) + [n])
            self.register_hook(p, path=new_path, key_paths=key_paths)

    def save_outputs_hook(self, path: str) -> Callable:
        def fn(_, __, output):
            # If output is a tuple, we assume the first element
            # is the target and extract it recursively,
            # until we get a tensor
            while isinstance(output, tuple):
                output = output[0]
                if not isinstance(output, tuple):
                    logging.warning(
                        f"{path} has tuple output, "
                        + "only the tensor from the 1st element is stored"
                    )

            self.feats_dict[path] = output

        return fn

    @torch.no_grad()
    def compute_feats(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
    ):
        """Extract model activations

        Args:
            data: Input speech data
        Returns:
            feats_dict: Dict{'feats_1': torch.tensor() or List(torch.tensor), ...}

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        batch = {"speech": speech, "speech_lengths": speech_lengths}

        self.feats_dict = {}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        _, encoder_out_lens = self.asr_model.encode(**batch)

        # If we use a batch size larger than one, try to unpad the tensors
        if self.batch_size > 1:
            encoder_max_len = encoder_out_lens.max().item()
            for k, v in self.feats_dict.items():
                if "encoder" in k and k not in self.skip_pad:
                    try:
                        pad_dims = set([dim for dim, len_ in enumerate(v.size()) 
                            if len_ == encoder_max_len])
                        if k in self.pad_dict.keys():
                            pad_dims = self.pad_dict[k].intersection(pad_dims)
                        if len(pad_dims) == 0:
                            raise ValueError("No padding dimension found. Unpadding failed.")
                        self.pad_dict[k] = pad_dims

                        unpad_feats = []
                        for i, feat in enumerate(v):
                            for pad_dim in pad_dims:
                                split = [
                                    encoder_out_lens[i].item(),
                                    v.size(pad_dim) - encoder_out_lens[i].item(),
                                ]
                                feat = feat.split(split, dim=pad_dim - 1)[0]
                            unpad_feats.append(feat)
                        self.feats_dict[k] = unpad_feats
                    except:
                        if not self.unpad_search_done:
                            self.skip_pad.append(k)
                        else:
                            raise ValueError(
                                f"Error unpadding {k}, use"
                                + f"--skip_pad_modules=[{k}] to disable unpadding."
                            )

    def __iter__(self):
        """Iterate through dataset to compute model activations

        Returns:
            key: utterance ID
            feats: Dict["feat1": List(torch.tensor) or torch.tensor, ...]

        """
        for keys, batch in self.loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            self.compute_feats(**batch)

            yield keys, self.feats_dict

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build ESPnetASRModel instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            asr_model: ESPnetASRModel instance.

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

        return Speech2Feat(**kwargs)


def inference(
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    output_dir: str,
    model_tag: Optional[str],
    num_workers: int = 1,
    filetype: str = "mat",
    compress: bool = False,
    compression_method: int = 2,
    verbose: int = 1,
    enh_s2t_task: bool = False,
    quantize_asr_model: bool = False,
    quantize_modules: List[str] = ["Linear"],
    quantize_dtype: str = "qint8",
    asr_train_config: Union[Path, str] = None,
    asr_model_file: Union[Path, str] = None,
    device: str = "cpu",
    key_file: str = None,
    batch_size: int = 1,
    dtype: str = np.float32,
    allow_variable_data_keys: bool = False,
    ngpu: int = 0,
    seed: int = 0,
    inference: bool = False,
    output_modules: List[str] = ["encoder.encoders.*"],
    skip_pad_modules: List[str] = [],
):
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2feat
    speech2feat_kwargs = dict(
        data_path_and_name_and_type=data_path_and_name_and_type,
        enh_s2t_task=enh_s2t_task,
        quantize_asr_model=quantize_asr_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        device=device,
        key_file=key_file,
        batch_size=batch_size,
        dtype=dtype,
        num_workers=num_workers,
        allow_variable_data_keys=allow_variable_data_keys,
        ngpu=ngpu,
        inference=inference,
        output_modules=output_modules,
        skip_pad_modules=skip_pad_modules,
    )
    speech2feat = Speech2Feat.from_pretrained(
        model_tag=model_tag,
        **speech2feat_kwargs,
    )

    # Compute features a few times to:
    # 1. Find features that cannot be padded
    # 2. Find all the feature names returned
    iter_ = iter(speech2feat)
    for i in range(100):
        try:
            _, feats_dict = next(iter_)
        except StopIteration:
            pass
    speech2feat.unpad_search_done = True
    if batch_size > 1 and len(speech2feat.skip_pad) > 0:
        logging.info(f"Skip unpadding {speech2feat.skip_pad} as it is unsuccessful")

    # Open a writer for each feature type
    os.makedirs(output_dir, exist_ok=True)
    writers = {}
    for feat_type, feats in feats_dict.items():
        writers[feat_type] = file_writer_helper(
            "ark,scp:"
            + os.path.join(output_dir, f"{feat_type}.ark")
            + ","
            + os.path.join(output_dir, f"{feat_type}.scp"),
            filetype=filetype,
            compress=compress,
            compression_method=compression_method,
        )

    for utt_ids, feats_dict in speech2feat:
        for utt_idx, utt_id in enumerate(utt_ids):
            for feat_type, feats in feats_dict.items():
                logging.info(f"{feat_type}: " + str(feats[utt_idx].size()))
                writers[feat_type][utt_id] = feats[utt_idx].cpu().numpy()

    # Close all writers
    for writer in writers.values():
        writer.close()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Precompute ASR model features from dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--output_modules",
        type=str,
        nargs="*",
        default=["encoder.encoders.*"],
        help="""List of module outputs to precompute.
        E.g.: --output_modules=[encoder.encoders.*].
        encoder and encoders are module names, and * means any module.""",
    )
    parser.add_argument(
        "--skip_pad_modules",
        type=str,
        nargs="*",
        default=[],
        help="""Skip unpadding specific module outputs
        E.g.: --skip_pad_modules=[encoder.encoders.0].
        Stop unpadding the specific module output if it is undesirable.
        Only used when batch size > 1.""",
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
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and asr joint model",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_asr_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to ASR model.",
    )
    group.add_argument(
        "--quantize_modules",
        type=str,
        nargs="*",
        default=["Linear"],
        help="""List of modules to be dynamically quantized.
        E.g.: --quantize_modules=[Linear,LSTM,GRU].
        Each specified module should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    group.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    group = parser.add_argument_group("compression related")
    parser.add_argument(
        "--filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5"],
        help="Specify the file format for output. "
        '"mat" is the matrix format in kaldi, but only supports storing 2D array',
    )
    parser.add_argument(
        "--compress", type=str2bool, default=False, help="Save in compressed format"
    )
    parser.add_argument(
        "--compression-method",
        type=int,
        default=2,
        help="Specify the method(if mat) or " "gzip-level(if hdf5)",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
