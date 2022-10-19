#!/usr/bin/env python3

# Copyright 2022 Nanyang Technological University (Kwok Chin Yuen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import kaldiio
import numpy as np
import resampy
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.transform.spectrogram import logmelspectrogram
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


# Build method from espnet2/bin/asr_inference.py
def build_asr_model(
    args: argparse.Namespace,
):
    task = ASRTask if not args.enh_s2t_task else EnhS2TTask

    if args.quantize_asr_model:
        if args.quantize_dtype == "float16" and torch.__version__ < LooseVersion(
            "1.5.0"
        ):
            raise ValueError(
                "float16 dtype for dynamic quantization is not supported with "
                "torch version < 1.5.0. Switch to qint8 dtype instead."
            )

    quantize_modules = set([getattr(torch.nn, q) for q in args.quantize_modules])
    quantize_dtype = getattr(torch, args.quantize_dtype)

    # 1. Build ASR model
    asr_model, asr_train_args = task.build_model_from_file(
        args.asr_train_config, args.asr_model_file, args.device
    )
    if args.enh_s2t_task:
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
    asr_model.to(dtype=getattr(torch, args.dtype)).eval()

    if args.quantize_asr_model:
        logging.info("Use quantized asr model for decoding.")

        asr_model = torch.quantization.quantize_dynamic(
            asr_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
        )

    decoder = asr_model.decoder

    return asr_model, asr_train_args


class Speech2Feat:
    def __init__(
        self,
        args: argparse.Namespace,
    ):

        # 1a. Build ASR model as in espnet2/bin/asr_inference.py
        if hasattr(args, "asr_model_file") and args.asr_model_file is not None:
            logging.info(f"Building model from checkpoint file: {args.asr_model_file}")
            self.asr_model, self.asr_train_args = build_asr_model(
                args=args,
            )
        # 1b. Build ASR model as in espnet2/bin/asr_train.py
        else:
            logging.info(f"Building model from train configs")
            self.asr_model = ASRTask.build_model(args)
            if args.device == "cuda":
                self.asr_model = self.asr_model.cuda()
            self.asr_train_args = args
        logging.info(repr(self.asr_model))

        # 2. Register output hook for modules
        print(args.output_modules)
        self.register_hook(self.asr_model, key_paths=args.output_modules)

        # 3. Build data-iterator
        self.data_loader = ASRTask.build_streaming_iterator(
            args.data_path_and_name_and_type,
            dtype=args.dtype,
            batch_size=args.batch_size,
            key_file=args.key_file,
            num_workers=args.num_workers,
            preprocess_fn=ASRTask.build_preprocess_fn(self.asr_train_args, False),
            collate_fn=ASRTask.build_collate_fn(self.asr_train_args, False),
            allow_variable_data_keys=args.allow_variable_data_keys,
            inference=True,
        )

        self.dtype = args.dtype
        self.device = args.device
        self.batch_size = args.batch_size
        self.skip_pad = []  # Name of features that should not be unpadded
        self.pad_dict = {}  # Dimensions to pad for each padded tensor
        self.batch_dict = {}  # Batch dimension for module outputs
        self.tuple_warn = []
        self.dim_search_done = False
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
                if path not in self.tuple_warn:
                    self.tuple_warn.append(path)
                    if isinstance(output, torch.Tensor):
                        logging.warning(
                            f"{path} has tuple output, "
                            + f"only the tensor {output.size()} from the 1st element is stored"
                        )
                    else:
                        logging.warning(
                            f"{path} has tuple output, "
                            + f"only the 1st element of type {type(output)} is stored"
                        )

            """
            Custom code to convert module output of other types 
            to torch.Tensor goes here
            """ 

            # Stack multiple layers from frontend.upstream
            # Here we assume output of frontend.upstream is 
            # a list of layer tensor output
            if (
                path == "frontend.upstream"
                and isinstance(output, list)
                and isinstance(output[0], torch.Tensor)
            ):
                output = torch.stack(output, dim=1)

            """
            Custom code to convert module output of other types 
            to torch.Tensor ends here
            """ 

            # Output should always be torch.Tensor
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"{path} has output of type {type(output)}. " + 
                    f"You may need to convert the output to torch.Tensor " + 
                    f"by putting you code here.")

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
            feats_dict: Dict{'feats_1': torch.Tensor or List(torch.Tensor), ...}

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        batch = {"speech": speech, "speech_lengths": speech_lengths}

        self.feats_dict = {}

        # 1. To device
        speech = to_device(speech, device=self.device)
        speech_lengths = to_device(speech_lengths, device=self.device)

        asr_model = self.asr_model
        with autocast(False):
            # 2. Extract feats
            feats, feats_lengths = asr_model._extract_feats(speech, speech_lengths)
            frontend_out_lens = feats_lengths

            # 3. Data augmentation
            if asr_model.specaug is not None and asr_model.training:
                feats, feats_lengths = asr_model.specaug(feats, feats_lengths)

            # 4. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if asr_model.normalize is not None:
                feats, feats_lengths = asr_model.normalize(feats, feats_lengths)

        # 5. Pre-encoder, e.g. used for raw input data
        if asr_model.preencoder is not None:
            feats, feats_lengths = asr_model.preencoder(feats, feats_lengths)
            perencoder_out_lens = feats_lengths

        # 6. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if asr_model.encoder.interctc_use_conditioning:
            _, encoder_out_lens, _ = asr_model.encoder(
                feats, feats_lengths, ctc=asr_model.ctc
            )
        else:
            _, encoder_out_lens, _ = asr_model.encoder(feats, feats_lengths)

        del feats

        # If we use a batch size larger than one, try to unpad the tensors
        if self.batch_size > 1:
            frontend_max_len = frontend_out_lens.max().item()
            encoder_max_len = encoder_out_lens.max().item()

            for k, v in self.feats_dict.items():
                if "frontend" in k and k not in self.skip_pad:
                    max_len = frontend_max_len
                elif "encoder" in k and k not in self.skip_pad:
                    max_len = encoder_max_len
                else:
                    continue

                try:
                    pad_dims = set(
                        [dim for dim, len_ in enumerate(v.size()) if len_ == max_len]
                    )
                    if k in self.pad_dict.keys():
                        pad_dims = self.pad_dict[k].intersection(pad_dims)
                    if len(pad_dims) == 0:
                        raise ValueError(
                            "No padding dimension found. "
                            + f"Tensor has shape {v.size()} but "
                            + f"the guessed padded length is {max_len}"
                        )
                    self.pad_dict[k] = pad_dims

                    if self.dim_search_done:
                        batch_dim = list(self.batch_dict[k])[0]
                        unpad_feats = []
                        for batch_idx, encoder_out_len in enumerate(encoder_out_lens):
                            unpad_feat = v.select(dim=batch_dim, index=batch_idx)
                            for pad_dim in pad_dims:
                                split = [
                                    encoder_out_len.item(),
                                    v.size(pad_dim) - encoder_out_len.item(),
                                ]
                                unpad_feat = unpad_feat.split(split, dim=pad_dim - 1)[0]
                            unpad_feats.append(unpad_feat)
                        self.feats_dict[k] = unpad_feats
                except ValueError as ve:
                    if not self.dim_search_done:
                        if k not in self.skip_pad:
                            logging.warning(f"Skipped unpadding for {k}: ")
                            logging.warning(ve)
                            logging.warning(f"If this is undesirable, you may change batch size to 1")
                        self.skip_pad.append(k)
                    else:
                        raise ValueError(
                            f"Error unpadding {k}, use"
                            + f"--skip_pad_modules=[{k}] to disable unpadding."
                        )

        # Identify the batch dimension
        for k, v in self.feats_dict.items():
            if not self.dim_search_done:
                batch_dim = set(
                    [
                        dim
                        for dim, len_ in enumerate(v.size())
                        if len_ == self.batch_size
                    ]
                )
                if k in self.batch_dict.keys():
                    batch_dim = self.batch_dict[k].intersection(batch_dim)
                self.batch_dict[k] = batch_dim
            else:
                batch_dim = list(self.batch_dict[k])
                if len(batch_dim) == 0:
                    raise ValueError(
                        f"Cannot find batch dimension for tensor from "
                        + f"module output {k}"
                    )
                if len(batch_dim) > 1:
                    raise ValueError(
                        f"Found multiple batch dimension for tensor from "
                        + f"module output {k}"
                    )
                if not isinstance(self.feats_dict[k], list):
                    self.feats_dict[k] = self.feats_dict[k].transpose(0, batch_dim[0])

    def __iter__(self):
        """Iterate through dataset to compute model activations

        Returns:
            key: utterance ID
            feats: Dict["feat1": List(torch.Tensor) or torch.Tensor, ...]

        """
        for keys, batch in self.data_loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            self.compute_feats(**batch)

            yield keys, self.feats_dict

    @staticmethod
    def from_pretrained(
        args: argparse.Namespace,
        model_tag: Optional[str] = None,
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
            # kwargs.update(**d.download_and_unpack(model_tag))
            args = Namespace(**vars(args), **d.download_and_unpack(model_tag))

        return Speech2Feat(args)


def inference(
    args: argparse.Namespace,
):
    assert check_argument_types()
    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if args.ngpu >= 1:
        args.device = "cuda"
    else:
        args.device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(args.seed)

    # 2. Build speech2feat
    speech2feat = Speech2Feat.from_pretrained(
        args,
        model_tag=args.model_tag,
    )

    # Compute features a few times to:
    # 1. Find features that cannot be padded
    # 2. Find the batch dimension
    # 3. Find all the feature names returned
    iter_ = iter(speech2feat)
    for i, _ in enumerate(speech2feat.data_loader):
        if i == 0:
            continue  # Make sure it is not the last batch for iter_
        if i > 50:
            break
        try:
            _, feats_dict = next(iter_)
        except StopIteration:
            raise ValueError(
                "Something went wrong. "
                + "data_loader and iter_ should be of the same length."
            )
    speech2feat.dim_search_done = True
    if args.batch_size > 1 and len(speech2feat.skip_pad) > 0:
        logging.info(f"Skip unpadding {speech2feat.skip_pad} as it is unsuccessful")

    if len(feats_dict.items()) == 0:
        logging.error(
            f"No module outputs found to save."
            + " Make sure the module name specified by --output_modules exists."
        )

    # Open a writer for each feature type
    os.makedirs(args.output_dir, exist_ok=True)
    writers = {}
    for feat_type, feats in feats_dict.items():
        writers[feat_type] = file_writer_helper(
            "ark,scp:"
            + os.path.join(args.output_dir, f"{feat_type}.ark")
            + ","
            + os.path.join(args.output_dir, f"{feat_type}.scp"),
            filetype=args.filetype,
            compress=args.compress,
            compression_method=args.compression_method,
        )

    for utt_ids, feats_dict in speech2feat:
        for utt_idx, utt_id in enumerate(utt_ids):
            for feat_type, feats in feats_dict.items():
                logging.info(f"{feat_type}: " + str(feats[utt_idx].size()))
                writers[feat_type][utt_id] = feats[utt_idx].cpu().numpy()

    # Close all writers
    for writer in writers.values():
        writer.close()


def add_inference_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(description="Inference specific")
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and asr joint model",
    )
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

    return parser


def add_data_loader_arguments(parser: argparse.ArgumentParser):
    required = parser.get_default("required")
    required += ["data_path_and_name_and_type"]

    group = parser.add_argument_group(description="Data loader specific")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    return parser


def add_precompute_feature_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(description="Precompute feature specific")
    parser.add_argument(
        "--output_modules",
        type=str,
        nargs="*",
        default=["encoder.encoders.*"],
        help="""List of module outputs to precompute.
        E.g.: --output_modules=[encoder.encoders.*].
        encoder and encoders are module names, and * means any module.
        Remarks:
        1. Use frontend.upstream instead of frontend.upstream.upstream.model.encoder.layers.* for WavLM""",
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

    return parser


def add_writer_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(description="Writer specific")
    parser.add_argument(
        "--filetype",
        type=str,
        default="hdf5",
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


def get_parser(base_parser: argparse.ArgumentParser = None):
    class ArgumentDefaultsRawTextHelpFormatter(
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
    ):
        pass

    parser = config_argparse.ArgumentParser(
        description="Precompute ASR model features from dataset",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        parents=[base_parser] if base_parser is not None else [],
        conflict_handler="resolve",
    )

    parser.set_defaults(required=["output_dir"])

    group = parser.add_argument_group("common options")
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser = add_inference_arguments(parser)
    parser = add_data_loader_arguments(parser)
    parser = add_precompute_feature_arguments(parser)
    parser = add_writer_arguments(parser)

    return parser


def main(cmd=None):
    parser = ASRTask.get_parser()  # Parse ASR task arguments
    parser = get_parser(parser)
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
