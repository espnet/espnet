#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.quantization
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.s2t import S2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args


class Speech2Language:
    @typechecked
    def __init__(
        self,
        s2t_train_config: Union[Path, str, None] = None,
        s2t_model_file: Union[Path, str, None] = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
        nbest: int = 1,
        quantize_s2t_model: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        first_lang_sym: str = "<abk>",
        last_lang_sym: str = "<zul>",
        use_flash_attn: bool = False,
    ):

        qconfig_spec = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype: torch.dtype = getattr(torch, quantize_dtype)

        s2t_model, s2t_train_args = S2TTask.build_model_from_file(
            s2t_train_config, s2t_model_file, device
        )
        s2t_model.to(dtype=getattr(torch, dtype)).eval()

        # Set flash_attn
        for m in s2t_model.modules():
            if hasattr(m, "use_flash_attn"):
                setattr(m, "use_flash_attn", use_flash_attn)

        if quantize_s2t_model:
            logging.info("Use quantized s2t model for decoding.")

            s2t_model = torch.quantization.quantize_dynamic(
                s2t_model, qconfig_spec=qconfig_spec, dtype=quantize_dtype
            )

        logging.info(f"Decoding device={device}, dtype={dtype}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.preprocessor_conf = s2t_train_args.preprocessor_conf
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        token_list = s2t_model.token_list
        self.first_lang_id = token_list.index(first_lang_sym)
        self.last_lang_id = token_list.index(last_lang_sym)

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
    ) -> List[Tuple[str, float]]:
        """Predict the language in input speech.

        The input speech will be padded or trimmed to the fixed length,
        which is consistent with training.

        Args:
            speech: input speech of shape (nsamples,) or (nsamples, nchannels=1)

        Returns:
            List of (language, probability)

        """

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Only support single-channel speech
        if speech.dim() > 1:
            assert (
                speech.dim() == 2 and speech.size(1) == 1
            ), f"speech of size {speech.size()} is not supported"
            speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

        speech_length = int(
            self.preprocessor_conf["fs"] * self.preprocessor_conf["speech_length"]
        )
        # Pad or trim speech to the fixed length
        if speech.size(-1) >= speech_length:
            speech = speech[:speech_length]
        else:
            speech = F.pad(speech, (0, speech_length - speech.size(-1)))

        # Batchify input
        # speech: (nsamples,) -> (1, nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_olens = self.s2t_model.encode(**batch)
        assert len(enc) == 1, len(enc)

        # c. Forward Decoder by one step
        ys = torch.tensor(
            [self.s2t_model.sos] * len(enc), dtype=torch.long, device=self.device
        ).unsqueeze(-1)
        logp, _ = self.s2t_model.decoder.batch_score(ys, [None], enc)
        assert len(logp) == 1, len(logp)

        prob = torch.softmax(logp[0, self.first_lang_id : self.last_lang_id + 1], -1)
        best_results = torch.topk(prob, self.nbest)
        results = []
        for idx, val in zip(best_results.indices, best_results.values):
            results.append(
                (self.s2t_model.token_list[idx + self.first_lang_id], val.item())
            )

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Language instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Language: Speech2Language instance.

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

        return Speech2Language(**kwargs)


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    s2t_train_config: Optional[str],
    s2t_model_file: Optional[str],
    model_tag: Optional[str],
    allow_variable_data_keys: bool,
    quantize_s2t_model: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    first_lang_sym: str,
    last_lang_sym: str,
):
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

    # 2. Build speech2text
    speech2language_kwargs = dict(
        s2t_train_config=s2t_train_config,
        s2t_model_file=s2t_model_file,
        device=device,
        dtype=dtype,
        nbest=nbest,
        quantize_s2t_model=quantize_s2t_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        first_lang_sym=first_lang_sym,
        last_lang_sym=last_lang_sym,
    )
    speech2language = Speech2Language.from_pretrained(
        model_tag=model_tag,
        **speech2language_kwargs,
    )

    # 3. Build data-iterator
    loader = S2TTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=S2TTask.build_preprocess_fn(
            speech2language.s2t_train_args, False
        ),
        collate_fn=S2TTask.build_collate_fn(speech2language.s2t_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            logging.info(keys[0])
            # N-best list of (lang, prob)
            try:
                results = speech2language(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                results = [(" ", 0.0)] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            for n, (lang, prob) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["score"][key] = str(prob)
                ibest_writer["text"][key] = lang


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="S2T Decoding",
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
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Model configuration related")
    group.add_argument(
        "--s2t_train_config",
        type=str,
        help="S2T training configuration",
    )
    group.add_argument(
        "--s2t_model_file",
        type=str,
        help="S2T model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    group.add_argument(
        "--first_lang_sym",
        type=str,
        default="<abk>",
        help="The first language symbol.",
    )
    group.add_argument(
        "--last_lang_sym", type=str, default="<zul>", help="The last language symbol."
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_s2t_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to S2T model.",
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

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")

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
