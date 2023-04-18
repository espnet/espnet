#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

# Alias for typing
ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Hypothesis,
    ]
]


class GenerateText:
    """GenerateText class

    Examples:
        >>> generatetext = GenerateText(
                lm_train_config="lm_config.yaml",
                lm_file="lm.pth",
                token_type="bpe",
                bpemodel="bpe.model",
            )
        >>> prompt = "I have travelled to many "
        >>> generatetext(prompt)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlen: int = 100,
        minlen: int = 0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ngram_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
    ):
        assert check_argument_types()

        # 1. Build language model
        lm, lm_train_args = LMTask.build_model_from_file(
            lm_train_config, lm_file, device
        )
        lm.to(dtype=getattr(torch, dtype)).eval()

        if quantize_lm:
            logging.info("Use quantized LM for decoding.")

            lm = torch.quantization.quantize_dynamic(
                lm,
                qconfig_spec=set([getattr(torch.nn, q) for q in quantize_modules]),
                dtype=getattr(torch, quantize_dtype),
            )

        token_list = lm_train_args.token_list

        # 2. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None

        # 3. Build BeamSearch object
        scorers = dict(
            lm=lm.lm,
            ngram=ngram,
            length_bonus=LengthBonus(len(token_list)),
        )
        weights = dict(
            lm=1.0,
            ngram=ngram_weight,
            length_bonus=penalty,
        )

        beam_search = BeamSearch(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=len(token_list),
            sos=lm.sos,
            eos=lm.eos,
            token_list=token_list,
            pre_beam_score_key="full",
        )

        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = lm_train_args.token_type
        if bpemodel is None:
            bpemodel = lm_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif (
            token_type == "bpe"
            or token_type == "hugging_face"
            or "whisper" in token_type
        ):
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        if bpemodel not in ["whisper_en", "whisper_multilingual"]:
            converter = TokenIDConverter(token_list=token_list)
        else:
            converter = OpenAIWhisperTokenIDConverter(model_type=bpemodel)
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.lm = lm
        self.lm_train_args = lm_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlen = maxlen
        self.minlen = minlen
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, text: Optional[Union[str, torch.Tensor, np.ndarray]] = None
    ) -> ListOfHypothesis:
        """Inference

        Args:
            text: Input text used as condition for generation
                If text is str, it will be converted to token ids
                and a <sos> token will be added at the beginning.
                If text is Tensor or ndarray, it will be used directly.
        Returns:
            List of (text, token, token_int, hyp)

        """
        assert check_argument_types()

        if isinstance(text, str):
            tokens = self.tokenizer.text2tokens(text)
            token_ids = self.converter.tokens2ids(tokens)
        elif text is None:
            token_ids = []
        else:
            token_ids = text.tolist()
        hyp_primer = [self.lm.sos] + token_ids
        self.beam_search.set_hyp_primer(hyp_primer)
        logging.info(f"hyp primer: {hyp_primer}")

        nbest_hyps = self.beam_search(
            x=torch.zeros(1, 1, device=self.device),  # only used to obtain device info
            maxlenratio=-self.maxlen,  # negative int means a constant max length
            minlenratio=-self.minlen,  # same for min length
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and convert to list
            token_int = hyp.yseq[1:-1]
            if not isinstance(token_int, list):
                token_int = token_int.tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            text = None
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build GenerateText instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            GenerateText: GenerateText instance.

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

        return GenerateText(**kwargs)


def inference(
    output_dir: str,
    maxlen: int,
    minlen: int,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
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

    # 1. Set random seed
    set_all_random_seed(seed)

    # 2. Build generatetext
    generatetext_kwargs = dict(
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlen=maxlen,
        minlen=minlen,
        dtype=dtype,
        beam_size=beam_size,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
    )
    generatetext = GenerateText.from_pretrained(
        model_tag=model_tag,
        **generatetext_kwargs,
    )

    # 3. Build data iterator
    loader = LMTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=LMTask.build_preprocess_fn(generatetext.lm_train_args, False),
        collate_fn=LMTask.build_collate_fn(generatetext.lm_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            results = generatetext(**batch)

            # Only supporting batch_size==1
            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="LM Decoding (conditional generation)",
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

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_lm",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to LM.",
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
        help="Batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlen",
        type=int,
        default=100,
        help="Maximum output length",
    )
    group.add_argument(
        "--minlen",
        type=int,
        default=1,
        help="Minimum output length",
    )
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "word", "bpe", None],
        help="Token type for LM. If not given, refers from the train args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="Model path for sentencepiece. If not given, refers from the train args",
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
