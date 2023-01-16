#!/usr/bin/env python3
import argparse
import logging
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.uasr import UASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.scorers.uasr import UASRPrefixScorer
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface

# from espnet.nets.scorers.uasr import UASRPrefixScorer
from espnet.utils.cli_utils import get_commandline_args


class Speech2Text:
    """Speech2Text class for unsupervised ASR

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("uasr_config.yml", "uasr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis_object), ...]

    """

    def __init__(
        self,
        uasr_train_config: Union[Path, str] = None,
        uasr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        quantize_uasr_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
    ):
        assert check_argument_types()

        if quantize_uasr_model or quantize_lm:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion(
                "1.5.0"
            ):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build UASR model
        scorers = {}
        uasr_model, uasr_train_args = UASRTask.build_model_from_file(
            uasr_train_config, uasr_model_file, device
        )
        # TODO(Jiatong): change to not used pre-extracted features for inference
        uasr_model.use_collected_training_feats = True
        uasr_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_uasr_model:
            logging.info("Use quantized uasr model for decoding.")

            uasr_model = torch.quantization.quantize_dynamic(
                uasr_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        decoder = UASRPrefixScorer(eos=3)  # FIXME(jiatong): remove hard code
        token_list = uasr_model.token_list
        scorers.update(decoder=decoder)
        logging.info(f"beam search token list: {token_list}")

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )

            if quantize_lm:
                logging.info("Use quantized lm for decoding.")

                lm = torch.quantization.quantize_dynamic(
                    lm, qconfig_spec=quantize_modules, dtype=quantize_dtype
                )

            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(decoder=1.0, lm=lm_weight, ngram=ngram_weight,)

        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=1,  # FIXME(jiatong): remove hardcode
            eos=3,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None,  # Note(jiatong): for frame-decoding (current scope)
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

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = uasr_train_args.token_type
        if bpemodel is None:
            bpemodel = uasr_train_args.bpemodel

        # delete
        token_type = "word"

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.uasr_model = uasr_model
        self.uasr_train_args = uasr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], Union[Hypothesis],]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward encoder
        generated_sample, generated_sample_padding_mask = self.uasr_model.inference(
            **batch
        )
        assert len(generated_sample) == 1, len(generated_sample)

        # TODO(jiatong): add beamsearch
        nbest_hyps = self.beam_search(x=generated_sample[0], maxlenratio=1.0)

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:-1]
            else:
                token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x >= 4, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

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

        return Speech2Text(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    lm_weight: float,
    ngram_weight: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    uasr_train_config: Optional[str],
    uasr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    quantize_uasr_model: bool,
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

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        uasr_train_config=uasr_train_config,
        uasr_model_file=uasr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        nbest=nbest,
        quantize_uasr_model=quantize_uasr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag, **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = UASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=UASRTask.build_preprocess_fn(speech2text.uasr_train_args, False),
        collate_fn=UASRTask.build_collate_fn(speech2text.uasr_train_args, False),
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

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest

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
                    logging.info("key: {} text: {}".format(key, text))
                    logging.info("key: {} token_int: {}\n".format(key, token_int))


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="UASR Decoding",
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
        "--uasr_train_config", type=str, help="uasr training configuration",
    )
    group.add_argument(
        "--uasr_model_file", type=str, help="uasr model parameter file",
    )
    group.add_argument(
        "--lm_train_config", type=str, help="LM training configuration",
    )
    group.add_argument(
        "--lm_file", type=str, help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config", type=str, help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file", type=str, help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file", type=str, help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_uasr_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to uasr model.",
    )
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
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for uasr model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
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
