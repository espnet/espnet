#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    get_hugging_face_model_lm_head,
    get_hugging_face_model_network,
)
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.lm import LMTask
from espnet2.tasks.mt import MTTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class Text2Text:
    """Text2Text class

    Examples:
        >>> text2text = Text2Text("mt_config.yml", "mt.pth")
        >>> text2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        mt_train_config: Union[Path, str] = None,
        mt_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        hugging_face_decoder: bool = False,
        hugging_face_decoder_conf: Dict[str, Any] = {},
    ):
        assert check_argument_types()

        # 1. Build MT model
        scorers = {}
        mt_model, mt_train_args = MTTask.build_model_from_file(
            mt_train_config, mt_model_file, device
        )
        mt_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = mt_model.decoder
        ctc = (
            CTCPrefixScorer(ctc=mt_model.ctc, eos=mt_model.eos)
            if ctc_weight != 0.0
            else None
        )
        token_list = mt_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
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
        if (
            decoder.__class__.__name__ == "HuggingFaceTransformersDecoder"
            and hugging_face_decoder
        ):
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available."
                    " Please install it via `pip install transformers`"
                    " or `cd /path/to/espnet/tools && . ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )

            if decoder.causal_lm:
                hugging_face_model = AutoModelForCausalLM.from_pretrained(
                    decoder.model_name_or_path
                )

                hugging_face_model.resize_token_embeddings(decoder.lm_head.out_features)

                transformer = get_hugging_face_model_network(hugging_face_model)
                transformer.load_state_dict(decoder.decoder.state_dict())

                lm_head = get_hugging_face_model_lm_head(hugging_face_model)
                lm_head.load_state_dict(decoder.lm_head.state_dict())
            else:
                hugging_face_model = AutoModelForSeq2SeqLM.from_pretrained(
                    decoder.model_name_or_path
                )

                hugging_face_model.lm_head.load_state_dict(decoder.lm_head.state_dict())

                if hasattr(hugging_face_model, "model"):
                    hugging_face_model.model.decoder.load_state_dict(
                        decoder.decoder.state_dict()
                    )
                    del hugging_face_model.model.encoder
                else:
                    hugging_face_model.decoder.load_state_dict(
                        decoder.decoder.state_dict()
                    )
                    del hugging_face_model.encoder

            del mt_model.decoder.lm_head
            del mt_model.decoder.decoder

            hugging_face_linear_in = decoder.linear_in
            hugging_face_model.to(device=device).eval()

            if (
                hugging_face_model.config.pad_token_id is None
                and "pad_token_id" not in hugging_face_decoder_conf
            ):
                hugging_face_decoder_conf[
                    "pad_token_id"
                ] = hugging_face_model.config.eos_token_id

            beam_search = None
        else:
            hugging_face_model = None
            hugging_face_linear_in = None

            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )
            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=mt_model.sos,
                eos=mt_model.eos,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                normalize_length=normalize_length,
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
            token_type = mt_train_args.token_type
        if bpemodel is None:
            bpemodel = mt_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe" or token_type == "hugging_face":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        if token_type == "hugging_face":
            converter = HuggingFaceTokenIDConverter(model_name_or_path=bpemodel)
        else:
            converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.mt_model = mt_model
        self.mt_train_args = mt_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.hugging_face_model = hugging_face_model
        self.hugging_face_linear_in = hugging_face_linear_in
        self.hugging_face_decoder_conf = hugging_face_decoder_conf
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, src_text: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input text data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(src_text, np.ndarray):
            src_text = torch.tensor(src_text)

        # data: (Nsamples,) -> (1, Nsamples)
        src_text = src_text.unsqueeze(0).to(torch.long)
        # lengths: (1,)
        lengths = src_text.new_full([1], dtype=torch.long, fill_value=src_text.size(1))
        batch = {"src_text": src_text, "src_text_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.mt_model.encode(**batch)
        # self-condition case
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        if self.hugging_face_model:
            enc = self.hugging_face_linear_in(enc)

            if self.maxlenratio > 0:
                self.hugging_face_decoder_conf["max_new_tokens"] = int(
                    enc.shape[1] * self.maxlenratio
                )

            if self.mt_model.decoder.causal_lm:
                forward_args, _ = self.mt_model.decoder.add_prefix_postfix(
                    enc,
                    torch.tensor([enc.shape[1]]).to(enc.device),
                    torch.ones([1, 1], dtype=int, device=enc.device),
                    torch.ones([1], dtype=int, device=enc.device),
                )

                # input_ids are ignored if we provide inputs_embeds,
                # but input_ids are still required, so we make fake ones
                input_ids = torch.ones(
                    [1, forward_args["inputs_embeds"].shape[1]],
                    dtype=int,
                    device=enc.device,
                )

                yseq = self.hugging_face_model.generate(
                    input_ids,
                    inputs_embeds=forward_args["inputs_embeds"],
                    attention_mask=input_ids,
                    **self.hugging_face_decoder_conf,
                )

                yseq = yseq[:, input_ids.shape[1] - 1 :]
            else:
                decoder_start_token_id = (
                    self.hugging_face_model.config.decoder_start_token_id
                )
                yseq = self.hugging_face_model.generate(
                    encoder_outputs=ModelOutput(last_hidden_state=enc),
                    decoder_start_token_id=decoder_start_token_id,
                    **self.hugging_face_decoder_conf,
                )

            nbest_hyps = [Hypothesis(yseq=yseq[0])]
            logging.info(
                "best hypo: "
                + self.tokenizer.tokens2text(
                    self.converter.ids2tokens(nbest_hyps[0].yseq[1:])
                )
                + "\n"
            )
        else:
            nbest_hyps = self.beam_search(
                x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
            nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            # token_int = hyp.yseq[1:-1].tolist()
            # TODO(sdalmia): check why the above line doesn't work
            token_int = hyp.yseq.tolist()
            token_int = list(filter(lambda x: x != self.mt_model.sos, token_int))
            token_int = list(filter(lambda x: x != self.mt_model.eos, token_int))

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

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
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Text2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
        Returns:
            Text2Text: Text2Text instance.

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

        return Text2Text(**kwargs)


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    normalize_length: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    mt_train_config: Optional[str],
    mt_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    hugging_face_decoder: bool,
    hugging_face_decoder_conf: Dict[str, Any],
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

    # 2. Build text2text
    text2text_kwargs = dict(
        mt_train_config=mt_train_config,
        mt_model_file=mt_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        normalize_length=normalize_length,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_conf=hugging_face_decoder_conf,
    )
    text2text = Text2Text.from_pretrained(
        model_tag=model_tag,
        **text2text_kwargs,
    )

    # 3. Build data-iterator
    loader = MTTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=MTTask.build_preprocess_fn(text2text.mt_train_args, False),
        collate_fn=MTTask.build_collate_fn(text2text.mt_train_args, False),
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
                results = text2text(**batch)
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


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="MT Decoding",
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
        "--mt_train_config",
        type=str,
        help="ST training configuration",
    )
    group.add_argument(
        "--mt_model_file",
        type=str,
        help="MT model parameter file",
    )
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
    group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
    group.add_argument(
        "--hugging_face_decoder_conf",
        type=NestedDictAction,
        default=dict(),
        help="Custom kwargs for the HF .generate()",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths."
        "If maxlenratio<0.0, its absolute value is interpreted"
        "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.0,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ST model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--normalize_length",
        type=str2bool,
        default=False,
        help="If true, pruning is based on length-normalized scores",
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
