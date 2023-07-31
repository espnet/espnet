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
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.asr.align_refine_model import AlignRefineModel
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.bin.asr_inference import ListOfHypothesis
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

try:
    from transformers import AutoModelForSeq2SeqLM
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class Speech2Text:
    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        penalty: float = 0.0,
        nbest: int = 1,
        align_refine_k: int = 4,
        ctc_only: bool = False,
    ):
        assert check_argument_types()

        assert align_refine_k >= 1
        self.align_refine_k = align_refine_k
        self.ctc_only = ctc_only

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        assert isinstance(asr_model, AlignRefineModel), type(asr_model)
        asr_model.to(dtype=getattr(torch, dtype)).eval()
        token_list = asr_model.token_list

        # BeamSearch if using CTC-only decoding
        if self.ctc_only:
            ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
            scorers = dict(
                ctc=ctc,
                length_bonus=LengthBonus(len(token_list)),
            )
            weights = dict(
                ctc=1.0,
                length_bonus=penalty,
            )
            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=asr_model.sos,
                eos=asr_model.eos,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=None,
            )

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

            self.beam_search = beam_search

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

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

        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.ctc = asr_model.ctc
        self.beam_size = beam_size

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> ListOfHypothesis:
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
        logging.info("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_olens = self.asr_model.encode(**batch)

        # c. Decode
        if isinstance(enc, tuple):  # ignore InterCTC
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        if self.ctc_only:
            nbest_hyps = self._decode_ctc(enc[0])
        else:
            nbest_hyps = self._decode_align_refine(enc[0])

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.asr_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

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

    def _decode_ctc(self, enc: torch.Tensor) -> List[Hypothesis]:
        """
        Decode CTC using beam search or greedy search
        Args:
            enc: (T, D)
        """

        if self.beam_size == 1:  # greedy search
            _, tokens = self._ctc_greedy_search(enc.unsqueeze(0))

            tokens = [self.asr_model.blank_id] + tokens + [self.asr_model.blank_id]
            nbest_hyps = [Hypothesis(yseq=torch.as_tensor(tokens, dtype=torch.long))]
        else:  # beam search
            nbest_hyps = self.beam_search(
                x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )

        return nbest_hyps

    def _decode_align_refine(
        self,
        enc: torch.Tensor,
    ) -> List[Hypothesis]:
        """
        Iterative Align-refine
        Args:
            enc: (T, D)
        """

        enc = enc.unsqueeze(0)
        enc_lens = torch.as_tensor([enc.shape[1]], dtype=torch.long, device=enc.device)

        # CTC predictions
        y, tokens = self._ctc_greedy_search(enc)
        y = y.unsqueeze(0)
        text = "".join(self.converter.ids2tokens(tokens))
        logging.info(f"CTC greedy search: {text}")

        # Iterative align-refine
        for i in range(self.align_refine_k):
            decoder_out, y = self.asr_model.run_align_refine_once(
                enc,
                enc_lens,
                y,
                enc_lens,
            )

            # get the clean sequence
            tokens = [
                int(x[0]) for x in groupby(y[0]) if int(x[0]) != self.asr_model.blank_id
            ]

            text = "".join(self.converter.ids2tokens(tokens))
            logging.info(f"Align-refine output at iter {i}: {text}")

        seq = torch.tensor(
            [self.asr_model.blank_id] + tokens + [self.asr_model.blank_id],
            device=enc.device,
            dtype=torch.long,
        )
        return [Hypothesis(yseq=seq)]

    def _ctc_greedy_search(self, enc: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Batched CTC greedy search
        Args:
            enc: (B, T, D)
        """
        yseq = self.ctc.argmax(enc)[0]

        tokens = yseq.detach().cpu().numpy()
        tokens = [
            int(x[0]) for x in groupby(tokens) if int(x[0]) != self.asr_model.blank_id
        ]

        return yseq, tokens

    @classmethod
    def from_pretrained(
        cls,
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
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

        return cls(**kwargs)


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    align_refine_k: int,
    ctc_only: bool,
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

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        penalty=penalty,
        nbest=nbest,
        align_refine_k=align_refine_k,
        ctc_only=ctc_only,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
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

            encoder_interctc_res = None
            if isinstance(results, tuple):
                results, encoder_interctc_res = results

            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text

            # Write intermediate predictions to
            # encoder_interctc_layer<layer_idx>.txt
            ibest_writer = writer[f"1best_recog"]
            if encoder_interctc_res is not None:
                for idx, text in encoder_interctc_res.items():
                    ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = " ".join(
                        text
                    )


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
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

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    group = parser.add_argument_group("Align-refine related")
    group.add_argument(
        "--align-refine-k",
        type=int,
        default=4,
        help="Number of iterations of align-refine",
    )
    group.add_argument(
        "--ctc_only",
        type=str2bool,
        default=False,
        help="Skip align-refine and only run CTC decoding",
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
