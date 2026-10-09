#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn.functional as F
import torch.quantization
import yaml
from typeguard import typechecked

from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.partially_AR_model import PartiallyARInference
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet2.legacy.nets.beam_search import BeamSearch, Hypothesis
from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.legacy.nets.scorer_interface import (
    BatchPartialScorerInterface,
    BatchScorerInterface,
)
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.lm import LMTask
from espnet2.tasks.s2t import S2TTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.pretrained import download_pretrained
from espnet2.utils.types import str2bool, str2triple_str, str_or_none

# Alias for typing
ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Optional[str],
        # None when the text came from the CTC head: nothing was searched,
        # so there is no hypothesis to report
        Optional[Hypothesis],
    ]
]


class ScoreFilter(BatchScorerInterface, torch.nn.Module):
    """Filter scores based on pre-defined rules.

    See comments in the score method.

    """

    def __init__(
        self,
        notimestamps: int,
        first_time: int,
        last_time: int,
        sos: int,
        eos: int,
        vocab_size: int,
    ):
        super().__init__()

        self.notimestamps = notimestamps
        self.first_time = first_time
        self.last_time = last_time
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size

        # dummy param used to obtain the current dtype and device
        self.param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """

        score = torch.zeros(
            self.vocab_size, dtype=self.param.dtype, device=self.param.device
        )
        if self.notimestamps in y:
            # Suppress timestamp tokens if we don't predict time
            score[self.first_time : self.last_time + 1] = -np.inf
        elif y[-3] == self.sos:
            # The first token must be a timestamp if we predict time
            score[: self.first_time] = -np.inf
            score[self.last_time + 1 :] = -np.inf
        else:
            prev_times = y[torch.logical_and(y >= self.first_time, y <= self.last_time)]
            if len(prev_times) % 2 == 1:
                # there are an odd number of timestamps, so the sentence is incomplete
                score[self.eos] = -np.inf
                # timestamps are monotonic
                score[self.first_time : prev_times[-1] + 1] = -np.inf
            else:
                # there are an even number of timestamps (all are paired)
                if y[-1] >= self.first_time and y[-1] <= self.last_time:
                    # the next tokon should be a timestamp or eos
                    score[: y[-1]] = -np.inf
                    score[self.last_time + 1 :] = -np.inf
                    score[self.eos] = 0.0
                else:
                    # this is an illegal hyp
                    score[:] = -np.inf

        return score, None

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """

        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


# espnet2.tasks.s2t and espnet2.tasks.s2t_ctc both write `model:` into the
# training config, and only the CTC-only task writes this value. Reading one
# key tells the two apart without building a model to see which one fails.
CTC_ONLY_MODEL = "espnet_ctc"


SUBSAMPLE = {"conv2d1": 1, "conv2d2": 2, "conv2d": 4, "conv2d6": 6, "conv2d8": 8}


def _frame_rate(s2t_train_args) -> Tuple[Optional[int], Optional[float]]:
    """The audio sample rate and the encoder's output frames per second.

    Only the long-form paths need these, and only a config that names its
    frontend's rate, hop and input layer can give them, so a config that does
    not gets (None, None) rather than a KeyError at construction.
    """
    frontend = getattr(s2t_train_args, "frontend_conf", None) or {}
    encoder = getattr(s2t_train_args, "encoder_conf", None) or {}
    sample_rate = frontend.get("fs")
    hop_length = frontend.get("hop_length")
    subsample = SUBSAMPLE.get(encoder.get("input_layer"))
    if sample_rate is None or hop_length is None or subsample is None:
        return None, None
    if isinstance(sample_rate, str):
        sample_rate = humanfriendly.parse_size(sample_rate)
    return sample_rate, sample_rate / hop_length / subsample


def _stated_seconds(symbol: str) -> Optional[float]:
    """The time a symbol is named after, if it is named after one.

    `<12.34>` is 12.34; `<sos>` and anything else is None. Used to check a
    config against itself, never to decide what a timestamp means.
    """
    try:
        return float(str(symbol).strip("<>"))
    except ValueError:
        return None


def _is_ctc_only(s2t_train_config) -> bool:
    """True when the config describes a model trained as CTC-only.

    Such a model - OWSM-CTC is the one in the wild - has an encoder and a CTC
    head and no decoder, so there is nothing for a beam search to search over.
    """
    if s2t_train_config is None:
        # only a checkpoint was given; the encoder-decoder task is the older
        # and more common one, and build_model_from_file will say if it is wrong
        return False
    with Path(s2t_train_config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return isinstance(config, dict) and config.get("model") == CTC_ONLY_MODEL


class Speech2Text:
    """Decode a speech-to-text model, of either kind, however you ask.

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("s2t_config.yml", "s2t.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, text_nospecial, hypothesis object), ...]

    Two methods, and the arguments decide the rest::

        Speech2Text
        |
        +-- __call__()          a search, over the scorers this model has
        |   |
        |   +-- decoder         weight 1 - ctc_weight
        |   +-- CTC             weight ctc_weight
        |   +-- LM              weight lm_weight
        |   +-- n-gram          weight ngram_weight
        |
        +-- best_path()         no search: CTC argmax, repeats collapsed

    Which gives, on an encoder-decoder checkpoint:

    =========================  ================================================
    attention beam search      ``ctc_weight=0``
    hybrid beam search         ``0 < ctc_weight < 1``
    CTC beam search            ``ctc_weight=1``
    with a language model      any of the above, plus ``lm_weight>0`` and an
                               ``lm_train_config``
    best path                  ``best_path()``, which reads the CTC branch and
                               ignores every weight above
    =========================  ================================================

    and on a CTC-only checkpoint - OWSM-CTC, anything the `s2t_ctc` task
    trained - the decoder simply is not there, so ``__call__`` is a CTC beam
    search, with a language model if one is given, and ``best_path`` is the
    same fast route. `decode_long` decodes a whole recording either way.

    The branching is in three places: which model to build (`_is_ctc_only`,
    at the top of `__init__`), which scorers the search gets (the
    ``self.ctc_only`` branch a few lines further down), and how the encoder
    is fed (`_encode`, since a CTC-only model takes the prompt as an encoder
    input rather than as a decoder prefix).

    """

    @typechecked
    def __init__(
        self,
        s2t_train_config: Union[Path, str, None] = None,
        s2t_model_file: Union[Path, str, None] = None,
        lm_train_config: Union[Path, str, None] = None,
        lm_file: Union[Path, str, None] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str, None] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 5,
        ctc_weight: float = 0.0,
        lm_weight: float = 0.0,
        ngram_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        quantize_s2t_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        partial_ar: bool = False,
        threshold_probability: float = 0.99,
        max_seq_len: int = 5,
        max_mask_parallel: int = -1,
        use_flash_attn: bool = False,
        # default values that can be overwritten in __call__
        lang_sym: str = "<eng>",
        task_sym: str = "<asr>",
        predict_time: bool = False,
        # simulated streaming, from the CTC-only inference this replaces
        streaming: bool = False,
        # only best-path decoding reports these, and only when asked: the
        # beam search path reports them whenever the model produces them
        generate_interctc_outputs: bool = False,
    ):

        if ctc_weight > 0.0 and predict_time:
            raise ValueError("CTC cannot predict timestamps")

        qconfig_spec = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype: torch.dtype = getattr(torch, quantize_dtype)

        # 1. Build S2T model, from whichever task trained it
        self.ctc_only = _is_ctc_only(s2t_train_config)
        if self.ctc_only:
            from espnet2.tasks.s2t_ctc import S2TTask as S2TCTCTask

            s2t_model, s2t_train_args = S2TCTCTask.build_model_from_file(
                s2t_train_config, s2t_model_file, device
            )
        else:
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

        token_list = s2t_model.token_list
        # Which scorers the search is given is the whole difference between
        # the ways an S2T model can be decoded:
        #
        #   attention beam search  decoder                ctc_weight = 0
        #   hybrid beam search     decoder + CTC          0 < ctc_weight < 1
        #   CTC beam search        CTC                    ctc_weight = 1, or any
        #                                                 CTC-only checkpoint
        #   best path              CTC, no search         best_path(), which
        #                                                 ignores all of this
        #
        # The last two are both "CTC decoding" and are not the same: the
        # third searches over label sequences and the fourth takes the
        # argmax of each frame. ctc_weight = 1 selects the third, whatever
        # beam_size is, which is what the log line below is for.
        #
        # A language model or an n-gram joins any of the searches, through
        # lm_weight and ngram_weight below. None of them reaches best_path():
        # an argmax over frames has nothing to score.
        ctc = CTCPrefixScorer(ctc=s2t_model.ctc, eos=s2t_model.eos)
        if self.ctc_only:
            if partial_ar:
                raise ValueError(
                    "partial_ar needs a decoder, and this checkpoint is CTC-only"
                )
            # no decoder to score with, and no timestamps for ScoreFilter to
            # constrain: a CTC-only model emits neither
            scorers: Dict[str, Any] = dict(
                decoder=None,
                ctc=ctc,
                length_bonus=LengthBonus(len(token_list)),
            )
            weights = dict(
                decoder=0.0,
                ctc=1.0,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )
            pre_beam_score_key = None
        else:
            scorers = dict(
                decoder=s2t_model.decoder,
                ctc=ctc,
                length_bonus=LengthBonus(len(token_list)),
                scorefilter=ScoreFilter(
                    notimestamps=token_list.index(
                        s2t_train_args.preprocessor_conf["notime_symbol"]
                    ),
                    first_time=token_list.index(
                        s2t_train_args.preprocessor_conf["first_time_symbol"]
                    ),
                    last_time=token_list.index(
                        s2t_train_args.preprocessor_conf["last_time_symbol"]
                    ),
                    sos=s2t_model.sos,
                    eos=s2t_model.eos,
                    vocab_size=len(token_list),
                ),
            )
            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
                scorefilter=1.0,
            )
            # with the decoder dropped there is no full scorer to pre-beam with
            pre_beam_score_key = None if ctc_weight == 1.0 else "full"

        if weights["ctc"] == 1.0:
            # "CTC decoding" names both of the things below, and "greedy" is
            # used loosely for the second, so a user who asked for one and
            # got the other has no way to tell except by the clock.
            #
            # Here rather than in the decoding methods: this is a property of
            # how the object was built, and a loop over a test set would
            # otherwise repeat it once per utterance.
            logging.info(
                "decoding on the CTC head alone: a prefix beam search over "
                f"{beam_size} hypotheses. This is not best-path decoding "
                "(also called greedy or argmax decoding), and beam_size=1 "
                "would not make it so - a prefix search scores label "
                "sequences, summing the frame paths that collapse to each "
                "one, where best-path takes the most likely symbol at each "
                "frame and collapses once. They can disagree, and the search "
                "is far slower. Speech2Text.best_path() is best-path decoding."
            )

        # 2. Build language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )

            if quantize_lm:
                logging.info("Use quantized lm for decoding.")

                lm = torch.quantization.quantize_dynamic(
                    lm, qconfig_spec=qconfig_spec, dtype=quantize_dtype
                )

            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet2.legacy.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet2.legacy.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
            scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        if partial_ar:
            beam_search = PartiallyARInference(
                s2t_model.ctc,
                s2t_model.decoder,
                threshold_probability=threshold_probability,
                sos=s2t_model.sos,
                eos=s2t_model.eos,
                mask_token=len(token_list),
                token_list=token_list,
                scorers={"decoder": s2t_model.decoder},
                weights=weights,
                beam_size=beam_size,
                max_seq_len=max_seq_len,
                max_mask_parallel=max_mask_parallel,
            )
        else:
            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=s2t_model.sos,
                eos=s2t_model.eos,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=pre_beam_score_key,
                normalize_length=normalize_length,
            )

            # TODO(karita): make all scorers batchfied
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            # NOTE: partial scorers too. `NgramPartScorer` is a plain
            # `PartialScorerInterface`, so batch decoding would reach
            # `batch_score_partial` and fail deep inside the search.
            non_batch += (
                [
                    k
                    for k, v in beam_search.part_scorers.items()
                    if not isinstance(v, BatchPartialScorerInterface)
                ]
                if batch_size > 1
                else []
            )
            if len(non_batch) > 0:
                if batch_size > 1:
                    raise NotImplementedError(
                        f"Batch decoding needs batch scorers, but {non_batch} "
                        f"are not. Please use --batch_size 1."
                    )
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
            elif streaming:
                beam_search.__class__ = BatchBeamSearchOnlineSim
                beam_search.set_streaming_config(s2t_train_config)
                logging.info("BatchBeamSearchOnlineSim implementation is selected.")
            else:
                beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected.")

        # NOTE: this used to sit inside the `else` above, so the partial_ar
        # beam search never got the requested device and dtype.
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = s2t_train_args.token_type
        if bpemodel is None:
            bpemodel = s2t_train_args.bpemodel

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
            if beam_search is not None:
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.sot_sequence_including_notimestamps)
                )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.preprocessor_conf = s2t_train_args.preprocessor_conf
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        self.lang_sym = lang_sym
        self.task_sym = task_sym
        self.predict_time = predict_time

        self.partial_ar = partial_ar
        self.batch_size = batch_size
        self.generate_interctc_outputs = generate_interctc_outputs

        if (
            batch_size > 1
            and not self.ctc_only
            and type(beam_search) is not BatchBeamSearch
        ):
            raise NotImplementedError(
                "Batch decoding is only supported for the attention/CTC beam "
                "search. Please use --batch_size 1."
            )

        # The sample rate and the encoder's frame rate, which the long-form
        # paths need to line chunks up with what the model was trained on.
        # Both are None when the config does not say - a configuration that
        # decodes utterance by utterance never needs them.
        self.sample_rate, self.frames_per_sec = _frame_rate(s2t_train_args)

    def _build_hyp_primer(
        self,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        predict_time: Optional[bool] = None,
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str, List]] = None,
    ) -> List[int]:
        """Build the prompt that every hypothesis of one utterance starts with."""
        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        predict_time = predict_time if predict_time is not None else self.predict_time

        lang_id = self.converter.token2id[lang_sym]
        task_id = self.converter.token2id[task_sym]
        notime_id = self.converter.token2id[self.preprocessor_conf["notime_symbol"]]

        hyp_primer = [self.s2t_model.sos, lang_id, task_id]
        if not predict_time:
            hyp_primer.append(notime_id)

        if text_prev is not None:
            if isinstance(text_prev, str):
                text_prev = self.converter.tokens2ids(
                    self.tokenizer.text2tokens(text_prev)
                )
            else:
                text_prev = text_prev.tolist()

            # Check if text_prev is valid
            if self.s2t_model.na in text_prev:
                text_prev = None

        if text_prev is not None:
            hyp_primer = [self.s2t_model.sop] + text_prev + hyp_primer

        return hyp_primer

    def _pad_or_trim(self, speech: torch.Tensor) -> torch.Tensor:
        """Pad or trim the last axis to the fixed length used in training."""
        speech_length = int(
            self.preprocessor_conf["fs"] * self.preprocessor_conf["speech_length"]
        )
        if speech.size(-1) > speech_length:
            # say so rather than dropping audio silently. In a batch this is
            # driven by the longest utterance, so every utterance is trimmed.
            logging.warning(
                f"trimming the input from {speech.size(-1)} to {speech_length} "
                f"samples, the fixed length this model was trained on. Use "
                f"decode_long() to transcribe audio longer than "
                f"{self.preprocessor_conf['speech_length']} s."
            )
        if speech.size(-1) >= speech_length:
            return speech[..., :speech_length]
        return F.pad(speech, (0, speech_length - speech.size(-1)))

    @torch.no_grad()
    @typechecked
    def batch_decode(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        text_prev: Optional[torch.Tensor] = None,
        text_prev_lengths: Optional[torch.Tensor] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        predict_time: Optional[bool] = None,
    ) -> List[ListOfHypothesis]:
        """Decode a minibatch of utterances in one beam search.

        Every utterance is padded or trimmed to the fixed length used in
        training, exactly as in :meth:`__call__`, so the encoder outputs of a
        batch all have the same length and no padding mask is needed.

        Args:
            speech: Padded speech of shape `(n_utt, nsamples)`.
            speech_lengths: Unused, and accepted only so that a collated
                batch can be passed straight through. Every utterance is
                padded or trimmed to the same fixed length, and the collated
                padding is zeros, which is exactly what `__call__` pads a
                short utterance with -- so ignoring the lengths gives the same
                input the single-utterance path would build.
            text_prev: Optional previous text of each utterance, used as a
                decoding condition. All the resulting prompts must have the
                same length, otherwise the hypotheses of the batch cannot be
                advanced in lock step.
            text_prev_lengths: Valid length of each row of `text_prev`.

        Returns:
            One n-best list of `(text, token, token_int, text_nospecial, hyp)`
            per utterance, in the order the utterances were given.

        """
        if speech.dim() == 3 and speech.size(2) == 1:
            speech = speech.squeeze(2)  # (n_utt, nsamples, 1) -> (n_utt, nsamples)
        if speech.dim() != 2:
            raise ValueError(f"speech of size {tuple(speech.shape)} is not supported")
        n_utt = speech.size(0)

        if self.ctc_only:
            # one encoder pass for the batch, then each utterance's own search
            speech = self._pad_or_trim(speech).to(getattr(torch, self.dtype))
            enc, _ = self._encode(speech, "<na>", lang_sym, task_sym)
            return [self._decode_single_sample(enc[b]) for b in range(n_utt)]

        if text_prev is None:
            primers = self._build_hyp_primer(lang_sym, task_sym, predict_time)
        else:
            if text_prev_lengths is None:
                text_prev_lengths = torch.full(
                    (n_utt,), text_prev.size(1), dtype=torch.long
                )
            primers = [
                self._build_hyp_primer(
                    lang_sym,
                    task_sym,
                    predict_time,
                    text_prev[b, : int(text_prev_lengths[b])],
                )
                for b in range(n_utt)
            ]
        self.beam_search.set_hyp_primer(primers)

        speech = self._pad_or_trim(speech).to(getattr(torch, self.dtype))
        lengths = speech.new_full([n_utt], dtype=torch.long, fill_value=speech.size(1))
        batch = to_device(
            {"speech": speech, "speech_lengths": lengths}, device=self.device
        )
        # NOTE: one line per utterance, in the exact wording that
        # pyscripts/utils/calculate_rtf.py parses as a decoding start time. It
        # asserts one such line per "best hypo" line, so this must not become a
        # single line for the whole batch.
        for _ in range(n_utt):
            logging.info("speech length: " + str(speech.size(1)))

        enc, enc_olens = self.s2t_model.encode(**batch)
        if isinstance(enc, tuple):
            # intermediate CTC outputs are not reported in batch decoding
            enc = enc[0]

        if hasattr(self.beam_search.nn_dict, "decoder"):
            if isinstance(self.beam_search.nn_dict.decoder, S4Decoder):
                # Setup: required for S4 autoregressive generation
                for module in self.beam_search.nn_dict.decoder.modules():
                    if hasattr(module, "setup_step"):
                        module.setup_step()

        nbest_hyps = self.beam_search(
            x=enc,
            x_lengths=enc_olens,
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
        )
        return [self._hyps_to_results(hyps) for hyps in nbest_hyps]

    def _ctc_batch(self, speech, text_prev, lang_sym, task_sym):
        """The batch a CTC-only model's encoder takes.

        OWSM-CTC conditions its encoder on the prompt, so the language and
        task symbols and the previous text are encoder inputs here, where an
        encoder-decoder model takes them as the decoder's prefix.
        """
        lang_id = self.converter.token2id[lang_sym or self.lang_sym]
        task_id = self.converter.token2id[task_sym or self.task_sym]

        if text_prev is None:
            # __call__ leaves it unset; the CTC models read "not available"
            text_prev = "<na>"
        if isinstance(text_prev, str):
            text_prev = self.converter.tokens2ids(self.tokenizer.text2tokens(text_prev))
        else:
            text_prev = list(text_prev)
        if self.s2t_model.na in text_prev:
            text_prev = [self.s2t_model.na]

        n_utt = speech.size(0)
        prev = torch.tensor(text_prev, dtype=torch.long).repeat(n_utt, 1)
        prefix = torch.tensor([[lang_id, task_id]], dtype=torch.long).repeat(n_utt, 1)
        return to_device(
            {
                "speech": speech,
                "speech_lengths": speech.new_full(
                    [n_utt], dtype=torch.long, fill_value=speech.size(1)
                ),
                "text_prev": prev,
                "text_prev_lengths": prev.new_full(
                    [n_utt], dtype=torch.long, fill_value=prev.size(1)
                ),
                "prefix": prefix,
                "prefix_lengths": prefix.new_full(
                    [n_utt], dtype=torch.long, fill_value=prefix.size(1)
                ),
            },
            device=self.device,
        )

    def _encode(self, speech, text_prev, lang_sym, task_sym):
        """Encoder output for a batch of utterances, whichever model this is."""
        if self.ctc_only:
            batch = self._ctc_batch(speech, text_prev, lang_sym, task_sym)
        else:
            # the encoder-decoder model was trained on a fixed window, and
            # __call__ pads or trims to it
            speech = self._pad_or_trim(speech)
            batch = to_device(
                {
                    "speech": speech,
                    "speech_lengths": speech.new_full(
                        [speech.size(0)], dtype=torch.long, fill_value=speech.size(1)
                    ),
                },
                device=self.device,
            )
        enc, _ = self.s2t_model.encode(**batch)
        intermediate_outs = None
        if isinstance(enc, tuple):
            enc, intermediate_outs = enc
        return enc, intermediate_outs

    def _collapse_ctc_path(self, enc: torch.Tensor) -> ListOfHypothesis:
        """One utterance's frame-wise argmax, with repeats and blanks removed.

        ctc.argmax is the CTC head's linear layer and an argmax over it: no
        softmax, which would cost a pass over the vocabulary without changing
        which symbol is largest.
        """
        token_int = self.s2t_model.ctc.argmax(enc.unsqueeze(0))[0]
        token_int = torch.unique_consecutive(token_int).cpu().tolist()
        token_int = [x for x in token_int if x != self.s2t_model.blank_id]
        token = self.converter.ids2tokens(token_int)
        # the language, task and timestamp symbols are the model's own, and
        # are not part of what was said
        token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]

        if self.tokenizer is not None:
            text = self.tokenizer.tokens2text(token)
            text_nospecial = self.tokenizer.tokens2text(token_nospecial)
        else:
            text, text_nospecial = None, None
        logging.info(f"best hypo: {text}")
        # no Hypothesis: nothing was searched, so there is no score to report
        return [(text, token, token_int, text_nospecial, None)]

    @torch.no_grad()
    def best_path(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Union[torch.Tensor, np.ndarray, str, List] = "<na>",
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> ListOfHypothesis:
        """Decode with the CTC head alone, no search.

        The most likely symbol per frame with the repeats collapsed: best-path
        decoding, also called greedy or argmax decoding. An order of magnitude
        faster than a search, and worse, which is the trade - a first look, a
        sanity check, a teaching example.

        Nothing here is scored, so `lm_weight`, `ngram_weight`, `ctc_weight`
        and `beam_size` do not apply: a language model can only join a search,
        which is `__call__`. On an encoder-decoder checkpoint this reads the
        CTC branch beside the decoder; on a CTC-only one it reads the only
        head there is.

        **A CTC branch answers what that branch was trained on, which is not
        always what `task_sym` asks for.** POWSM's answers phones whether it
        is asked for `<pr>` or `<asr>`: on espnet/powsm, best_path returns
        the same phones for both, and only the decoder - `__call__` - reads
        the task. A caller that must honour the task on an encoder-decoder
        checkpoint should call the object instead, and pay for the search.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if speech.dim() > 1:
            raise ValueError(
                f"speech of size {tuple(speech.shape)} is not one utterance; "
                "use batch_decode for a batch"
            )
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        enc, intermediate_outs = self._encode(speech, text_prev, lang_sym, task_sym)
        results = self._collapse_ctc_path(enc[0])
        if intermediate_outs is not None and self.generate_interctc_outputs:
            return results, self._decode_interctc(intermediate_outs)
        return results

    def decode_window(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        text_prev: Union[torch.Tensor, np.ndarray, str, List] = "<na>",
    ) -> str:
        """One window of audio, decoded the way this checkpoint has to be.

        A CTC-only checkpoint is read off its CTC head with no search, which
        is an order of magnitude faster and loses nothing: that head is the
        whole model. One with a decoder is called, because its CTC branch
        answers what that branch was trained on rather than what `task_sym`
        asks for - see `best_path`.

        Here rather than in each front end: `espnet phonemize` and the
        browser demo both had to know which kind of checkpoint they were
        holding, and it is the checkpoint that knows.

        Args:
            speech: One window, no longer than the model's own.
            lang_sym, task_sym: As for `best_path` and `__call__`.
            text_prev: What the model is given to condition on, for a task
                that has an input besides the audio. POWSM's `<g2p>` takes
                the words that were said and answers with phones, and its
                `<p2g>` takes phones and answers with words; `<asr>` and
                `<pr>` take `<na>`, which is the default.

        Returns:
            The decoded text, with whatever symbols the model wrote.
        """
        decode = self.best_path if self.ctc_only else self.__call__
        return decode(
            speech, text_prev=text_prev, lang_sym=lang_sym, task_sym=task_sym
        )[0][0]

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str, List]] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        predict_time: Optional[bool] = None,
    ) -> Union[
        ListOfHypothesis,
        Tuple[
            ListOfHypothesis,
            Optional[Dict[int, List[str]]],
        ],
    ]:
        """Decode a single utterance with a beam search.

        The input speech will be padded or trimmed to the fixed length,
        which is consistent with training.

        Which search is decided by the weights the object was built with,
        and by whether the checkpoint has a decoder at all: see the class
        docstring. A CTC-only checkpoint, or `ctc_weight=1.0`, means a CTC
        prefix beam search - a search over label sequences, which is not
        best-path decoding and is not made into it by `beam_size=1`.
        :meth:`best_path` is best-path decoding, and is much faster.

        Args:
            speech: input speech of shape (nsamples,) or (nsamples, nchannels=1)
            text_prev: previous text used as condition (optional)

        Returns:
            n-best list of (text, token, token_int, text_nospecial, hyp)

        """
        if self.ctc_only:
            # The prompt is an encoder input for a CTC-only model, not a
            # decoder prefix, so there is no hyp primer to set and nothing to
            # pad: the search still runs, over the CTC scorer alone.
            if isinstance(speech, np.ndarray):
                speech = torch.tensor(speech)
            enc, intermediate_outs = self._encode(
                speech.unsqueeze(0).to(getattr(torch, self.dtype)),
                text_prev,
                lang_sym,
                task_sym,
            )
            results = self._decode_single_sample(enc[0])
            # a CTC-only model always produces intermediate outputs, so unlike
            # the encoder-decoder path it reports them only when asked; that is
            # what the class this replaces did, and callers unpack accordingly
            if intermediate_outs is not None and self.generate_interctc_outputs:
                return results, self._decode_interctc(intermediate_outs)
            return results

        self.beam_search.set_hyp_primer(
            self._build_hyp_primer(lang_sym, task_sym, predict_time, text_prev)
        )

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Only support single-channel speech
        if speech.dim() > 1:
            assert (
                speech.dim() == 2 and speech.size(1) == 1
            ), f"speech of size {speech.size()} is not supported"
            speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

        # Pad or trim speech to the fixed length
        speech = self._pad_or_trim(speech)

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

        intermediate_outs = None
        if isinstance(enc, tuple):
            enc, intermediate_outs = enc

        assert len(enc) == 1, len(enc)

        # c. Pass the encoder result to the beam search
        results = self._decode_single_sample(enc[0])

        # Encoder intermediate CTC predictions
        if intermediate_outs is not None:
            encoder_interctc_res = self._decode_interctc(intermediate_outs)
            results = (results, encoder_interctc_res)

        return results

    def _decode_single_sample(self, enc: torch.Tensor):
        if hasattr(self.beam_search.nn_dict, "decoder"):
            if isinstance(self.beam_search.nn_dict.decoder, S4Decoder):
                # Setup: required for S4 autoregressive generation
                for module in self.beam_search.nn_dict.decoder.modules():
                    if hasattr(module, "setup_step"):
                        module.setup_step()

        nbest_hyps = self.beam_search(
            x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )
        return self._hyps_to_results(nbest_hyps)

    def _hyps_to_results(self, nbest_hyps: List[Hypothesis]):
        """Convert an n-best list of hypotheses into text/token tuples."""
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            start_pos = 1 if self.partial_ar else 0
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[start_pos:last_pos]
            else:
                token_int = hyp.yseq[start_pos:last_pos].tolist()

            if not self.partial_ar:
                token_int = token_int[token_int.index(self.s2t_model.sos) + 1 :]

            # remove blank symbol id
            token_int = list(filter(lambda x: x != self.s2t_model.blank_id, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            # remove special tokens (task, timestamp, etc.)
            token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]

            text, text_nospecial = None, None
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                text_nospecial = self.tokenizer.tokens2text(token_nospecial)

            results.append((text, token, token_int, text_nospecial, hyp))

        return results

    @typechecked
    def _decode_interctc(
        self, intermediate_outs: List[Tuple[int, torch.Tensor]]
    ) -> Dict[int, List[str]]:

        exclude_ids = [self.s2t_model.blank_id, self.s2t_model.sos, self.s2t_model.eos]
        res = {}
        token_list = self.beam_search.token_list

        for layer_idx, encoder_out in intermediate_outs:
            y = self.s2t_model.ctc.argmax(encoder_out)[0]  # batch_size = 1
            y = [x[0] for x in groupby(y) if x[0] not in exclude_ids]
            y = [token_list[x] for x in y]

            res[layer_idx] = y

        return res

    def read_audio(self, speech) -> np.ndarray:
        """One channel of float audio at the rate the model was trained on.

        Takes a path, an array or a tensor. Accepting a path is what lets
        `decode_long` be the whole transcription API for a recording, rather
        than something every caller wraps in file reading and resampling -
        and it is public for the same reason: a caller that cuts a recording
        up itself should not have to repeat the resampling, nor guess the
        rate the checkpoint wants.
        """
        if isinstance(speech, (str, Path)):
            import soundfile as sf

            speech, rate = sf.read(str(speech), dtype="float32", always_2d=False)
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            if self.sample_rate is not None and rate != self.sample_rate:
                import librosa

                speech = librosa.resample(
                    speech, orig_sr=rate, target_sr=self.sample_rate
                )
            return speech
        if isinstance(speech, torch.Tensor):
            speech = speech.cpu().numpy()
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim == 2 and speech.shape[1] == 1:
            speech = speech[:, 0]
        if speech.ndim != 1:
            raise ValueError(f"speech of size {speech.shape} is not one recording")
        return speech

    @torch.no_grad()
    def ctc_log_probs(
        self,
        speech: Union[str, Path, torch.Tensor, np.ndarray],
        batch_size: int = 1,
        context_len_in_secs: float = 2,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> np.ndarray:
        """The CTC head's log posteriors for a recording, as (frames, vocab).

        The model sees one training-length buffer at a time, with context on
        either side that is encoded and then dropped, so the frames kept from
        each buffer were never at its edge.

        Long-form best-path decoding is an argmax over what this returns, and
        a forced alignment (espnet2.bin.align) is a Viterbi path through it:
        one buffering, read two ways.
        """
        speech = self.read_audio(speech)
        lang_id = self.converter.token2id[lang_sym or self.lang_sym]
        task_id = self.converter.token2id[task_sym or self.task_sym]

        buffer_len_in_secs = self.preprocessor_conf["speech_length"]
        chunk_len_in_secs = buffer_len_in_secs - 2 * context_len_in_secs
        buffer_len = int(self.sample_rate * buffer_len_in_secs)
        chunk_len = int(self.sample_rate * chunk_len_in_secs)
        context = int(self.sample_rate * context_len_in_secs)

        padded = np.pad(speech, (context, context))
        buffers = []
        for start in range(0, len(padded), chunk_len):
            buffer = padded[start : start + buffer_len]
            if len(buffer) < buffer_len:
                buffers.append(np.pad(buffer, (0, buffer_len - len(buffer))))
                break
            buffers.append(buffer)

        batched = torch.tensor(np.array(buffers)).to(getattr(torch, self.dtype))
        buffer_frames = int(self.frames_per_sec * buffer_len_in_secs)
        context_frames = int(self.frames_per_sec * context_len_in_secs)

        kept = []
        for idx in range(0, batched.size(0), batch_size):
            window = batched[idx : idx + batch_size]
            n = window.size(0)
            prev = torch.tensor([self.s2t_model.na], dtype=torch.long).repeat(n, 1)
            prefix = torch.tensor([lang_id, task_id], dtype=torch.long).repeat(n, 1)
            batch = to_device(
                {
                    "speech": window,
                    "speech_lengths": window.new_full(
                        [n], dtype=torch.long, fill_value=window.size(1)
                    ),
                    "text_prev": prev,
                    "text_prev_lengths": prev.new_full(
                        [n], dtype=torch.long, fill_value=prev.size(1)
                    ),
                    "prefix": prefix,
                    "prefix_lengths": prefix.new_full(
                        [n], dtype=torch.long, fill_value=prefix.size(1)
                    ),
                },
                device=self.device,
            )
            enc, _ = self.s2t_model.encode(**batch)
            if isinstance(enc, tuple):
                enc = enc[0]
            # the convolutional front end can return more frames than the
            # buffer itself, so the tail goes before the context does
            enc = enc[:, :buffer_frames]
            frames = self.s2t_model.ctc.log_softmax(enc)
            kept.append(frames[:, context_frames:-context_frames])

        # (buffers, frames, vocab) back into one run of frames, cut to the
        # frames the recording itself covers rather than the padding
        probs = torch.cat([k.reshape(-1, k.size(-1)) for k in kept])
        wanted = int(round(len(speech) / self.sample_rate * self.frames_per_sec))
        return probs[:wanted].cpu().numpy()

    def _decode_long_ctc(
        self,
        speech: np.ndarray,
        batch_size: int = 1,
        context_len_in_secs: float = 2,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> str:
        """Best-path decoding of a long recording: an argmax over the frames."""
        probs = self.ctc_log_probs(
            speech,
            batch_size=batch_size,
            context_len_in_secs=context_len_in_secs,
            lang_sym=lang_sym,
            task_sym=task_sym,
        )
        frames = torch.tensor(probs).argmax(dim=-1)
        merged = torch.unique_consecutive(frames).tolist()
        token_int = [x for x in merged if x != self.s2t_model.blank_id]
        token = self.converter.ids2tokens(token_int)
        token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]
        return self.tokenizer.tokens2text(token_nospecial)

    @torch.no_grad()
    def decode_long(
        self,
        speech: Union[str, Path, torch.Tensor, np.ndarray],
        batch_size: int = 1,
        context_len_in_secs: float = 2,
        condition_on_prev_text: bool = False,
        init_text: Optional[str] = None,
        end_time_threshold: Optional[str] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        skip_last_chunk_threshold: float = 0.2,
    ) -> List[Tuple[float, float, str]]:
        """Decode one unsegmented recording of any length.

        Takes a path, an array or a tensor, and returns a list of
        `(start_time, end_time, text)`.

        How the recording is cut up depends on the checkpoint, because the two
        kinds of model were trained to be read differently. An encoder-decoder
        model emits timestamps, is decoded segment by segment and optionally
        conditioned on what it said before, and returns one entry per
        utterance. A CTC-only model has no timestamps, so it is decoded in
        overlapping buffers and returns a single entry covering the recording.

        Args:
            batch_size: buffers decoded together, on a CTC-only checkpoint.
            context_len_in_secs: context decoded and then dropped on either
                side of each buffer, on a CTC-only checkpoint.
            condition_on_prev_text, init_text, end_time_threshold,
                skip_last_chunk_threshold: the encoder-decoder path.
                `end_time_threshold` defaults to one second before the end of
                the model's own window, which is where it was hardcoded as
                OWSM's `<29.00>` until a 20 s model asked for a token that
                does not exist in its vocabulary.

        """
        speech = self.read_audio(speech)
        if self.sample_rate is None:
            raise RuntimeError(
                "this config does not say what sample rate and hop length the "
                "model was trained with, so a long recording cannot be lined "
                "up with what it expects"
            )
        if self.ctc_only:
            text = self._decode_long_ctc(
                speech,
                batch_size=batch_size,
                context_len_in_secs=context_len_in_secs,
                lang_sym=lang_sym,
                task_sym=task_sym,
            )
            return [(0.0, len(speech) / self.sample_rate, text)]
        return self._decode_long_attention(
            speech,
            condition_on_prev_text=condition_on_prev_text,
            init_text=init_text,
            end_time_threshold=end_time_threshold,
            lang_sym=lang_sym,
            task_sym=task_sym,
            skip_last_chunk_threshold=skip_last_chunk_threshold,
        )

    def _time_ids(self) -> Tuple[int, int, float]:
        """The first and last timestamp symbols, and the seconds between two.

        The window is the source of truth - `speech_length`, which every
        other part of this class already reads - and the timestamp symbols
        are one model's way of writing positions inside it. So the step is
        that window divided by the symbols that span it, and nothing here
        parses a symbol's name or assumes a format for it.

        Not `speech_resolution`: POWSM's config states 0.04 while its
        vocabulary steps 0.02, and a timestamp read at twice its value points
        past the end of the window it came from.

        What this does assume, and checks rather than trusts, is that the
        timestamps run one id a step from the start of the window to its end.
        A model that changes its window, its step, or both changes both
        numbers together and needs nothing here; a model that breaks the
        assumption is told which of its own two statements disagree, rather
        than quietly returning times that are a constant factor out.
        """
        first = self.converter.token2id[self.preprocessor_conf["first_time_symbol"]]
        last = self.converter.token2id[self.preprocessor_conf["last_time_symbol"]]
        if last <= first:
            raise RuntimeError(
                f"{self.preprocessor_conf['first_time_symbol']} and "
                f"{self.preprocessor_conf['last_time_symbol']} are not in order "
                f"in this vocabulary, so a position inside the window cannot "
                f"be read from an id"
            )
        window = float(self.preprocessor_conf["speech_length"])
        step = window / (last - first)

        # When the symbols are named after the times they mark - both models
        # in the wild are - the names have to agree with the window. This is
        # a check on the config, not the contract: a checkpoint whose
        # timestamps are named some other way skips it.
        named = [
            _stated_seconds(self.preprocessor_conf[key])
            for key in ("first_time_symbol", "last_time_symbol")
        ]
        if all(t is not None for t in named):
            spanned = named[1] - named[0]
            if abs(spanned - window) > step:
                raise RuntimeError(
                    f"this config's timestamps span {spanned:g} s "
                    f"({self.preprocessor_conf['first_time_symbol']} to "
                    f"{self.preprocessor_conf['last_time_symbol']}) while its "
                    f"speech_length says {window:g} s. One of the two is wrong, "
                    f"and either would put every timestamp in the wrong place"
                )
        return first, last, step

    def no_language(self) -> str:
        """The symbol this checkpoint uses for "work the language out yourself".

        Both POWSM checkpoints use `<unk>` and OWSM uses `<nolang>`, but
        only some configs say so: POWSM-CTC records `nolang_symbol`, POWSM
        does not, and POWSM has no `<nolang>` in its vocabulary at all, so a
        guess turns into a KeyError several seconds after the model has
        loaded. What the config says if it says anything, then either
        spelling, each checked against the token list before it is offered.

        Raises:
            ValueError: the checkpoint has no such symbol, and the caller has
                to name a language instead.
        """
        tokens = set(getattr(self.s2t_model, "token_list", None) or ())
        named = (self.preprocessor_conf or {}).get("nolang_symbol")
        for candidate in (named, "<nolang>", "<unk>"):
            if candidate and (not tokens or candidate in tokens):
                return str(candidate)
        raise ValueError("this model has no symbol for an unknown language: name one")

    def _near_window_end(self) -> int:
        """The timestamp a second before the end of the model's own window.

        An utterance whose end timestamp is past this one is taken to be cut
        off by the window rather than finished, so the next segment starts
        where it began. It used to be written out as OWSM's `<29.00>`, that
        model's 30 s window minus a second; POWSM's window is 20 s and
        `<29.00>` is not in its vocabulary at all, so the decode ended in a
        KeyError rather than in a transcript.
        """
        first, last, step = self._time_ids()
        # one second back, and never less than one step: a model whose steps
        # are coarser than a second would otherwise have no threshold at all
        return max(first, last - max(1, round(1.0 / step)))

    @torch.no_grad()
    @typechecked
    def _decode_long_attention(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        condition_on_prev_text: bool = False,
        init_text: Optional[str] = None,
        end_time_threshold: Optional[str] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        skip_last_chunk_threshold: float = 0.2,
    ):
        """Decode unsegmented long-form speech.

        Args:
            speech: 1D long-form input speech
            condition_on_prev_text (bool): whether to condition on previous text
            init_text: text used as condition for the first segment
            end_time_threshold: the last utterance is considered as incomplete
                if its end timestamp exceeds this threshold. None means one
                second before the end of the model's window.

        Returns:
            utterances: list of tuples of (start_time, end_time, text)

        """

        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        segment_len = int(
            self.preprocessor_conf["speech_length"] * self.preprocessor_conf["fs"]
        )
        first_time_id, last_time_id, resolution = self._time_ids()
        end_time_id_threshold = (
            self._near_window_end()
            if end_time_threshold is None
            else self.converter.token2id[end_time_threshold]
        )
        fs = self.preprocessor_conf["fs"]

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if speech.dim() > 1:
            assert (
                speech.dim() == 2 and speech.size(1) == 1
            ), f"speech of size {speech.size()} is not supported"
            speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

        utterances = []
        offset = 0
        text_prev = init_text
        while offset < len(speech):
            logging.info(f"Current start time in seconds: {offset / fs:.2f}")
            segment = speech[offset : offset + segment_len]
            if len(segment) / fs < skip_last_chunk_threshold:
                logging.warning(
                    f"Skip the last chunk as it's too short: {len(segment) / fs:.2f}s"
                )
                offset += segment_len
                continue

            # segment will be padded in __call__
            result = self.__call__(
                speech=segment,
                text_prev=text_prev if condition_on_prev_text else None,
                lang_sym=lang_sym,
                task_sym=task_sym,
                predict_time=True,
            )
            if isinstance(result, tuple):
                result = result[0]

            # NOTE(yifan): sos and eos have been removed
            text, token, token_int, text_nospecial, hyp = result[0]  # best hyp
            token_int = token_int[2:]  # remove lang and task

            # Find all timestamp positions
            time_pos = [
                idx
                for idx, tok in enumerate(token_int)
                if tok >= first_time_id and tok <= last_time_id
            ]
            # NOTE(yifan): this is an edge case with only a start time
            if len(time_pos) == 1:
                token_int.append(last_time_id)
                time_pos.append(len(token_int) - 1)

            if len(time_pos) % 2 == 0:  # Timestamps are all paired
                if (
                    len(time_pos) > 2
                    and token_int[time_pos[-1]] > end_time_id_threshold
                ):
                    # The last utterance is incomplete
                    new_start_time_id = token_int[time_pos[-2]]
                    time_pos = time_pos[:-2]
                else:
                    new_start_time_id = token_int[time_pos[-1]]
            else:  # The last utterance only has start time
                new_start_time_id = token_int[time_pos[-1]]
                time_pos = time_pos[:-1]

            # Get utterances in this segment
            text_prev = ""
            for i in range(0, len(time_pos), 2):
                utt = (
                    round(
                        (token_int[time_pos[i]] - first_time_id) * resolution
                        + offset / fs,
                        2,
                    ),
                    round(
                        (token_int[time_pos[i + 1]] - first_time_id) * resolution
                        + offset / fs,
                        2,
                    ),
                    self.tokenizer.tokens2text(
                        self.converter.ids2tokens(
                            token_int[time_pos[i] + 1 : time_pos[i + 1]]
                        )
                    ),
                )
                text_prev = text_prev + utt[-1]
                utterances.append(utt)

            offset += round((new_start_time_id - first_time_id) * resolution * fs)

        return utterances

    @classmethod
    def from_pretrained(
        cls,
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build an instance from a published model.

        A classmethod rather than a staticmethod so that a subclass gets an
        instance of itself: naming the class here returned the base class for
        anything that inherited this.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: an instance of the class this was called on.

        """
        if model_tag is not None:
            kwargs.update(download_pretrained(model_tag))

        return cls(**kwargs)


@typechecked
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
    s2t_train_config: Optional[str],
    s2t_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    quantize_s2t_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    lang_sym: str,
    task_sym: str,
    predict_time: bool,
    partial_ar: bool,
    threshold_probability: float,
    max_seq_len: int,
    max_mask_parallel: int,
):
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

    # NOTE(yifan): < and > cannot be passed in command line
    lang_sym = f"<{lang_sym.lstrip('<').rstrip('>')}>"
    task_sym = f"<{task_sym.lstrip('<').rstrip('>')}>"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        s2t_train_config=s2t_train_config,
        s2t_model_file=s2t_model_file,
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
        quantize_s2t_model=quantize_s2t_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        lang_sym=lang_sym,
        task_sym=task_sym,
        predict_time=predict_time,
        partial_ar=partial_ar,
        threshold_probability=threshold_probability,
        max_seq_len=max_seq_len,
        max_mask_parallel=max_mask_parallel,
        batch_size=batch_size,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = S2TTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=S2TTask.build_preprocess_fn(speech2text.s2t_train_args, False),
        collate_fn=S2TTask.build_collate_fn(speech2text.s2t_train_args, False),
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

            # One n-best list of
            # (text, token, token_int, text_nospecial, hyp_object) per key
            try:
                if batch_size > 1:
                    batch_results = speech2text.batch_decode(**batch)
                else:
                    batch_results = [
                        speech2text(
                            **{
                                k: v[0]
                                for k, v in batch.items()
                                if not k.endswith("_lengths")
                            }
                        )
                    ]
            except TooShortUttError as e:
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                placeholder = [[" ", ["<space>"], [2], " ", hyp]] * nbest
                if _bs == 1:
                    logging.warning(f"Utterance {keys} {e}")
                    batch_results = [placeholder]
                else:
                    # one too-short utterance would otherwise discard the
                    # results of the whole minibatch
                    logging.warning(
                        f"Utterances {keys} {e}; retrying them one at a time"
                    )
                    batch_results = []
                    for b, key in enumerate(keys):
                        one = {k: v[b : b + 1] for k, v in batch.items()}
                        try:
                            batch_results.append(speech2text.batch_decode(**one)[0])
                        except TooShortUttError as one_e:
                            logging.warning(f"Utterance {key} {one_e}")
                            batch_results.append(placeholder)

            for key, results in zip(keys, batch_results):
                encoder_interctc_res = None
                if isinstance(results, tuple):
                    results, encoder_interctc_res = results

                for n, (text, token, token_int, text_nospecial, hyp) in zip(
                    range(1, nbest + 1), results
                ):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text
                    if text_nospecial is not None:
                        ibest_writer["text_nospecial"][key] = text_nospecial

                # Write intermediate predictions to
                # encoder_interctc_layer<layer_idx>.txt
                ibest_writer = writer["1best_recog"]
                if encoder_interctc_res is not None:
                    for idx, text in encoder_interctc_res.items():
                        ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = (
                            " ".join(text)
                        )


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

    group.add_argument("--lang_sym", type=str, default="<eng>", help="Language symbol.")
    group.add_argument("--task_sym", type=str, default="<asr>", help="Task symbol.")
    group.add_argument(
        "--predict_time",
        type=str2bool,
        default=False,
        help="Predict timestamps.",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_s2t_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to S2T model.",
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
        "--batch_size",
        type=int,
        default=1,
        help="The number of utterances decoded in one beam search. "
        "Values > 1 need a model whose scorers are all batch scorers "
        "(attention decoder / CTC / neural LM).",
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
    group.add_argument("--lm_weight", type=float, default=0.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.0, help="ngram weight")
    group.add_argument(
        "--normalize_length",
        type=str2bool,
        default=False,
        help="If true, best hypothesis is selected by length-normalized scores",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", "word", None],
        help="The token type for S2T model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    group = parser.add_argument_group("Partially AR related")
    group.add_argument(
        "--partial_ar",
        type=str2bool,
        default=False,
        help="Flag to use the partially AR decoding",
    )
    group.add_argument(
        "--threshold_probability",
        type=float,
        default=0.99,
        help="Threshold for probability of the token to be masked",
    )
    group.add_argument(
        "--max_seq_len",
        type=int,
        default=5,
        help="Maximum sequence length for each hypothesis."
        + "Will stop beam_search after max_seq_len iteration in partially AR decoding.",
    )
    group.add_argument(
        "--max_mask_parallel",
        type=int,
        default=-1,
        help="Maximum number of masks to predict in parallel."
        + "If you got OOM error, try to decrease this value."
        + "Default to -1, which means always predict all masks simultaneously.",
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
