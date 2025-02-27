#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import humanfriendly
import librosa
import numpy as np
import torch
import torch.quantization
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.lm import LMTask
from espnet2.tasks.s2t_ctc import S2TTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

# Alias for typing
ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Optional[str],
        Optional[Hypothesis],
    ]
]


class Speech2Text:
    """Speech2Text class"""

    def __init__(
        self,
        s2t_train_config: Optional[Union[Path, str]] = None,
        s2t_model_file: Optional[Union[Path, str]] = None,
        lm_train_config: Optional[Union[Path, str]] = None,
        lm_file: Optional[Union[Path, str]] = None,
        ngram_scorer: str = "full",
        ngram_file: Optional[Union[Path, str]] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        lm_weight: float = 0.0,
        ngram_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        quantize_s2t_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        lang_sym: str = "<nolang>",
        task_sym: str = "<asr>",
        use_flash_attn: bool = False,
        generate_interctc_outputs: bool = False,
    ):

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build S2T model
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
                s2t_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        ctc = CTCPrefixScorer(ctc=s2t_model.ctc, eos=s2t_model.eos)
        token_list = s2t_model.token_list
        scorers = dict(
            decoder=None,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build language model
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
            scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=0.0,
            ctc=1.0,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=s2t_model.sos,
            eos=s2t_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None,
        )

        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(s2t_train_config)
                    logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                else:
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
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.generate_interctc_outputs = generate_interctc_outputs

        # default lang and task symbols
        self.lang_sym = lang_sym
        self.task_sym = task_sym

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Union[torch.Tensor, np.ndarray, str] = "<na>",
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> Union[
        ListOfHypothesis,
        Tuple[
            ListOfHypothesis,
            Optional[Dict[int, List[str]]],
        ],
    ]:
        """Inference for a short utterance.

        Args:
            speech: Input speech
            text_prev: Previous text used as condition
        Returns:
            text, token, token_int, hyp

        """

        # Obtain lang and task tokens
        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        lang_id = self.converter.token2id[lang_sym]
        task_id = self.converter.token2id[task_sym]

        if isinstance(text_prev, str):
            text_prev = self.converter.tokens2ids(self.tokenizer.text2tokens(text_prev))
        else:
            text_prev = text_prev.tolist()

        # Check if text_prev is valid
        if self.s2t_model.na in text_prev:
            text_prev = [self.s2t_model.na]

        text_prev = torch.tensor(text_prev, dtype=torch.long).unsqueeze(
            0
        )  # (1, length)
        text_prev_lengths = text_prev.new_full(
            [1], dtype=torch.long, fill_value=text_prev.size(1)
        )

        # Prepare prefix
        prefix = torch.tensor([[lang_id, task_id]], dtype=torch.long)  # (1, 2)
        prefix_lengths = prefix.new_full(
            [1], dtype=torch.long, fill_value=prefix.size(-1)
        )

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Batchify input
        # speech: (nsamples,) -> (1, nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        speech_lengths = speech.new_full(
            [1], dtype=torch.long, fill_value=speech.size(1)
        )
        logging.info("speech length: " + str(speech.size(1)))

        batch = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text_prev": text_prev,
            "text_prev_lengths": text_prev_lengths,
            "prefix": prefix,
            "prefix_lengths": prefix_lengths,
        }

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
        if intermediate_outs is not None and self.generate_interctc_outputs:
            encoder_interctc_res = self._decode_interctc(intermediate_outs)
            results = (results, encoder_interctc_res)

        return results

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

    def _decode_single_sample(self, enc: torch.Tensor):
        nbest_hyps = self.beam_search(
            x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[:last_pos]
            else:
                token_int = hyp.yseq[:last_pos].tolist()
            token_int = token_int[token_int.index(self.s2t_model.sos) + 1 :]

            # remove blank symbol id
            token_int = list(filter(lambda x: x != self.s2t_model.blank_id, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            # remove special tokens (task, timestamp, etc.)
            token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                text_nospecial = self.tokenizer.tokens2text(token_nospecial)
            else:
                text, text_nospecial = None, None
            results.append((text, token, token_int, text_nospecial, hyp))

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
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


class Speech2TextGreedySearch:
    """Speech2Text with greedy search for CTC."""

    def __init__(
        self,
        s2t_train_config: Optional[Union[Path, str]] = None,
        s2t_model_file: Optional[Union[Path, str]] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
        quantize_s2t_model: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        lang_sym: str = "<nolang>",
        task_sym: str = "<asr>",
        use_flash_attn: bool = False,
        generate_interctc_outputs: bool = False,
        **kwargs,
    ):

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build S2T model
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
                s2t_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        logging.info(f"Decoding device={device}, dtype={dtype}")

        # [Optional] Build Text converter: e.g. bpe-sym -> Text
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
            converter = TokenIDConverter(token_list=s2t_model.token_list)
        else:
            converter = OpenAIWhisperTokenIDConverter(model_type=bpemodel)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.preprocessor_conf = s2t_train_args.preprocessor_conf
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.generate_interctc_outputs = generate_interctc_outputs

        # default lang and task symbols
        self.lang_sym = lang_sym
        self.task_sym = task_sym

        # retrieve sample rate and compute frames per second
        sample_rate = s2t_train_args.frontend_conf["fs"]
        if isinstance(sample_rate, str):
            sample_rate = humanfriendly.parse_size(sample_rate)
        self.sample_rate = sample_rate
        logging.info(f"Audio sampling rate: {sample_rate}")

        subsample_dict = {
            "conv2d1": 1,
            "conv2d2": 2,
            "conv2d": 4,
            "conv2d6": 6,
            "conv2d8": 8,
        }
        subsample_factor = subsample_dict[s2t_train_args.encoder_conf["input_layer"]]
        frames_per_sec = sample_rate / s2t_train_args.frontend_conf["hop_length"]
        frames_per_sec /= subsample_factor
        self.frames_per_sec = frames_per_sec
        logging.info(f"Final encoder frames per second: {frames_per_sec}")

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Union[torch.Tensor, np.ndarray, str] = "<na>",
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> Union[
        ListOfHypothesis,
        Tuple[
            ListOfHypothesis,
            Optional[Dict[int, List[str]]],
        ],
    ]:
        """Inference for a short utterance.

        Args:
            speech: Input speech
            text_prev: Previous text used as condition
        Returns:
            text, token, token_int, hyp

        """

        # Obtain lang and task tokens
        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        lang_id = self.converter.token2id[lang_sym]
        task_id = self.converter.token2id[task_sym]

        if isinstance(text_prev, str):
            text_prev = self.converter.tokens2ids(self.tokenizer.text2tokens(text_prev))
        else:
            text_prev = text_prev.tolist()

        # Check if text_prev is valid
        if self.s2t_model.na in text_prev:
            text_prev = [self.s2t_model.na]

        text_prev = torch.tensor(text_prev, dtype=torch.long).unsqueeze(
            0
        )  # (1, length)
        text_prev_lengths = text_prev.new_full(
            [1], dtype=torch.long, fill_value=text_prev.size(1)
        )

        # Prepare prefix
        prefix = torch.tensor([[lang_id, task_id]], dtype=torch.long)  # (1, 2)
        prefix_lengths = prefix.new_full(
            [1], dtype=torch.long, fill_value=prefix.size(-1)
        )

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Batchify input
        # speech: (nsamples,) -> (1, nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        speech_lengths = speech.new_full(
            [1], dtype=torch.long, fill_value=speech.size(1)
        )
        logging.info("speech length: " + str(speech.size(1)))

        batch = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text_prev": text_prev,
            "text_prev_lengths": text_prev_lengths,
            "prefix": prefix,
            "prefix_lengths": prefix_lengths,
        }

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_olens = self.s2t_model.encode(**batch)

        intermediate_outs = None
        if isinstance(enc, tuple):
            enc, intermediate_outs = enc

        assert len(enc) == 1, len(enc)

        # c. Pass the encoder result to the beam search
        results = self._decode_single_sample(enc)

        # Encoder intermediate CTC predictions
        if intermediate_outs is not None and self.generate_interctc_outputs:
            encoder_interctc_res = self._decode_interctc(intermediate_outs)
            results = (results, encoder_interctc_res)

        return results

    def _decode_interctc(
        self, intermediate_outs: List[Tuple[int, torch.Tensor]]
    ) -> Dict[int, List[str]]:

        exclude_ids = [self.s2t_model.blank_id, self.s2t_model.sos, self.s2t_model.eos]
        token_list = self.s2t_model.token_list

        res = {}
        for layer_idx, encoder_out in intermediate_outs:
            y = self.s2t_model.ctc.argmax(encoder_out)[0]  # batch_size = 1
            y = [x[0] for x in groupby(y) if x[0] not in exclude_ids]
            y = [token_list[x] for x in y]

            res[layer_idx] = y

        return res

    def _decode_single_sample(self, enc: torch.Tensor):
        # enc: (B, T, D)
        token_int = self.s2t_model.ctc.argmax(enc)[0]  # batch size is 1; (T,)
        token_int = torch.unique_consecutive(token_int).cpu().tolist()
        token_int = list(filter(lambda x: x != self.s2t_model.blank_id, token_int))
        token = self.converter.ids2tokens(token_int)
        token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]

        if self.tokenizer is not None:
            text = self.tokenizer.tokens2text(token)
            text_nospecial = self.tokenizer.tokens2text(token_nospecial)
        else:
            text, text_nospecial = None, None

        logging.info(f"best hypo: {text}")

        results = [(text, token, token_int, text_nospecial, None)]
        return results

    @torch.no_grad()
    def decode_long_batched_buffered(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        batch_size: int = 1,
        context_len_in_secs: float = 2,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ):
        """Decode unsegmented long-form speech.

        Args:
            speech: 1D long-form input speech
            batch_size (int): decode this number of segments together in parallel
        """

        sample_rate = self.sample_rate
        frames_per_sec = self.frames_per_sec

        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        lang_id = self.converter.token2id[lang_sym]
        task_id = self.converter.token2id[task_sym]

        buffer_len_in_secs = self.s2t_train_args.preprocessor_conf["speech_length"]
        chunk_len_in_secs = buffer_len_in_secs - 2 * context_len_in_secs
        buffer_len = int(sample_rate * buffer_len_in_secs)
        chunk_len = int(sample_rate * chunk_len_in_secs)

        speech = np.pad(
            speech,
            (
                int(sample_rate * context_len_in_secs),
                int(sample_rate * context_len_in_secs),
            ),
        )
        buffer_list = []
        for i in range(0, len(speech), chunk_len):
            cur_buffer = speech[i : i + buffer_len]
            if len(cur_buffer) < buffer_len:
                buffer_list.append(
                    np.pad(cur_buffer, (0, buffer_len - len(cur_buffer)))
                )
                break
            else:
                buffer_list.append(cur_buffer)

        speech = torch.tensor(np.array(buffer_list)).to(getattr(torch, self.dtype))
        buffer_frames = int(frames_per_sec * buffer_len_in_secs)
        context_frames = int(frames_per_sec * context_len_in_secs)

        unmerged = []
        for idx in range(0, speech.size(0), batch_size):
            cur_speech = speech[idx : idx + batch_size]
            cur_speech_lengths = cur_speech.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=cur_speech.size(1)
            )

            text_prev = torch.tensor([self.s2t_model.na], dtype=torch.long).repeat(
                cur_speech.size(0), 1
            )
            text_prev_lengths = text_prev.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=text_prev.size(1)
            )

            prefix = torch.tensor([lang_id, task_id], dtype=torch.long).repeat(
                cur_speech.size(0), 1
            )
            prefix_lengths = prefix.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=prefix.size(-1)
            )

            batch = {
                "speech": cur_speech,
                "speech_lengths": cur_speech_lengths,
                "text_prev": text_prev,
                "text_prev_lengths": text_prev_lengths,
                "prefix": prefix,
                "prefix_lengths": prefix_lengths,
            }

            # a. To device
            batch = to_device(batch, device=self.device)

            # b. Forward Encoder
            enc, enc_olens = self.s2t_model.encode(**batch)

            intermediate_outs = None
            if isinstance(enc, tuple):
                enc, intermediate_outs = enc

            # enc: (B, T, D)
            enc = enc[
                :, :buffer_frames
            ]  # NOTE(yifan): IMPORTANT: it might be longer due to padding in conv
            batched_token_int = self.s2t_model.ctc.argmax(enc)  # (B, T)
            valid_token_int = batched_token_int[
                :, context_frames:-context_frames
            ].reshape(-1)
            unmerged.append(valid_token_int)

        unmerged = torch.cat(unmerged)
        merged = torch.unique_consecutive(unmerged).cpu().tolist()
        token_int = list(filter(lambda x: x != self.s2t_model.blank_id, merged))
        token = self.converter.ids2tokens(token_int)
        token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]
        text_nospecial = self.tokenizer.tokens2text(token_nospecial)

        return text_nospecial

    def _prepare_inputs(
        self,
        speech: List[Union[str, Path, torch.Tensor, np.ndarray]],
        context_len_in_secs: float,
        text_prev: List[str],
        lang_sym: List[str],
        task_sym: List[str],
    ):
        """Prepare inputs for batched decoding of short- or long-form audio."""

        sample_rate = self.sample_rate
        buffer_len_in_secs = self.s2t_train_args.preprocessor_conf["speech_length"]
        chunk_len_in_secs = buffer_len_in_secs - 2 * context_len_in_secs
        buffer_len = int(sample_rate * buffer_len_in_secs)
        chunk_len = int(sample_rate * chunk_len_in_secs)

        # number of chunks for each audio
        # for short-form audio, it is always 1
        n_chunks = []
        all_speech = []
        all_text_prev = []
        all_prefix = []
        for cur_speech, cur_text_prev, cur_lang_sym, cur_task_sym in zip(
            speech, text_prev, lang_sym, task_sym
        ):
            if isinstance(cur_speech, (str, Path)):
                # load audio as mono-channel (1-D array)
                cur_speech, _ = librosa.load(cur_speech, sr=self.sample_rate)
            elif isinstance(cur_speech, torch.Tensor):
                cur_speech = cur_speech.cpu().numpy()

            if len(cur_speech) <= buffer_len:
                # short-form audio
                n_chunks.append(1)
                cur_speech = librosa.util.fix_length(cur_speech, size=buffer_len)
                all_speech.append(
                    torch.tensor(cur_speech, dtype=getattr(torch, self.dtype))
                )

                cur_text_prev = self.converter.tokens2ids(
                    self.tokenizer.text2tokens(cur_text_prev)
                )
                if self.s2t_model.na in cur_text_prev:
                    cur_text_prev = [self.s2t_model.na]
                all_text_prev.append(torch.tensor(cur_text_prev, dtype=torch.long))

                cur_lang_id = self.converter.token2id[cur_lang_sym]
                cur_task_id = self.converter.token2id[cur_task_sym]
                all_prefix.append(
                    torch.tensor([cur_lang_id, cur_task_id], dtype=torch.long)
                )

            else:
                # long-form audio needs to be split into overlapped segments
                cur_speech = np.pad(
                    cur_speech,
                    (
                        int(sample_rate * context_len_in_secs),
                        int(sample_rate * context_len_in_secs),
                    ),
                )
                buffer_list = []
                for i in range(0, len(cur_speech), chunk_len):
                    cur_buffer = cur_speech[i : i + buffer_len]
                    if len(cur_buffer) < buffer_len:
                        buffer_list.append(
                            np.pad(cur_buffer, (0, buffer_len - len(cur_buffer)))
                        )
                        break
                    else:
                        buffer_list.append(cur_buffer)
                buffer_list = [
                    torch.tensor(x, dtype=getattr(torch, self.dtype))
                    for x in buffer_list
                ]

                n_chunks.append(len(buffer_list))
                all_speech.extend(buffer_list)

                cur_text_prev = self.converter.tokens2ids(
                    self.tokenizer.text2tokens(cur_text_prev)
                )
                if self.s2t_model.na in cur_text_prev:
                    cur_text_prev = [self.s2t_model.na]
                all_text_prev.append(torch.tensor(cur_text_prev, dtype=torch.long))
                for _ in range(len(buffer_list) - 1):
                    all_text_prev.append(
                        torch.tensor([self.s2t_model.na], dtype=torch.long)
                    )

                cur_lang_id = self.converter.token2id[cur_lang_sym]
                cur_task_id = self.converter.token2id[cur_task_sym]
                all_prefix.extend(
                    [
                        torch.tensor([cur_lang_id, cur_task_id], dtype=torch.long)
                        for _ in buffer_list
                    ]
                )

        all_speech = torch.stack(all_speech)
        all_speech_lengths = all_speech.new_full(
            [all_speech.size(0)], dtype=torch.long, fill_value=all_speech.size(1)
        )

        all_text_prev_lengths = torch.tensor(
            [x.size(0) for x in all_text_prev], dtype=torch.long
        )
        all_text_prev = pad_list(all_text_prev, self.s2t_model.eos)

        all_prefix = torch.stack(all_prefix)
        all_prefix_lengths = all_prefix.new_full(
            [all_prefix.size(0)], dtype=torch.long, fill_value=all_prefix.size(1)
        )

        return (
            all_speech,
            all_speech_lengths,
            all_text_prev,
            all_text_prev_lengths,
            all_prefix,
            all_prefix_lengths,
            n_chunks,
        )

    def _get_predictions(
        self,
        outputs: torch.Tensor,
        n_chunks: List[int],
        context_len_in_secs: float,
    ) -> List[str]:
        assert sum(n_chunks) == outputs.size(0)

        frames_per_sec = self.frames_per_sec
        buffer_len_in_secs = self.s2t_train_args.preprocessor_conf["speech_length"]
        buffer_frames = int(frames_per_sec * buffer_len_in_secs)
        context_frames = int(frames_per_sec * context_len_in_secs)

        predictions = []
        start = 0
        for n_chunk in n_chunks:
            cur_outputs = outputs[start : start + n_chunk]

            if len(cur_outputs) == 1:
                # short-form audio
                token_int = cur_outputs[0]
            else:
                token_int = cur_outputs[:, :buffer_frames][
                    :, context_frames:-context_frames
                ].reshape(-1)

            token_int = torch.unique_consecutive(token_int).cpu().tolist()
            token_int = list(filter(lambda x: x != self.s2t_model.blank_id, token_int))
            token = self.converter.ids2tokens(token_int)
            token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]
            text_nospecial = self.tokenizer.tokens2text(token_nospecial)
            predictions.append(text_nospecial)

            start += n_chunk

        return predictions

    @torch.no_grad()
    @typechecked
    def batch_decode(
        self,
        speech: Union[
            str,
            Path,
            torch.Tensor,
            np.ndarray,
            List[Union[str, Path, torch.Tensor, np.ndarray]],
        ],
        batch_size: int = 16,
        context_len_in_secs: float = 4,
        text_prev: Union[str, List[str]] = "<na>",
        lang_sym: Optional[Union[str, List[str]]] = None,
        task_sym: Optional[Union[str, List[str]]] = None,
    ):
        """Decode a batch of audios (either short-form or long-form)."""

        if isinstance(speech, (str, Path, torch.Tensor, np.ndarray)):
            speech = [speech]
            # for nonbatched input, the output will be nonbatched
            is_batched = False
        else:
            is_batched = True

        n_audios = len(speech)

        if isinstance(text_prev, str):
            text_prev = [text_prev] * n_audios

        if lang_sym is None:
            lang_sym = self.lang_sym
        if isinstance(lang_sym, str):
            lang_sym = [lang_sym] * n_audios

        if task_sym is None:
            task_sym = self.task_sym
        if isinstance(task_sym, str):
            task_sym = [task_sym] * n_audios

        # speech, text_prev, lang_sym, task_sym are all lists of the same length
        (
            all_speech,
            all_speech_lengths,
            all_text_prev,
            all_text_prev_lengths,
            all_prefix,
            all_prefix_lengths,
            n_chunks,
        ) = self._prepare_inputs(
            speech, context_len_in_secs, text_prev, lang_sym, task_sym
        )

        all_outputs = []
        for idx in range(0, all_speech.size(0), batch_size):
            batch = {
                "speech": all_speech[idx : idx + batch_size],
                "speech_lengths": all_speech_lengths[idx : idx + batch_size],
                "text_prev": all_text_prev[idx : idx + batch_size],
                "text_prev_lengths": all_text_prev_lengths[idx : idx + batch_size],
                "prefix": all_prefix[idx : idx + batch_size],
                "prefix_lengths": all_prefix_lengths[idx : idx + batch_size],
            }

            batch = to_device(batch, device=self.device)

            enc, enc_olens = self.s2t_model.encode(**batch)

            if isinstance(enc, tuple):
                enc = enc[0]

            all_outputs.append(self.s2t_model.ctc.argmax(enc))  # (B, T)

        all_outputs = torch.cat(all_outputs)

        predictions = self._get_predictions(all_outputs, n_chunks, context_len_in_secs)
        return predictions if is_batched else predictions[0]

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2TextGreedySearch instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2TextGreedySearch: Speech2TextGreedySearch instance.

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

        return Speech2TextGreedySearch(**kwargs)


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
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
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
    streaming: bool,
    quantize_s2t_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    lang_sym: str,
    task_sym: str,
    generate_interctc_outputs: bool,
):
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
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        quantize_s2t_model=quantize_s2t_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        lang_sym=lang_sym,
        task_sym=task_sym,
        use_flash_attn=False,
        generate_interctc_outputs=generate_interctc_outputs,
    )
    speech2text_class = Speech2TextGreedySearch if beam_size == 1 else Speech2Text
    logging.info(f"Speech2Text Class: {speech2text_class}")
    speech2text = speech2text_class.from_pretrained(
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
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, text_nospecial, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], " ", hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
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
                if hyp is not None:
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
                    ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = " ".join(
                        text
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

    group = parser.add_argument_group("The model configuration related")
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
    group.add_argument("--lm_weight", type=float, default=0.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.0, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)

    group.add_argument(
        "--lang_sym", type=str, default="nolang", help="Language symbol."
    )
    group.add_argument("--task_sym", type=str, default="asr", help="Task symbol.")
    group.add_argument(
        "--generate_interctc_outputs",
        type=bool,
        default=False,
        help="Also write intermediate CTC outputs.",
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
