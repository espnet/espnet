#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.lm import LMTask
from espnet2.tasks.s2t import S2TTask
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
        Hypothesis,
    ]
]


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("s2t_config.yml", "s2t.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        s2t_train_config: Union[Path, str] = None,
        s2t_model_file: Union[Path, str] = None,
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
        ctc_weight: float = 0.0,
        lm_weight: float = 0.0,
        ngram_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        streaming: bool = False,
        quantize_s2t_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        category_sym: str = "<en>",
        task_sym: str = "<asr>",
        time_sym: Optional[str] = "<notimestamps>",
    ):
        assert check_argument_types()

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build S2T model
        s2t_model, s2t_train_args = S2TTask.build_model_from_file(
            s2t_train_config, s2t_model_file, device
        )
        s2t_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_s2t_model:
            logging.info("Use quantized s2t model for decoding.")

            s2t_model = torch.quantization.quantize_dynamic(
                s2t_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        decoder = s2t_model.decoder
        ctc = CTCPrefixScorer(ctc=s2t_model.ctc, eos=s2t_model.eos)
        token_list = s2t_model.token_list
        scorers = dict(
            decoder=decoder,
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
            sos=s2t_model.sos,
            eos=s2t_model.eos,
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

        self.category_id = converter.token2id[category_sym]
        self.task_id = converter.token2id[task_sym]
        self.time_id = converter.token2id[time_sym] if time_sym is not None else None

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str]] = None,
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
            text_prev: Previous text used as condition (optional)
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Prepare hyp_primer
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
            hyp_primer = (
                [self.s2t_model.sop]
                + text_prev
                + [self.s2t_model.sos, self.category_id, self.task_id]
            )
        else:
            hyp_primer = [self.s2t_model.sos, self.category_id, self.task_id]
        if self.time_id is not None:
            hyp_primer.append(self.time_id)
        self.beam_search.set_hyp_primer(hyp_primer)

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

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

        assert check_return_type(results)

        return results

    def _decode_interctc(
        self, intermediate_outs: List[Tuple[int, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        assert check_argument_types()

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
        if hasattr(self.beam_search.nn_dict, "decoder"):
            if isinstance(self.beam_search.nn_dict.decoder, S4Decoder):
                # Setup: required for S4 autoregressive generation
                for module in self.beam_search.nn_dict.decoder.modules():
                    if hasattr(module, "setup_step"):
                        module.setup_step()
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

    @torch.no_grad()
    def decode_long(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        segment_sec: float = 30,
        fs: int = 16000,
        condition_on_prev_text: bool = False,
        init_text: Optional[str] = None,
        start_time: Optional[str] = "<0.00>",
        end_time_threshold: str = "<29.00>",
        first_time_sym: str = "<0.00>",
        last_time_sym: str = "<30.00>",
        resolution: float = 0.02,
    ):
        """Decode unsegmented long-form speech.

        Args:
            speech: long-form speech of shape (nsamples,)
            segment_sec: segment length in seconds, default: 30
            fs: sampling rate, default: 16000
            condition_on_prev_text: whether to condition on previous text
            init_text: text used as condition for the first segment
            start_time: the start timestamp symbol of the first segment
            end_time_threshold: the last utterance is considered as incomplete
                if its end timestamp exceeds this threshold
            first_time_sym: first timestamp symbol
            last_time_sym: last timestamp symbol
            resolution: time resolution
        """

        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        segment_len = int(segment_sec * fs)
        start_time_id = (
            self.converter.token2id[start_time] if start_time is not None else None
        )
        end_time_id_threshold = self.converter.token2id[end_time_threshold]
        first_time_id = self.converter.token2id[first_time_sym]
        last_time_id = self.converter.token2id[last_time_sym]

        self.time_id = start_time_id
        logging.warning(f"Overwrite start time as: {start_time}, {start_time_id}")

        utterances = []
        offset = 0
        text_prev = init_text
        while offset < len(speech):
            segment = speech[offset : min(offset + segment_len, len(speech))]
            if len(segment) < segment_len:
                segment = F.pad(segment, (0, segment_len - len(segment)))

            result = self.__call__(
                speech=segment,
                text_prev=text_prev if condition_on_prev_text else None,
            )
            if isinstance(result, tuple):
                result = result[0]

            # NOTE(yifan): sos and eos have been removed
            text, token, token_int, text_nospecial, hyp = result[0]  # best hyp
            token_int = token_int[2:]  # remove category and task

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
            self.time_id = first_time_id

        return utterances

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
    streaming: bool,
    quantize_s2t_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    category_sym: str,
    task_sym: str,
    time_sym: str,
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

    # NOTE(yifan): < and > cannot be passed in command line
    category_sym = f"<{category_sym.lstrip('<').rstrip('>')}>"
    task_sym = f"<{task_sym.lstrip('<').rstrip('>')}>"
    time_sym = f"<{time_sym.lstrip('<').rstrip('>')}>"

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
        streaming=streaming,
        quantize_s2t_model=quantize_s2t_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        category_sym=category_sym,
        task_sym=task_sym,
        time_sym=time_sym,
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
            ibest_writer = writer[f"1best_recog"]
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
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.0,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=0.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.0, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)

    group.add_argument(
        "--category_sym", type=str, default="<en>", help="Category symbol."
    )
    group.add_argument(
        "--task_sym", type=str, default="<transcribe>", help="Task symbol."
    )
    group.add_argument(
        "--time_sym",
        type=str,
        default="<notimestamps>",
        help="First start time symbol.",
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
    group.add_argument(
        "--normalize_length",
        type=str2bool,
        default=False,
        help="If true, best hypothesis is selected by length-normalized scores",
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
