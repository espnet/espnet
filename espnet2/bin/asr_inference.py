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

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    get_hugging_face_model_lm_head,
    get_hugging_face_model_network,
)
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.beam_search_timesync import BeamSearchTimeSync
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
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

# Alias for typing
ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Union[Hypothesis, ExtTransHypothesis, TransHypothesis],
    ]
]


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        transducer_conf: dict = None,
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
        streaming: bool = False,
        enh_s2t_task: bool = False,
        quantize_asr_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        hugging_face_decoder: bool = False,
        hugging_face_decoder_conf: Dict[str, Any] = {},
        time_sync: bool = False,
        multi_asr: bool = False,
        lid_prompt: bool = False,
        lang_prompt_token: Optional[str] = None,
        nlp_prompt_token: Optional[str] = None,
        prompt_token_file: Optional[str] = None,
    ):
        assert check_argument_types()

        task = ASRTask if not enh_s2t_task else EnhS2TTask

        if quantize_asr_model or quantize_lm:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion(
                "1.5.0"
            ):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build ASR model
        scorers = {}
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

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
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
        if asr_model.use_transducer_decoder:
            # In multi-blank RNNT, we assume all big blanks are
            # just before the standard blank in token_list
            multi_blank_durations = getattr(
                asr_model, "transducer_multi_blank_durations", []
            )[::-1] + [1]
            multi_blank_indices = [
                asr_model.blank_id - i + 1
                for i in range(len(multi_blank_durations), 0, -1)
            ]

            if transducer_conf is None:
                transducer_conf = {}

            beam_search_transducer = BeamSearchTransducer(
                decoder=asr_model.decoder,
                joint_network=asr_model.joint_network,
                beam_size=beam_size,
                lm=scorers["lm"] if "lm" in scorers else None,
                lm_weight=lm_weight,
                multi_blank_durations=multi_blank_durations,
                multi_blank_indices=multi_blank_indices,
                token_list=token_list,
                **transducer_conf,
            )
            beam_search = None
            hugging_face_model = None
            hugging_face_linear_in = None
        elif (
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

            del asr_model.decoder.lm_head
            del asr_model.decoder.decoder

            hugging_face_linear_in = decoder.linear_in
            hugging_face_model.to(device=device).eval()

            if "num_beams" not in hugging_face_decoder_conf:
                hugging_face_decoder_conf[
                    "num_beams"
                ] = hugging_face_model.config.num_beams

            if (
                hugging_face_model.config.pad_token_id is None
                and "pad_token_id" not in hugging_face_decoder_conf
            ):
                hugging_face_decoder_conf[
                    "pad_token_id"
                ] = hugging_face_model.config.eos_token_id

            beam_search = None
            beam_search_transducer = None
        else:
            beam_search_transducer = None
            hugging_face_model = None
            hugging_face_linear_in = None

            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )

            if time_sync:
                if not hasattr(asr_model, "ctc"):
                    raise NotImplementedError(
                        "BeamSearchTimeSync without CTC is not supported."
                    )
                if batch_size != 1:
                    raise NotImplementedError(
                        "BeamSearchTimeSync with batching is not yet supported."
                    )
                logging.info("BeamSearchTimeSync implementation is selected.")

                scorers["ctc"] = asr_model.ctc
                beam_search = BeamSearchTimeSync(
                    beam_size=beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=asr_model.sos,
                    token_list=token_list,
                )
            else:
                beam_search = BeamSearch(
                    beam_size=beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=asr_model.sos,
                    eos=asr_model.eos,
                    vocab_size=len(token_list),
                    token_list=token_list,
                    pre_beam_score_key=None if ctc_weight == 1.0 else "full",
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
                            beam_search.set_streaming_config(asr_train_config)
                            logging.info(
                                "BatchBeamSearchOnlineSim implementation is selected."
                            )
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
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        # compatibility for whisper tokenizer
        preprocessor_conf = getattr(asr_train_args, "preprocessor_conf", {})
        whisper_language = preprocessor_conf.get("whisper_language", None)
        whisper_task = preprocessor_conf.get("whisper_task", None)

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe" or token_type == "hugging_face":
            if bpemodel is not None:
                tokenizer = build_tokenizer(
                    token_type=token_type,
                    bpemodel=bpemodel,
                )
            else:
                tokenizer = None
        elif "whisper" in token_type:
            tokenizer_language = asr_train_args.preprocessor_conf.get(
                "tokenizer_language", "en"
            )
            tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                whisper_language=whisper_language,
                whisper_task=whisper_task,
                non_linguistic_symbols=prompt_token_file,
            )
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        if token_type == "hugging_face":
            converter = HuggingFaceTokenIDConverter(model_name_or_path=bpemodel)
        elif bpemodel not in ["whisper_en", "whisper_multilingual"]:
            converter = TokenIDConverter(token_list=token_list)
        else:
            if "speaker_change_symbol" in preprocessor_conf:
                sot_asr = True
            else:
                sot_asr = False
            converter = OpenAIWhisperTokenIDConverter(
                model_type=bpemodel,
                added_tokens_txt=prompt_token_file,
                language=whisper_language or "en",
                task=whisper_task or "transcribe",
                sot=sot_asr,
            )
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
            if lang_prompt_token is not None:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                a1 = a1[:1] + lang_prompt_token.split() + a1[3:]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )
            elif nlp_prompt_token is not None:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                prompt_tokens = tokenizer.text2tokens(nlp_prompt_token)
                a1 = a1[:2] + prompt_tokens + a1[3:]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )
            elif lid_prompt:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                a1 = a1[:1]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.hugging_face_model = hugging_face_model
        self.hugging_face_linear_in = hugging_face_linear_in
        self.hugging_face_decoder_conf = hugging_face_decoder_conf
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.enh_s2t_task = enh_s2t_task
        self.multi_asr = multi_asr

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> Union[
        ListOfHypothesis,
        Tuple[
            ListOfHypothesis,
            Optional[Dict[int, List[str]]],
        ],
    ]:
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
        if self.multi_asr:
            enc = enc.unbind(dim=1)  # (batch, num_inf, ...) -> num_inf x [batch, ...]
        if self.enh_s2t_task or self.multi_asr:
            # Enh+ASR joint task or Multispkr ASR task
            # NOTE (Wangyou): the return type in this case is List[default_return_type]
            if self.multi_asr:
                num_spk = getattr(self.asr_model, "num_inf", 1)
            else:
                num_spk = getattr(self.asr_model.enh_model, "num_spk", 1)
            assert len(enc) == num_spk, (len(enc), num_spk)
            results = []
            for spk, enc_spk in enumerate(enc, 1):
                logging.info(f"=== [{str(self.asr_model.__class__)}] Speaker {spk} ===")
                if isinstance(enc_spk, tuple):
                    enc_spk = enc_spk[0]
                assert len(enc_spk) == 1, len(enc_spk)

                # c. Passed the encoder result and the beam search
                ret = self._decode_single_sample(enc_spk[0])
                assert check_return_type(ret)
                results.append(ret)

        else:
            # Normal ASR
            intermediate_outs = None
            if isinstance(enc, tuple):
                intermediate_outs = enc[1]
                enc = enc[0]
            assert len(enc) == 1, len(enc)

            # c. Passed the encoder result and the beam search
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

        exclude_ids = [self.asr_model.blank_id, self.asr_model.sos, self.asr_model.eos]
        res = {}
        token_list = self.beam_search.token_list

        for layer_idx, encoder_out in intermediate_outs:
            y = self.asr_model.ctc.argmax(encoder_out)[0]  # batch_size = 1
            y = [x[0] for x in groupby(y) if x[0] not in exclude_ids]
            y = [token_list[x] for x in y]

            res[layer_idx] = y

        return res

    def _decode_single_sample(self, enc: torch.Tensor):
        if self.beam_search_transducer:
            logging.info("encoder output length: " + str(enc.shape[0]))
            nbest_hyps = self.beam_search_transducer(enc)

            best = nbest_hyps[0]
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(
                f"normalized log probability: {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                "best hypo: " + "".join(self.converter.ids2tokens(best.yseq[1:])) + "\n"
            )
        elif self.hugging_face_model:
            num_beams = self.hugging_face_decoder_conf["num_beams"]
            enc = self.hugging_face_linear_in(enc).unsqueeze(0)
            if self.asr_model.decoder.causal_lm:
                forward_args, _ = self.asr_model.decoder.add_prefix_postfix(
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
                    input_ids.repeat(num_beams, 1),
                    inputs_embeds=forward_args["inputs_embeds"].repeat(num_beams, 1, 1),
                    attention_mask=input_ids.repeat(num_beams, 1),
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
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    streaming: bool,
    enh_s2t_task: bool,
    quantize_asr_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    hugging_face_decoder: bool,
    hugging_face_decoder_conf: Dict[str, Any],
    time_sync: bool,
    multi_asr: bool,
    lang_prompt_token: Optional[str],
    nlp_prompt_token: Optional[str],
    prompt_token_file: Optional[str],
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
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        transducer_conf=transducer_conf,
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
        streaming=streaming,
        enh_s2t_task=enh_s2t_task,
        multi_asr=multi_asr,
        quantize_asr_model=quantize_asr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_conf=hugging_face_decoder_conf,
        time_sync=time_sync,
        prompt_token_file=prompt_token_file,
        lang_prompt_token=lang_prompt_token,
        nlp_prompt_token=nlp_prompt_token,
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
                if enh_s2t_task:
                    num_spk = getattr(speech2text.asr_model.enh_model, "num_spk", 1)
                    results = [results for _ in range(num_spk)]

            # Only supporting batch_size==1
            key = keys[0]
            if enh_s2t_task or multi_asr:
                # Enh+ASR joint task
                for spk, ret in enumerate(results, 1):
                    for n, (text, token, token_int, hyp) in zip(
                        range(1, nbest + 1), ret
                    ):
                        # Create a directory: outdir/{n}best_recog_spk?
                        ibest_writer = writer[f"{n}best_recog"]

                        # Write the result to each file
                        ibest_writer[f"token_spk{spk}"][key] = " ".join(token)
                        ibest_writer[f"token_int_spk{spk}"][key] = " ".join(
                            map(str, token_int)
                        )
                        ibest_writer[f"score_spk{spk}"][key] = str(hyp.score)

                        if text is not None:
                            ibest_writer[f"text_spk{spk}"][key] = text

            else:
                # Normal ASR
                encoder_interctc_res = None
                if isinstance(results, tuple):
                    results, encoder_interctc_res = results

                for n, (text, token, token_int, hyp) in zip(
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

                # Write intermediate predictions to
                # encoder_interctc_layer<layer_idx>.txt
                ibest_writer = writer[f"1best_recog"]
                if encoder_interctc_res is not None:
                    for idx, text in encoder_interctc_res.items():
                        ibest_writer[f"encoder_interctc_layer{idx}.txt"][
                            key
                        ] = " ".join(text)


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
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="Whether we are using an enhancement and ASR joint model",
    )
    group.add_argument(
        "--multi_asr",
        type=str2bool,
        default=False,
        help="Whether we are using a monolithic multi-speaker ASR model "
        "(This flag should be False if a speech separation model is used before ASR)",
    )
    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_asr_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to ASR model.",
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
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
    group.add_argument(
        "--hugging_face_decoder_conf",
        type=NestedDictAction,
        default=dict(),
        help="Custom kwargs for the HF .generate()",
    )

    group.add_argument(
        "--transducer_conf",
        default=None,
        help="The keyword arguments for transducer beam search.",
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
    group.add_argument(
        "--time_sync",
        type=str2bool,
        default=False,
        help="Time synchronous beam search.",
    )
    group.add_argument(
        "--lang_prompt_token",
        type=str,
        default=None,
        help="Prompt token for mulitlingual prompting",
    )
    group.add_argument(
        "--nlp_prompt_token",
        type=str,
        default=None,
        help="Prompt token for natural language phrases as prompting",
    )
    group.add_argument(
        "--prompt_token_file",
        type=str,
        default=None,
        help="Prompt token file",
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
