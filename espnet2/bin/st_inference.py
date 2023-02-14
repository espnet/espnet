#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.st import STTask
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
from espnet2.st.espnet_model_seqattn2 import ESPnetSTModelSA2
from espnet2.st.espnet_model_seqattn4 import ESPnetSTModelSA4
from espnet2.st.espnet_model_md2 import ESPnetSTModelMD2


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("st_config.yml", "st.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        st_train_config: Union[Path, str] = None,
        st_model_file: Union[Path, str] = None,
        transducer_conf: dict = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        src_lm_train_config: Union[Path, str] = None,
        src_lm_file: Union[Path, str] = None,
        src_ngram_scorer: str = "full",
        src_ngram_file: Union[Path, str] = None,
        src_token_type: str = None,
        src_bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        asr_maxlenratio: float = 0.0,
        asr_minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.0,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        asr_beam_size: int = 20,
        asr_lm_weight: float = 1.0,
        asr_ngram_weight: float = 0.9,
        asr_penalty: float = 0.0,
        asr_ctc_weight: float = 0.3,
        asr_nbest: int = 1,
        enh_s2t_task: bool = False,
        ctc_greedy: bool = False,
    ):
        assert check_argument_types()

        task = STTask if not enh_s2t_task else EnhS2TTask

        # 1. Build ST model
        scorers = {}
        asr_scorers = {}
        st_model, st_train_args = task.build_model_from_file(
            st_train_config, st_model_file, device
        )
        if enh_s2t_task:
            st_model.inherite_attributes(
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
        st_model.to(dtype=getattr(torch, dtype)).eval()

        if hasattr(st_model, "decoder"):
            decoder = st_model.decoder
        else:
            decoder = None
        token_list = st_model.token_list
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        if ctc_weight > 0:
            assert hasattr(st_model, "st_ctc")
            ctc = CTCPrefixScorer(ctc=st_model.st_ctc, eos=st_model.eos)
            scorers.update(ctc=ctc)

        src_token_list = st_model.src_token_list
        if st_model.use_multidecoder:
            asr_decoder = st_model.extra_asr_decoder
            asr_ctc = CTCPrefixScorer(ctc=st_model.ctc, eos=st_model.src_eos)
            asr_scorers.update(
                decoder=asr_decoder,
                ctc=asr_ctc,
                length_bonus=LengthBonus(len(src_token_list)),
            )
        else:
            asr_decoder = None

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        if src_lm_train_config is not None:
            src_lm, src_lm_train_args = LMTask.build_model_from_file(
                src_lm_train_config, src_lm_file, device
            )
            asr_scorers["lm"] = src_lm.lm

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

        if src_ngram_file is not None:
            if src_ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                src_ngram = NgramFullScorer(src_ngram_file, src_token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                src_ngram = NgramPartScorer(src_ngram_file, src_token_list)
        else:
            src_ngram = None
        asr_scorers["ngram"] = src_ngram

        # 4. Build BeamSearch object
        if st_model.st_use_transducer_decoder:
            beam_search_transducer = BeamSearchTransducer(
                decoder=st_model.decoder,
                joint_network=st_model.st_joint_network,
                beam_size=beam_size,
                lm=scorers["lm"] if "lm" in scorers else None,
                lm_weight=lm_weight,
                penalty=penalty,
                token_list=token_list,
                **transducer_conf,
            )

            beam_search = None
        else:
            beam_search_transducer = None

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
                sos=st_model.sos,
                eos=st_model.eos,
                vocab_size=len(token_list),
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

        asr_weights = dict(
            decoder=1.0 - asr_ctc_weight,
            ctc=asr_ctc_weight,
            lm=asr_lm_weight,
            ngram=asr_ngram_weight,
            length_bonus=asr_penalty,
        )
        asr_beam_search = BeamSearch(
            beam_size=asr_beam_size,
            weights=asr_weights,
            scorers=asr_scorers,
            sos=st_model.src_sos,
            eos=st_model.src_eos,
            vocab_size=len(src_token_list),
            token_list=src_token_list,
            pre_beam_score_key="full",
            return_hs=True
        )
        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in asr_beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                asr_beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected for ASR.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        asr_beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in asr_scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"ASR Beam_search: {asr_beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")        

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = st_train_args.token_type
        if bpemodel is None:
            bpemodel = st_train_args.bpemodel

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

        if src_token_type is None:
            src_token_type = st_train_args.src_token_type
        if src_bpemodel is None:
            src_bpemodel = st_train_args.src_bpemodel

        if src_token_type is None:
            src_tokenizer = None
        elif src_token_type == "bpe":
            if src_bpemodel is not None:
                src_tokenizer = build_tokenizer(token_type=src_token_type, bpemodel=src_bpemodel)
            else:
                src_tokenizer = None
        else:
            src_tokenizer = build_tokenizer(token_type=src_token_type)
        src_converter = TokenIDConverter(token_list=src_token_list)
        logging.info(f"Src Text tokenizer: {src_tokenizer}")

        self.st_model = st_model
        self.st_train_args = st_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.src_converter = src_converter
        self.src_tokenizer = src_tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.asr_beam_search = asr_beam_search
        self.asr_maxlenratio = asr_maxlenratio
        self.asr_minlenratio = asr_minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.asr_nbest = asr_nbest
        self.ctc_greedy = ctc_greedy

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], Union[Hypothesis, TransHypothesis]]]:
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

        # b. Forward Encoder
        enc, _, asr_enc, _ = self.st_model.encode(**batch, return_int_enc=True)
        assert len(enc) == 1, len(enc)
        x = enc[0]

        # Multi-decoder ASR beam search
        if self.st_model.use_multidecoder:
            asr_nbest_hyps = self.asr_beam_search(
                x=asr_enc[0], maxlenratio=self.asr_maxlenratio, minlenratio=self.asr_minlenratio
            )
            
            asr_results = []
            for hyp in asr_nbest_hyps:
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                if isinstance(hyp.hs, List):
                    asr_hs = torch.stack(hyp.hs)
                else:
                    asr_hs = hyp.hs

                src_token_int = hyp.yseq.tolist()
                src_token_int = list(filter(lambda x: x != self.st_model.src_sos, src_token_int))
                src_token_int = list(filter(lambda x: x != self.st_model.src_eos, src_token_int))

                # remove blank symbol id, which is assumed to be 0
                src_token_int = list(filter(lambda x: x != 0, src_token_int))

                # Change integer-ids to tokens
                src_token = self.src_converter.ids2tokens(src_token_int)

                if self.src_tokenizer is not None:
                    src_hyp_text = self.src_tokenizer.tokens2text(src_token)
                else:
                    src_hyp_text = None
                asr_results.append((src_hyp_text, src_token, src_token_int, hyp, asr_hs))

            # Encode 1 best ASR result
            asr_src_text = asr_results[0][0]
            asr_hs = asr_results[0][-1].unsqueeze(0)
            asr_hs = to_device(asr_hs, device=self.device)
            asr_hs_lengths = asr_hs.new_full([1], dtype=torch.long, fill_value=asr_hs.size(1))
            md_enc, _, _ = self.st_model.md_encoder(asr_hs, asr_hs_lengths)
            x = md_enc[0]
            pre_x = enc[0]

        # c. Passed the encoder result and the beam search
        if self.ctc_greedy:
            from itertools import groupby
            lpz = self.st_model.st_ctc.argmax(enc)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != 0, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.st_model.sos] + hyp + [self.st_model.eos]}]
            nbest_hyps = [Hypothesis(
                            score=hyp["score"],
                            yseq=torch.tensor(hyp["yseq"]),
                        ) for hyp in nbest_hyps]
        elif self.st_model.use_multidecoder and self.st_model.use_speech_attn:
            if isinstance(self.st_model, ESPnetSTModelSA2):
                nbest_hyps = self.beam_search(
                    x=pre_x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio, pre_x=x, sa2=True
                )
            elif isinstance(self.st_model, ESPnetSTModelSA4):
                nbest_hyps = self.beam_search(
                    x=pre_x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio, pre_x=asr_enc[0], sa2=True, pre_x2=x
                )
            else:
                nbest_hyps = self.beam_search(
                    x=x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio, pre_x=pre_x, md2=isinstance(self.st_model, ESPnetSTModelMD2)
                )
        elif self.beam_search_transducer:
            logging.info("encoder output length: " + str(x.shape[0]))
            nbest_hyps = self.beam_search_transducer(x)

            best = nbest_hyps[0]
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(
                f"normalized log probability: {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                "best hypo: " + "".join(self.converter.ids2tokens(best.yseq[1:])) + "\n"
            )
        else:
            nbest_hyps = self.beam_search(
                x=x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
        nbest_hyps = nbest_hyps[: self.nbest]
        
        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.st_model.st_use_transducer_decoder else -1
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
        
        if self.st_model.use_multidecoder:
            return (results, asr_results)
        assert check_return_type(results)
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
    asr_maxlenratio: float,
    asr_minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    asr_beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    asr_ctc_weight: float,
    asr_lm_weight: float,
    asr_ngram_weight: float,
    asr_penalty: float,
    asr_nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    st_train_config: Optional[str],
    st_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    src_lm_train_config: Optional[str],
    src_lm_file: Optional[str],
    src_word_lm_train_config: Optional[str],
    src_word_lm_file: Optional[str],
    src_ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    src_token_type: Optional[str],
    src_bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    enh_s2t_task: bool,
    ctc_greedy: bool,
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
        st_train_config=st_train_config,
        st_model_file=st_model_file,
        transducer_conf=transducer_conf,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        src_lm_train_config=src_lm_train_config,
        src_lm_file=src_lm_file,
        src_ngram_file=src_ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        src_token_type=src_token_type,
        src_bpemodel=src_bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        asr_maxlenratio=asr_maxlenratio,
        asr_minlenratio=asr_minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        asr_beam_size=asr_beam_size,
        asr_ctc_weight=asr_ctc_weight,
        asr_lm_weight=asr_lm_weight,
        asr_ngram_weight=asr_ngram_weight,
        asr_penalty=asr_penalty,
        asr_nbest=asr_nbest,
        enh_s2t_task=enh_s2t_task,
        ctc_greedy=ctc_greedy
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = STTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=STTask.build_preprocess_fn(speech2text.st_train_args, False),
        collate_fn=STTask.build_collate_fn(speech2text.st_train_args, False),
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
                # If multi-decoder, then also write ASR results
                if len(results) == 2:
                    asr_results = results[-1]
                    results = results[0]
                else:
                    asr_results = None
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
            
            if asr_results is not None:
                for n, (text, token, token_int, hyp, _) in zip(range(1, asr_nbest + 1), asr_results):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}asr_best_recog"]

                    # Write the result to each file
                    ibest_writer["asr_token"][key] = " ".join(token)
                    ibest_writer["asr_token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["asr_score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["asr_text"][key] = text

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ST Decoding",
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
        "--st_train_config",
        type=str,
        help="ST training configuration",
    )
    group.add_argument(
        "--st_model_file",
        type=str,
        help="ST model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--src_lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--src_lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--src_word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--src_word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--src_ngram_file",
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
        help="enhancement and asr joint model",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--asr_nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--asr_beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument("--asr_penalty", type=float, default=0.0, help="Insertion penalty")
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
        "--asr_maxlenratio",
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
        "--asr_minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--asr_lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--asr_ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--ctc_weight", type=float, default=0.0, help="ST CTC weight")
    group.add_argument("--asr_ctc_weight", type=float, default=0.3, help="ASR CTC weight")

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
        help="The token type for ST model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--src_token_type",
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
        "--src_bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--ctc_greedy",
        type=str2bool,
        default=False,
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
