#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet2.st.decoder.transformer_md_decoder import TransformerMDDecoder
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.mt import MTTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.st import STTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


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
        mt_train_config: Union[Path, str] = None,
        mt_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        md_lm_train_config: Union[Path, str] = None,
        md_lm_file: Union[Path, str] = None,
        md_asr_train_config: Union[Path, str] = None,
        md_asr_file: Union[Path, str] = None,
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
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        src_token_type: str = None,
        src_bpemodel: str = None,
        md_beam_size: int = 20,
        md_nbest: int = 1,
        md_penalty: float = 0.0,
        md_maxlenratio: float = 0.0,
        md_minlenratio: float = 0.0,
        md_ctc_weight: float = 0.3,
        md_lm_weight: float = 1.0,
        md_asr_weight: float = 1.0,
        mt_weight: float = 0.0,
    ):
        assert check_argument_types()

        # 1. Build ST model
        scorers = {}
        asr_scorers = {}

        st_model, st_train_args = STTask.build_model_from_file(
            st_train_config, st_model_file, device
        )
        st_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = st_model.decoder
        if isinstance(decoder,TransformerMDDecoder):
            self.speech_attn = True
        asr_decoder = st_model.asr_decoder
        token_list = st_model.token_list
        if getattr(st_model,"src_token_list", None) is None:
            src_token_list = st_train_args.src_token_list
        else:
            src_token_list = st_model.src_token_list

        asr_ctc = CTCPrefixScorer(ctc=st_model.ctc, eos=st_model.src_eos)
        asr_scorers.update(
            decoder=asr_decoder,
            ctc=asr_ctc,
            length_bonus=LengthBonus(len(src_token_list)),
        )
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build MD ASR
        self.asr_model=None
        if md_asr_train_config is not None:
            md_asr, md_asr_train_args = ASRTask.build_model_from_file(
                md_asr_train_config, md_asr_file, device
            )
            md_asr.to(dtype=getattr(torch, dtype)).eval()
            self.asr_model=md_asr
            asr_scorers["asr"] = md_asr.decoder

        # 2. Build MD Language model
        if md_lm_train_config is not None:
            md_lm, md_lm_train_args = LMTask.build_model_from_file(
                md_lm_train_config, md_lm_file, device
            )
            asr_scorers["lm"] = md_lm.lm

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 2. Build MT model
        if mt_train_config is not None:
            mt, mt_train_args = MTTask.build_model_from_file(
                mt_train_config, mt_file, device
            )
            mt.to(dtype=getattr(torch, dtype)).eval()
            self.mt_model = mt
            scorers["mt"] = mt.decoder

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
        asr_scorers["ngram"] = None

        # ASR BeamSearch Object
        asr_weights = dict(
            decoder=1.0 - md_ctc_weight,
            ctc=md_ctc_weight,
            lm=md_lm_weight,
            asr=md_asr_weight,
            length_bonus=md_penalty,
        )
        asr_beam_search = BeamSearch(
            beam_size=md_beam_size,
            weights=asr_weights,
            scorers=asr_scorers,
            sos=st_model.src_sos,
            eos=st_model.src_eos,
            vocab_size=len(src_token_list),
            token_list=src_token_list,
            pre_beam_score_key=None if md_ctc_weight == 1.0 else "full",
            return_hidden=True,
        )

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0,
            lm=lm_weight,
            mt=mt_weight,
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

        if batch_size == 1:
            non_batch = [
                k
                for k, v in asr_beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                asr_beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        asr_beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in asr_scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {asr_beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = st_train_args.token_type
        if bpemodel is None:
            bpemodel = st_train_args.bpemodel

        if src_token_type is None:
            src_token_type = st_train_args.src_token_type
        if src_bpemodel is None:
            src_bpemodel = st_train_args.src_bpemodel

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
            asr_tokenizer = None
        elif src_token_type == "bpe":
            if src_bpemodel is not None:
                asr_tokenizer = build_tokenizer(token_type=src_token_type, bpemodel=src_bpemodel)
            else:
                asr_tokenizer = None
        else:
            asr_tokenizer = build_tokenizer(token_type=src_token_type)
        asr_converter = TokenIDConverter(token_list=src_token_list)
        logging.info(f"ASR Text tokenizer: {asr_tokenizer}")

        self.st_model = st_model
        self.st_train_args = st_train_args
        self.device = device
        self.dtype = dtype

        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.nbest = nbest


        self.asr_converter = asr_converter
        self.asr_tokenizer = asr_tokenizer
        self.asr_beam_search = asr_beam_search
        self.md_maxlenratio = md_maxlenratio
        self.md_minlenratio = md_minlenratio
        self.md_nbest = md_nbest

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], src_text: Optional[torch.Tensor] = None
    ) -> List[Tuple[Optional[str], Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            src_text, text, token, token_int, hyp

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
        enc, enc_lens = self.st_model.encode(**batch)
        assert len(enc) == 1, len(enc)

        if src_text is not None:
            # data: (Nsamples,) -> (1, Nsamples)
            src_text = src_text.unsqueeze(0)
            src_text_lengths = src_text.new_full([1], dtype=torch.long, fill_value=src_text.size(1))
            src_text_in, _ = add_sos_eos(src_text, self.st_model.src_sos, self.st_model.src_eos, self.st_model.ignore_id)
            src_text_in_lens = src_text_lengths + 1
            decoder_out, _, hs_dec_asr = self.st_model.asr_decoder(
                enc, enc_lens, src_text_in, src_text_in_lens, return_hidden=True
            )
            ys_hat = decoder_out.argmax(dim=-1)
            asr_nbest_hyps = [
                    Hypothesis(
                        score=None,
                        yseq=ys_hat[0],
                        scores=None,
                        states=None,
                        hs=hs_dec_asr[0])
                    ]
        else:
            md_asr_x = None
            if self.asr_model is not None:
                asr_enc, _ = self.asr_model.encode(**batch)
                md_asr_x = asr_enc[0]
            # c. Passed the encoder result and the beam search
            asr_nbest_hyps = self.asr_beam_search(
                x=enc[0], maxlenratio=self.md_maxlenratio, minlenratio=self.md_minlenratio, md_asr_x=md_asr_x
            )
            asr_nbest_hyps = asr_nbest_hyps[: self.md_nbest]

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
            src_token = self.asr_converter.ids2tokens(src_token_int)

            if self.tokenizer is not None:
                src_hyp_text = self.asr_tokenizer.tokens2text(src_token)
            else:
                src_hyp_text = None
            asr_results.append((src_hyp_text, src_token, src_token_int, hyp, asr_hs))

        # Currently only support nbest 1
        asr_src_text = asr_results[0][0]
        asr_hs = asr_results[0][-1].unsqueeze(0)
        asr_hs = to_device(asr_hs, device=self.device)
        asr_hs_lengths = asr_hs.new_full([1], dtype=torch.long, fill_value=asr_hs.size(1))
        enc_mt, enc_mt_lengths, _ = self.st_model.encoder_mt(asr_hs, asr_hs_lengths)
        if self.speech_attn:
            x = enc[0]
            md_x = enc_mt[0]
        else:
            x = enc_mt[0]
            md_x = None

        mt_x=None
        if self.mt_model is not None:
            if src_text is not None:
                asr_text_token = src_text
            else:
                asr_text_token = torch.tensor(asr_results[0][2]).unsqueeze(0)
            asr_text_token = asr_text_token.to(torch.long)
            asr_text_lengths = asr_text_token.new_full([1], dtype=torch.long, fill_value=asr_text_token.size(1))
            mt_batch = {"src_text": asr_text_token, "src_text_lengths": asr_text_lengths}
            mt_batch = to_device(mt_batch, device=self.device)
            mt_enc, _ = self.mt_model.encode(**mt_batch)
            mt_x = mt_enc[0]


        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio, md_x = md_x, mt_x = mt_x
        )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((asr_src_text,text, token, token_int, hyp))

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


def inference_md(
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
    st_train_config: Optional[str],
    st_model_file: Optional[str],
    mt_train_config: Optional[str],
    mt_file: Optional[str],
    md_asr_train_config: Optional[str],
    md_asr_file: Optional[str],
    md_lm_train_config: Optional[str],
    md_lm_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    md_beam_size: int,
    md_nbest: int,
    md_penalty: float,
    md_maxlenratio: float,
    md_minlenratio: float,
    md_ctc_weight: float,
    md_lm_weight: float,
    md_asr_weight: float,
    mt_weight: float,
    allow_variable_data_keys: bool,
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
        md_lm_train_config=md_lm_train_config,
        md_lm_file=md_lm_file,
        md_asr_train_config=md_asr_train_config,
        md_asr_file=md_asr_file,
        mt_train_config=mt_train_config,
        mt_file=mt_file,
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
        md_beam_size=md_beam_size,
        md_nbest=md_nbest,
        md_penalty=md_penalty,
        md_maxlenratio=md_maxlenratio,
        md_minlenratio=md_minlenratio,
        md_ctc_weight=md_ctc_weight,
        md_lm_weight=md_lm_weight,
        md_asr_weight=md_asr_weight,
        mt_weight=mt_weight,
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
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            for n, (src_text, text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text

                if src_text is not None:
                    ibest_writer["src_text"][key] = src_text
