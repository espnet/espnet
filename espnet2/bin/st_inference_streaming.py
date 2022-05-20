#!/usr/bin/env python3
import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.encoder.contextual_block_conformer_encoder import (  # noqa: H301
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (  # noqa: H301
    ContextualBlockTransformerEncoder,
)
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.lm import LMTask
from espnet2.tasks.st import STTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search_online import BatchBeamSearchOnline
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args


class Speech2TextStreaming:
    """Speech2TextStreaming class

    Details in "Streaming Transformer ASR with Blockwise Synchronous Beam Search"
    (https://arxiv.org/abs/2006.14941)

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2TextStreaming("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        st_train_config: Union[Path, str],
        st_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        disable_repetition_detection=False,
        decoder_text_length_limit=0,
        encoded_feat_length_limit=0,
    ):
        assert check_argument_types()

        # 1. Build ST model
        scorers = {}
        st_model, st_train_args = STTask.build_model_from_file(
            st_train_config, st_model_file, device
        )
        st_model.to(dtype=getattr(torch, dtype)).eval()

        assert isinstance(
            st_model.encoder, ContextualBlockTransformerEncoder
        ) or isinstance(st_model.encoder, ContextualBlockConformerEncoder)

        decoder = st_model.decoder
        token_list = st_model.token_list
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build BeamSearch object
        weights = dict(
            decoder=1.0,
            lm=lm_weight,
            length_bonus=penalty,
        )

        assert "encoder_conf" in st_train_args
        assert "look_ahead" in st_train_args.encoder_conf
        assert "hop_size" in st_train_args.encoder_conf
        assert "block_size" in st_train_args.encoder_conf
        # look_ahead = st_train_args.encoder_conf['look_ahead']
        # hop_size   = st_train_args.encoder_conf['hop_size']
        # block_size = st_train_args.encoder_conf['block_size']

        assert batch_size == 1

        beam_search = BatchBeamSearchOnline(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=st_model.sos,
            eos=st_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key="full",
            disable_repetition_detection=disable_repetition_detection,
            decoder_text_length_limit=decoder_text_length_limit,
            encoded_feat_length_limit=encoded_feat_length_limit,
        )

        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        assert len(non_batch) == 0

        # TODO(karita): make all scorers batchfied
        logging.info("BatchBeamSearchOnline implementation is selected.")

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
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

        self.st_model = st_model
        self.st_train_args = st_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        if "n_fft" in st_train_args.frontend_conf:
            self.n_fft = st_train_args.frontend_conf["n_fft"]
        else:
            self.n_fft = 512
        if "hop_length" in st_train_args.frontend_conf:
            self.hop_length = st_train_args.frontend_conf["hop_length"]
        else:
            self.hop_length = 128
        if (
            "win_length" in st_train_args.frontend_conf
            and st_train_args.frontend_conf["win_length"] is not None
        ):
            self.win_length = st_train_args.frontend_conf["win_length"]
        else:
            self.win_length = self.n_fft

        self.reset()

    def reset(self):
        self.frontend_states = None
        self.encoder_states = None
        self.beam_search.reset()

    def apply_frontend(
        self, speech: torch.Tensor, prev_states=None, is_final: bool = False
    ):
        if prev_states is not None:
            buf = prev_states["waveform_buffer"]
            speech = torch.cat([buf, speech], dim=0)

        if is_final:
            speech_to_process = speech
            waveform_buffer = None
        else:
            n_frames = (
                speech.size(0) - (self.win_length - self.hop_length)
            ) // self.hop_length
            n_residual = (
                speech.size(0) - (self.win_length - self.hop_length)
            ) % self.hop_length
            speech_to_process = speech.narrow(
                0, 0, (self.win_length - self.hop_length) + n_frames * self.hop_length
            )
            waveform_buffer = speech.narrow(
                0,
                speech.size(0) - (self.win_length - self.hop_length) - n_residual,
                (self.win_length - self.hop_length) + n_residual,
            ).clone()

        # data: (Nsamples,) -> (1, Nsamples)
        speech_to_process = speech_to_process.unsqueeze(0).to(
            getattr(torch, self.dtype)
        )
        lengths = speech_to_process.new_full(
            [1], dtype=torch.long, fill_value=speech_to_process.size(1)
        )
        batch = {"speech": speech_to_process, "speech_lengths": lengths}

        # lenghts: (1,)
        # a. To device
        batch = to_device(batch, device=self.device)

        feats, feats_lengths = self.st_model._extract_feats(**batch)
        if self.st_model.normalize is not None:
            feats, feats_lengths = self.st_model.normalize(feats, feats_lengths)

        # Trimming
        if is_final:
            if prev_states is None:
                pass
            else:
                feats = feats.narrow(
                    1,
                    math.ceil(math.ceil(self.win_length / self.hop_length) / 2),
                    feats.size(1)
                    - math.ceil(math.ceil(self.win_length / self.hop_length) / 2),
                )
        else:
            if prev_states is None:
                feats = feats.narrow(
                    1,
                    0,
                    feats.size(1)
                    - math.ceil(math.ceil(self.win_length / self.hop_length) / 2),
                )
            else:
                feats = feats.narrow(
                    1,
                    math.ceil(math.ceil(self.win_length / self.hop_length) / 2),
                    feats.size(1)
                    - 2 * math.ceil(math.ceil(self.win_length / self.hop_length) / 2),
                )

        feats_lengths = feats.new_full([1], dtype=torch.long, fill_value=feats.size(1))

        if is_final:
            next_states = None
        else:
            next_states = {"waveform_buffer": waveform_buffer}
        return feats, feats_lengths, next_states

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = True
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
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

        feats, feats_lengths, self.frontend_states = self.apply_frontend(
            speech, self.frontend_states, is_final=is_final
        )
        enc, _, self.encoder_states = self.st_model.encoder(
            feats,
            feats_lengths,
            self.encoder_states,
            is_final=is_final,
            infer_mode=True,
        )
        nbest_hyps = self.beam_search(
            x=enc[0],
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
            is_final=is_final,
        )

        ret = self.assemble_hyps(nbest_hyps)
        if is_final:
            self.reset()
        return ret

    def assemble_hyps(self, hyps):
        nbest_hyps = hyps[: self.nbest]
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
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results


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
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    st_train_config: str,
    st_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    sim_chunk_length: int,
    disable_repetition_detection: bool,
    encoded_feat_length_limit: int,
    decoder_text_length_limit: int,
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
    speech2text = Speech2TextStreaming(
        st_train_config=st_train_config,
        st_model_file=st_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        lm_weight=lm_weight,
        penalty=penalty,
        nbest=nbest,
        disable_repetition_detection=disable_repetition_detection,
        decoder_text_length_limit=decoder_text_length_limit,
        encoded_feat_length_limit=encoded_feat_length_limit,
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
            assert len(batch.keys()) == 1

            try:
                if sim_chunk_length == 0:
                    # N-best list of (text, token, token_int, hyp_object)
                    results = speech2text(**batch)
                else:
                    speech = batch["speech"]
                    if (len(speech) // sim_chunk_length) > 1:
                        for i in range(len(speech) // sim_chunk_length):
                            speech2text(
                                speech=speech[
                                    i * sim_chunk_length : (i + 1) * sim_chunk_length
                                ],
                                is_final=False,
                            )
                        results = speech2text(
                            speech[(i + 1) * sim_chunk_length : len(speech)],
                            is_final=True,
                        )
                    else:
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
    group.add_argument(
        "--sim_chunk_length",
        type=int,
        default=0,
        help="The length of one chunk, to which speech will be "
        "divided for evalution of streaming processing.",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--st_train_config", type=str, required=True)
    group.add_argument("--st_model_file", type=str, required=True)
    group.add_argument("--lm_train_config", type=str)
    group.add_argument("--lm_file", type=str)
    group.add_argument("--word_lm_train_config", type=str)
    group.add_argument("--word_lm_file", type=str)

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
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--disable_repetition_detection", type=str2bool, default=False)

    group.add_argument(
        "--encoded_feat_length_limit",
        type=int,
        default=0,
        help="Limit the lengths of the encoded feature" "to input to the decoder.",
    )
    group.add_argument(
        "--decoder_text_length_limit",
        type=int,
        default=0,
        help="Limit the lengths of the text" "to input to the decoder.",
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
