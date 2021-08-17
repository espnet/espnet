#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import k2
import k2.ragged as k2r
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fst.common import compile_HLG
from espnet2.fst.common import find_first_disambig_symbol
from espnet2.fst.lm_rescore import rescore_with_whole_lattice
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


# copied from:
# https://github.com/k2-fsa/snowfall/blob/master/snowfall/training/ctc_graph.py#L13
def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    """Build CTC topology.

    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.
    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


# Modified from: https://github.com/k2-fsa/snowfall/blob/master/snowfall/common.py#L309
def get_texts(best_paths: k2.Fsa) -> List[List[int]]:
    """Extract the texts from the best-path FSAs.

     Args:
         best_paths:  a k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
                  containing multiple FSAs, which is expected to be the result
                  of k2.shortest_path (otherwise the returned values won't
                  be meaningful).  Must have the 'aux_labels' attribute, as
                a ragged tensor.
    Return:
        Returns a list of lists of int, containing the label sequences we
        decoded.
    """
    # remove any 0's or -1's (there should be no 0's left but may be -1's.)

    if isinstance(best_paths.aux_labels, k2.RaggedInt):
        aux_labels = k2r.remove_values_leq(best_paths.aux_labels, 0)
        aux_shape = k2r.compose_ragged_shapes(
            best_paths.arcs.shape(), aux_labels.shape()
        )

        # remove the states and arcs axes.
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    else:
        # remove axis corresponding to states.
        aux_shape = k2r.remove_axis(best_paths.arcs.shape(), 1)
        aux_labels = k2.RaggedInt(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(aux_labels, 0)

    assert aux_labels.num_axes() == 2
    return k2r.to_list(aux_labels)


class k2Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = k2Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech = np.expand_dims(audio, 0) # shape: [batch_size, speech_length]
        >>> speech_lengths = np.array([audio.shape[0]]) # shape: [batch_size]
        >>> batch = {"speech": speech, "speech_lengths", speech_lengths}
        >>> speech2text(batch)
        [(text, token, token_int, score), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 8,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        output_beam_size: int = 8,
        is_ctc_decoding: bool = True,
        lang_dir: Optional[str] = None,
        use_fgram_rescoring: bool = False,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        token_list = asr_model.token_list

        self.is_ctc_decoding = is_ctc_decoding
        self.use_fgram_rescoring = use_fgram_rescoring
        ctc_topo = k2.arc_sort(build_ctc_topo(list(range(len(token_list)))))
        if self.is_ctc_decoding:
            self.decode_graph = ctc_topo
        else:
            assert lang_dir is not None
            lang_dir = Path(lang_dir)
            self.symbol_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
            token_symbol_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
            if not os.path.exists(lang_dir / "HLG.pt"):
                logging.debug("Loading L_disambig.fst.txt")
                with open(lang_dir / "L_disambig.fst.txt") as f:
                    L = k2.Fsa.from_openfst(f.read(), acceptor=False)
                logging.debug("Loading G.fst.txt")
                with open(lang_dir / "G.fst.txt") as f:
                    G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                first_token_disambig_id = find_first_disambig_symbol(token_symbol_table)
                first_word_disambig_id = find_first_disambig_symbol(self.symbol_table)
                HLG = compile_HLG(
                    L=L,
                    G=G,
                    H=ctc_topo,
                    labels_disambig_id_start=first_token_disambig_id,
                    aux_labels_disambig_id_start=first_word_disambig_id,
                )
                torch.save(HLG.as_dict(), lang_dir / "HLG.pt")
            else:
                logging.debug("Loading pre-compiled HLG")
                d = torch.load(lang_dir / "HLG.pt")
                HLG = k2.Fsa.from_dict(d)

            self.decode_graph = HLG

            if self.use_fgram_rescoring:
                first_word_disambig_id = find_first_disambig_symbol(self.symbol_table)
                if not os.path.exists(lang_dir / "G_4_gram.pt"):
                    logging.debug("Loading G_4_gram.fst.txt")
                    with open(lang_dir / "G_4_gram.fst.txt") as f:
                        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                        # G.aux_labels is not needed in later computations, so
                        # remove it here.
                        del G.aux_labels
                        # CAUTION(fangjun): The following line is crucial.
                        # Arcs entering the back-off state have label equal to #0.
                        # We have to change it to 0 here.
                        G.labels[G.labels >= first_word_disambig_id] = 0
                        G = k2.create_fsa_vec([G]).to(device)
                        G = k2.arc_sort(G)
                        torch.save(G.as_dict(), lang_dir / "G_4_gram.pt")
                else:
                    logging.debug("Loading pre-compiled G_4_gram.pt")
                    d = torch.load(lang_dir / "G_4_gram.pt")
                    G = k2.Fsa.from_dict(d).to(device)

                # Add epsilon self-loops to G as we will compose
                # it with the whole lattice later
                G = k2.add_epsilon_self_loops(G)
                G = k2.arc_sort(G)
                G = G.to(device)
                # G.lm_scores is used to replace HLG.lm_scores during
                # LM rescoring.
                G.lm_scores = G.scores.clone()
                self.G = G

        self.decode_graph = self.decode_graph.to(device)
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

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
        logging.info(f"Running on : {device}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.output_beam_size = output_beam_size

    @torch.no_grad()
    def __call__(
        self, batch: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> List[Tuple[Optional[str], Optional[List[str]], List[int], float]]:
        """Inference

        Args:
            batch: Input speech data and corresponding lengths
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        if isinstance(batch["speech"], np.ndarray):
            batch["speech"] = torch.tensor(batch["speech"])
        if isinstance(batch["speech_lengths"], np.ndarray):
            batch["speech_lengths"] = torch.tensor(batch["speech_lengths"])

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        # enc: [N, T, C]
        enc, encoder_out_lens = self.asr_model.encode(**batch)

        # logp_encoder_output: [N, T, C]
        logp_encoder_output = torch.nn.functional.log_softmax(
            self.asr_model.ctc.ctc_lo(enc), dim=2
        )

        batch_size = encoder_out_lens.size(0)
        sequence_idx = torch.arange(0, batch_size).unsqueeze(0).t().to(torch.int32)
        start_frame = torch.zeros([batch_size], dtype=torch.int32).unsqueeze(0).t()
        num_frames = encoder_out_lens.cpu().unsqueeze(0).t().to(torch.int32)
        supervision_segments = torch.cat([sequence_idx, start_frame, num_frames], dim=1)

        supervision_segments = supervision_segments.to(torch.int32)

        dense_fsa_vec = k2.DenseFsaVec(logp_encoder_output, supervision_segments)

        lattices = k2.intersect_dense_pruned(
            self.decode_graph, dense_fsa_vec, 20.0, self.output_beam_size, 30, 10000
        )
        if self.use_fgram_rescoring:
            lattices = rescore_with_whole_lattice(
                lattices, self.G, lm_scale_list=None, need_rescored_lats=True
            )

        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        scores = best_paths.get_tot_scores(
            use_double_scores=True, log_semiring=False
        ).tolist()

        hyps = get_texts(best_paths)
        assert len(scores) == len(hyps)

        results = []

        for token_int, score in zip(hyps, scores):
            # Change integer-ids to tokens
            if self.is_ctc_decoding:
                # convert token_id to text with self.tokenizer
                token = self.converter.ids2tokens(token_int)

                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None
            else:
                # decode with TLG
                # Now token_int, actually stores word indexes.
                # which is lattice.aux_labels
                token = None
                text = " ".join(
                    [self.symbol_table.get(word_idx) for word_idx in token_int]
                )
            results.append((text, token, token_int, score))

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
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    is_ctc_decoding: bool,
    lang_dir: Optional[str],
    use_fgram_rescoring: bool,
):
    assert check_argument_types()
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
    speech2text = k2Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        is_ctc_decoding=is_ctc_decoding,
        lang_dir=lang_dir,
        use_fgram_rescoring=use_fgram_rescoring,
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

    with DatadirWriter(output_dir) as writer:
        start_decoding_time = datetime.datetime.now()
        for batch_idx, (keys, batch) in enumerate(loader):
            if batch_idx % 10 == 0:
                logging.info(f"Processing {batch_idx} batch")
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            # 1-best list of (text, token, token_int)
            results = speech2text(batch)

            for key_idx, (text, token, token_int, score) in enumerate(results):
                key = keys[key_idx]
                best_writer = writer["1best_recog"]
                # Write the result to each file
                if token is not None:
                    best_writer["token"][key] = " ".join(token)
                best_writer["token_int"][key] = " ".join(map(str, token_int))
                best_writer["score"][key] = str(score)

                if text is not None:
                    best_writer["text"][key] = text
        end_decoding_time = datetime.datetime.now()
        decoding_duration = end_decoding_time - start_decoding_time
        logging.info(f"Decoding duration is {decoding_duration.seconds} seconds")


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
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
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
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--streaming", type=str2bool, default=False)

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
    group.add_argument("--is_ctc_decoding", type=str2bool, default=True)
    group.add_argument("--lang_dir", type=str, default=None)
    group.add_argument("--use_fgram_rescoring", type=str2bool, default=False)

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
