#!/usr/bin/env python3
import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import get_texts
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fst.lm_rescore import nbest_am_lm_scores
from espnet2.tasks.lm import LMTask
from espnet2.tasks.uasr import UASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args

try:
    import k2  # for CI import
except ImportError or ModuleNotFoundError:
    k2 = None


def indices_to_split_size(indices, total_elements: int = None):
    """convert indices to split_size

    During decoding, the api torch.tensor_split should be used.
    However, torch.tensor_split is only available with pytorch >= 1.8.0.
    So torch.split is used to pass ci with pytorch < 1.8.0.
    This fuction is used to prepare input for torch.split.
    """
    if indices[0] != 0:
        indices = [0] + indices

    split_size = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    if total_elements is not None and sum(split_size) != total_elements:
        split_size.append(total_elements - sum(split_size))
    return split_size


class k2Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = k2Speech2Text("uasr_config.yml", "uasr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech = np.expand_dims(audio, 0) # shape: [batch_size, speech_length]
        >>> speech_lengths = np.array([audio.shape[0]]) # shape: [batch_size]
        >>> batch = {"speech": speech, "speech_lengths", speech_lengths}
        >>> speech2text(batch)
        [(text, token, token_int, score), ...]

    """

    def __init__(
        self,
        uasr_train_config: Union[Path, str],
        decoding_graph: str,
        uasr_model_file: Union[Path, str] = None,
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
        search_beam_size: int = 20,
        output_beam_size: int = 20,
        min_active_states: int = 14000,
        max_active_states: int = 56000,
        blank_bias: float = 0.0,
        lattice_weight: float = 1.0,
        is_ctc_decoding: bool = True,
        lang_dir: Optional[str] = None,
        token_list_file: Optional[str] = None,
        use_fgram_rescoring: bool = False,
        use_nbest_rescoring: bool = False,
        am_weight: float = 0.5,
        decoder_weight: float = 0.5,
        nnlm_weight: float = 1.0,
        num_paths: int = 1000,
        nbest_batch_size: int = 500,
        nll_batch_size: int = 100,
    ):
        assert check_argument_types()

        # 1. Build UASR model
        logging.info(f"==========device to build model from: {device}===========")
        uasr_model, uasr_train_args = UASRTask.build_model_from_file(
            uasr_train_config, uasr_model_file, device
        )
        uasr_model.use_collected_training_feats = True
        uasr_model.to(dtype=getattr(torch, dtype)).eval()

        if token_list_file is not None:
            token_list = []
            with open(token_list_file, "r") as tlf:
                for line in tlf.readlines():
                    token, _ = line.split()
                    assert token not in token_list
                    token_list.append(token)
        else:
            token_list = uasr_model.token_list

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            self.lm = lm

        self.is_ctc_decoding = is_ctc_decoding
        self.use_fgram_rescoring = use_fgram_rescoring
        self.use_nbest_rescoring = use_nbest_rescoring

        # load decoding graph
        self.decoding_graph = k2.Fsa.from_dict(torch.load(decoding_graph))
        self.decoding_graph = self.decoding_graph.to(device)

        assert token_type is not None
        tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")
        logging.info(f"Running on : {device}")

        self.uasr_model = uasr_model
        self.uasr_train_args = uasr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.search_beam_size = search_beam_size
        self.output_beam_size = output_beam_size
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states
        self.blank_bias = blank_bias
        self.lattice_weight = lattice_weight
        self.am_weight = am_weight
        self.decoder_weight = decoder_weight
        self.nnlm_weight = nnlm_weight
        self.num_paths = num_paths
        self.nbest_batch_size = nbest_batch_size
        self.nll_batch_size = nll_batch_size
        self.uasr_model_ignore_id = 0

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], float]]:
        """Inference

        Args:
            batch: Input speech data and corresponding lengths
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

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
        # enc: [N, T, C]
        generated_sample, _ = self.uasr_model.inference(**batch)

        # nnet_output: [N, T, C]
        logp_encoder_output = torch.nn.functional.log_softmax(generated_sample, dim=-1)

        # It maybe useful to tune blank_bias.
        # The valid range of blank_bias is [-inf, 0]
        #        logp_encoder_output[:, :, 4] += 0

        batch_size, time_length, _ = generated_sample.shape
        assert batch_size == 1
        sequence_idx = 0
        start_frame = 0
        num_frames = time_length
        supervision_segments = torch.Tensor([[sequence_idx, start_frame, num_frames]])
        supervision_segments = supervision_segments.to(torch.int32)

        # An introduction to DenseFsaVec:
        # https://k2-fsa.github.io/k2/core_concepts/index.html#dense-fsa-vector
        # It could be viewed as a fsa-type lopg_encoder_output,
        # whose weight on the arcs are initialized with logp_encoder_output.
        # The goal of converting tensor-type to fsa-type is using
        # fsa related functions in k2. e.g. k2.intersect_dense_pruned below

        # The term "intersect" is similar to "compose" in k2.
        # The differences is are:
        # for "compose" functions, the composition involves
        # mathcing output label of a.fsa and input label of b.fsa
        # while for "intersect" functions, the composition involves
        # matching input label of a.fsa and input label of b.fsa
        # Actually, in compose functions, b.fsa is inverted and then
        # a.fsa and inv_b.fsa are intersected together.
        # For difference between compose and interset:
        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/fsa_algo.py#L308
        # For definition of k2.intersect_dense_pruned:
        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/autograd.py#L648

        lattices = get_lattice(
            nnet_output=logp_encoder_output,
            decoding_graph=self.decoding_graph,
            supervision_segments=supervision_segments,
            search_beam=self.search_beam_size,
            output_beam=self.output_beam_size,
            min_active_states=self.min_active_states,
            max_active_states=self.max_active_states,
        )

        # lattices.scores is the sum of decode_graph.scores(a.k.a. lm weight) and
        # dense_fsa_vec.scores(a.k.a. am weight) on related arcs.
        # For ctc decoding graph, lattices.scores only store am weight
        # since the decoder_graph only define the ctc topology and
        # has no lm weight on its arcs.
        # While for 3-gram decoding, whose graph is converted from language models,
        # lattice.scores contains both am weights and lm weights
        #
        # It maybe useful to tune lattice.scores
        # The valid range of lattice_weight is [0, inf)
        # The lattice_weight will affect the search of k2.random_paths
        lattices.scores *= self.lattice_weight

        results = []
        if self.use_nbest_rescoring:
            (
                am_scores,
                lm_scores,
                token_ids,
                new2old,
                path_to_seq_map,
                seq_to_path_splits,
            ) = nbest_am_lm_scores(
                lattices, self.num_paths, self.device, self.nbest_batch_size
            )

            ys_pad_lens = torch.tensor([len(hyp) for hyp in token_ids]).to(self.device)
            max_token_length = max(ys_pad_lens)
            ys_pad_list = []
            for hyp in token_ids:
                ys_pad_list.append(
                    torch.cat(
                        [
                            torch.tensor(hyp, dtype=torch.long),
                            torch.tensor(
                                [self.uasr_model_ignore_id]
                                * (max_token_length.item() - len(hyp)),
                                dtype=torch.long,
                            ),
                        ]
                    )
                )

            ys_pad = (
                torch.stack(ys_pad_list).to(torch.long).to(self.device)
            )  # [batch, max_token_length]

            encoder_out = generated_sample.index_select(
                0, path_to_seq_map.to(torch.long)
            ).to(
                self.device
            )  # [batch, T, dim]
            encoder_out_lens = encoder_out_lens.index_select(
                0, path_to_seq_map.to(torch.long)
            ).to(
                self.device
            )  # [batch]

            decoder_scores = -self.uasr_model.batchify_nll(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, self.nll_batch_size
            )

            # padded_value for nnlm is 0
            ys_pad[ys_pad == self.uasr_model_ignore_id] = 0
            nnlm_nll, x_lengths = self.lm.batchify_nll(
                ys_pad, ys_pad_lens, self.nll_batch_size
            )
            nnlm_scores = -nnlm_nll.sum(dim=1)

            batch_tot_scores = (
                self.am_weight * am_scores
                + self.decoder_weight * decoder_scores
                + self.nnlm_weight * nnlm_scores
            )
            split_size = indices_to_split_size(
                seq_to_path_splits.tolist(), total_elements=batch_tot_scores.size(0)
            )
            batch_tot_scores = torch.split(
                batch_tot_scores,
                split_size,
            )

            hyps = []
            scores = []
            processed_seqs = 0
            for tot_scores in batch_tot_scores:
                if tot_scores.nelement() == 0:
                    # the last element by torch.tensor_split may be empty
                    # e.g.
                    # torch.tensor_split(torch.tensor([1,2,3,4]), torch.tensor([2,4]))
                    # (tensor([1, 2]), tensor([3, 4]), tensor([], dtype=torch.int64))
                    break
                best_seq_idx = processed_seqs + torch.argmax(tot_scores)

                assert best_seq_idx < len(token_ids)
                best_token_seqs = token_ids[best_seq_idx]
                processed_seqs += tot_scores.nelement()
                hyps.append(best_token_seqs)
                scores.append(tot_scores.max().item())

            assert len(hyps) == len(split_size)
        else:
            best_paths = one_best_decoding(lattices, use_double_scores=True)
            scores = best_paths.get_tot_scores(
                use_double_scores=True, log_semiring=False
            ).tolist()
            hyps = get_texts(best_paths)

        assert len(scores) == len(hyps)

        for token_int, score in zip(hyps, scores):
            # For decoding methods nbest_rescoring and ctc_decoding
            # hyps stores token_index, which is lattice.labels.

            # convert token_id to text with self.tokenizer
            token = self.converter.ids2tokens(token_int)
            assert self.tokenizer is not None
            text = self.tokenizer.tokens2text(token)
            results.append((text, token, token_int, score))

        assert check_return_type(results)
        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build k2Speech2Text instance from the pretrained model.

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

        return k2Speech2Text(**kwargs)


def inference(
    output_dir: str,
    decoding_graph: str,
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
    uasr_train_config: Optional[str],
    uasr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    word_token_list: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    is_ctc_decoding: bool,
    use_nbest_rescoring: bool,
    num_paths: int,
    nbest_batch_size: int,
    nll_batch_size: int,
    k2_config: Optional[str],
):
    assert is_ctc_decoding, "Currently, only ctc_decoding graph is supported."
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
    with open(k2_config) as k2_config_file:
        dict_k2_config = yaml.safe_load(k2_config_file)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        uasr_train_config=uasr_train_config,
        uasr_model_file=uasr_model_file,
        decoding_graph=decoding_graph,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        token_list_file=word_token_list,
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
        use_nbest_rescoring=use_nbest_rescoring,
        num_paths=num_paths,
        nbest_batch_size=nbest_batch_size,
        nll_batch_size=nll_batch_size,
    )

    speech2text_kwargs = dict(**speech2text_kwargs, **dict_k2_config)
    speech2text = k2Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = UASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=UASRTask.build_preprocess_fn(speech2text.uasr_train_args, False),
        collate_fn=UASRTask.build_collate_fn(speech2text.uasr_train_args, False),
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
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # 1-best list of (text, token, token_int)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")

            for key_idx, (text, token, token_int, score) in enumerate(results):
                key = keys[key_idx]
                best_writer = writer["1best_recog"]
                # Write the result to each file
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
        description="UASR Decoding",
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
        "--uasr_train_config",
        type=str,
        help="UASR training configuration",
    )
    group.add_argument(
        "--uasr_model_file",
        type=str,
        help="UASR model parameter file",
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
        choices=["phn", "word"],
        help="The token type for UASR model. "
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
        "--is_ctc_decoding",
        type=str2bool,
        default=True,
        help="Use ctc topology as decoding graph",
    )
    group.add_argument("--use_nbest_rescoring", type=str2bool, default=False)
    group.add_argument(
        "--num_paths",
        type=int,
        default=1000,
        help="The third argument for k2.random_paths",
    )
    group.add_argument(
        "--nbest_batch_size",
        type=int,
        default=500,
        help="batchify nbest list when computing am/lm scores to avoid OOM",
    )
    group.add_argument(
        "--nll_batch_size",
        type=int,
        default=100,
        help="batch_size when computing nll during nbest rescoring",
    )
    group.add_argument("--decoding_graph", type=str, help="decoding graph")
    group.add_argument(
        "--word_token_list", type=str_or_none, default=None, help="output token list"
    )
    group.add_argument("--k2_config", type=str, help="Config file for decoding with k2")

    return parser


def main(cmd=None):
    assert (
        k2 is not None
    ), "k2 is not installed, please follow 'tools/installers' to install"
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
