"""SOT inference script for ESPnet2 native Whisper.

Uses OpenAI Whisper encoder/decoder with tiktoken for decoding.
Includes SOT constraint scoring and post-processing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from typeguard import typechecked

from espnet2.asr.postprocess.sot_postprocess import process_sot_output
from espnet2.asr.scorers.sot_constraint_scorer import SOTConstraintScorer
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.sot_asr import SOTASRTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none

# Default suppress_tokens from Whisper generation_config.json
WHISPER_SUPPRESS_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50359,
    50360,
    50361,
    50362,
    50363,
]


class SOTBeamSearch(BeamSearch):
    """BeamSearch with probability-based timestamp forcing.

    After combining decoder + constraint scores, if the total probability
    of all timestamp tokens exceeds the max text token probability, suppress
    all text tokens.  This matches Step 4 of the original DiCoW
    WhisperTimeStampLogitsProcessorCustom.
    """

    def __init__(
        self,
        timestamp_begin: int = 50365,
        sot_separator_token_id: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timestamp_begin = timestamp_begin
        self.sot_separator_token_id = sot_separator_token_id

    def search(self, running_hyps, x, pre_x=None):
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)
        for hyp in running_hyps:
            weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
            if self.return_hs:
                hs, scores, states = self.score_full(hyp, x, pre_x=pre_x)
            else:
                scores, states = self.score_full(hyp, x, pre_x=pre_x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]

            # Step 4: probability-based timestamp forcing
            logprobs = F.log_softmax(weighted_scores.float(), dim=-1)
            ts_logprob = logprobs[self.timestamp_begin :].logsumexp(dim=-1)
            max_text_logprob = logprobs[: self.timestamp_begin].max()
            if ts_logprob > max_text_logprob:
                # Save separator score before suppression (it's a text
                # token below timestamp_begin that must be preserved)
                sep_id = self.sot_separator_token_id
                sep_score = (
                    weighted_scores[sep_id].clone() if sep_id is not None else None
                )
                weighted_scores[: self.timestamp_begin] = float("-inf")
                # Restore separator if constraint scorer allowed it
                if (
                    sep_id is not None
                    and sep_score is not None
                    and sep_score > float("-inf")
                ):
                    weighted_scores[sep_id] = sep_score

            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            weighted_scores += hyp.score

            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                if self.return_hs:
                    new_hs = hyp.hs + [hs.squeeze(0)]
                else:
                    new_hs = []
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                        hs=new_hs,
                    )
                )

            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps


logger = logging.getLogger(__name__)


class TiktokenDecoderAdapter:
    """Makes tiktoken encoding act like HF tokenizer for process_sot_output."""

    def __init__(self, encoding, token_list=None):
        self.encoding = encoding
        self.pad_token_id = None
        self.token_list = token_list

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            try:
                tokens.append(self.encoding.decode([i]))
            except (KeyError, ValueError):
                # Added token outside tiktoken's vocab — use token_list
                if self.token_list is not None and i < len(self.token_list):
                    tokens.append(self.token_list[i])
                else:
                    tokens.append(f"<unk_{i}>")
        return tokens

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class SOTSpeech2Text:
    """SOT Speech2Text class using native OpenAI Whisper + tiktoken.

    Examples:
        >>> speech2text = SOTSpeech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]
    """

    @typechecked
    def __init__(
        self,
        asr_train_config: Union[Path, str, None] = None,
        asr_model_file: Union[Path, str, None] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        dtype: str = "float32",
        beam_size: int = 5,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        use_sot_constraint: bool = True,
        separator_token: Optional[str] = None,
    ):
        # Build model from SOTASRTask
        asr_model, asr_train_args = SOTASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder
        token_list = asr_model.token_list

        # Set up tiktoken for text decoding
        self._setup_tiktoken(asr_train_args, token_list)

        # Build hyp_primer: [SOS, <|en|>, <|transcribe|>]
        # SOS = <|startoftranscript|>, which is asr_model.sos
        hyp_primer = [asr_model.sos]
        if "<|en|>" in token_list:
            hyp_primer.append(token_list.index("<|en|>"))
        if "<|transcribe|>" in token_list:
            hyp_primer.append(token_list.index("<|transcribe|>"))
        begin_index = len(hyp_primer)

        # Build scorers
        scorers = dict(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )
        weights = dict(
            decoder=1.0,
            length_bonus=0.0,
        )

        # Auto-detect SOT token IDs and build constraint scorer
        self.separator_token_id = None

        if use_sot_constraint:
            try:
                timestamp_begin = token_list.index("<|0.00|>")
                no_timestamps_id = token_list.index("<|notimestamps|>")
                # Determine separator token: use override if provided,
                # otherwise try <sc> (ESPnet-native) then ???? (HF-trained)
                if separator_token is not None:
                    sc_id = token_list.index(separator_token)
                elif "<sc>" in token_list:
                    sc_id = token_list.index("<sc>")
                elif "????" in token_list:
                    sc_id = token_list.index("????")
                else:
                    raise ValueError("No separator token found in token_list")
                self.separator_token_id = sc_id

                sot_scorer = SOTConstraintScorer(
                    vocab_size=len(token_list),
                    eos=asr_model.eos,
                    timestamp_begin=timestamp_begin,
                    no_timestamps_token_id=no_timestamps_id,
                    sot_separator_token_id=sc_id,
                    suppress_token_ids=WHISPER_SUPPRESS_TOKENS,
                    begin_index=begin_index,
                )
                scorers["sot_constraint"] = sot_scorer
                weights["sot_constraint"] = 1.0

                logger.info(
                    f"SOT constraint scorer enabled: "
                    f"timestamp_begin={timestamp_begin}, "
                    f"sc={sc_id}, begin_index={begin_index}"
                )
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Could not set up SOT constraint scorer: {e}. "
                    f"Proceeding without constraints."
                )

        # Build beam search with probability-based timestamp forcing
        beam_search = SOTBeamSearch(
            timestamp_begin=(
                timestamp_begin
                if use_sot_constraint
                else (
                    token_list.index("<|0.00|>") if "<|0.00|>" in token_list else 50365
                )
            ),
            sot_separator_token_id=self.separator_token_id,
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key="full",
            hyp_primer=hyp_primer,
        )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()

        # Keep maxlenratio=0.0 so that ESPnet beam search enables end_detect
        # (early stopping when EOS hypotheses converge).  The decoder's
        # positional-embedding limit (448) is enforced via clamping inside
        # forward_one_step, not by setting a negative maxlenratio (which
        # would disable end_detect entirely).

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.token_list = token_list
        self.begin_index = begin_index

    def _setup_tiktoken(self, asr_train_args, token_list):
        """Set up tiktoken encoding for text decoding."""
        try:
            import whisper

            # Use num_languages=100 for v3/turbo (51866 base vocab)
            num_languages = 100 if len(token_list) > 51865 else 99
            tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, num_languages=num_languages
            )
            self.tiktoken_adapter = TiktokenDecoderAdapter(
                tokenizer.encoding, token_list=token_list
            )
            logger.info("Tiktoken decoder adapter ready")
        except Exception as e:
            logger.warning(f"Failed to set up tiktoken: {e}")
            self.tiktoken_adapter = None

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Run inference.

        Args:
            speech: (T_samples,) audio waveform

        Returns:
            List of (text, tokens, token_int, hypothesis)
        """
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

        speech = speech.to(self.device)
        lengths = lengths.to(self.device)

        # Encode
        enc, enc_olens = self.asr_model.encode(speech, lengths)
        assert len(enc) == 1, len(enc)

        # Beam search decode
        nbest_hyps = self.beam_search(
            x=enc[0],
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
        )

        results = []
        for hyp in nbest_hyps[: self.nbest]:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # Remove primer prefix and EOS from yseq
            token_int = hyp.yseq[self.begin_index :].tolist()

            if len(token_int) > 0 and token_int[-1] == self.asr_model.eos:
                token_int = token_int[:-1]

            token = [self.token_list[t] for t in token_int]

            # Post-process with tiktoken adapter
            if (
                self.tiktoken_adapter is not None
                and self.separator_token_id is not None
            ):
                sep_str = self.token_list[self.separator_token_id]
                per_spk, raw_transcript = process_sot_output(
                    token_int=token_int,
                    hf_tokenizer=self.tiktoken_adapter,
                    separator_token_id=self.separator_token_id,
                    separator_str=sep_str,
                )
                text = raw_transcript
            elif self.tiktoken_adapter is not None:
                text = self.tiktoken_adapter.convert_tokens_to_string(
                    self.tiktoken_adapter.convert_ids_to_tokens(token_int)
                )
            else:
                text = " ".join(token)

            results.append((text, token, token_int, hyp))

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> "SOTSpeech2Text":
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

                d = ModelDownloader()
                kwargs.update(**d.download_and_unpack(model_tag))
            except ImportError:
                raise ImportError(
                    "espnet_model_zoo is not installed. "
                    "Install with: pip install espnet_model_zoo"
                )

        return SOTSpeech2Text(**kwargs)


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
    normalize_length: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    use_sot_constraint: bool = True,
    separator_token: Optional[str] = None,
):
    """Run SOT inference."""
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    device = "cuda" if ngpu >= 1 else "cpu"
    set_all_random_seed(seed)

    speech2text = SOTSpeech2Text(
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
        normalize_length=normalize_length,
        use_sot_constraint=use_sot_constraint,
        separator_token=separator_token,
    )

    # Build data iterator
    loader = SOTASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SOTASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=SOTASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            speech = batch["speech"][0]

            try:
                results = speech2text(speech)
            except Exception as e:
                logging.warning(f"Utterance {keys} failed: {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [(" ", ["<space>"], [2], hyp)] * nbest

            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                ibest_writer = writer[f"{n}best_recog"]
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)
                if text is not None:
                    ibest_writer["text"][key] = text


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="SOT ASR Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x,
        default="INFO",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logging level",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_workers", type=int, default=1)

    group = parser.add_argument_group("Input data")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none, default=None)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Model")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument("--token_type", type=str_or_none, default=None)
    group.add_argument("--bpemodel", type=str_or_none, default=None)

    group = parser.add_argument_group("Decoding")
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--beam_size", type=int, default=5)
    group.add_argument("--penalty", type=float, default=0.0)
    group.add_argument("--maxlenratio", type=float, default=0.0)
    group.add_argument("--minlenratio", type=float, default=0.0)
    group.add_argument("--nbest", type=int, default=1)
    group.add_argument("--normalize_length", type=str2bool, default=False)
    group.add_argument(
        "--use_sot_constraint",
        type=str2bool,
        default=True,
        help="Enable SOT constraint scorer for structured decoding",
    )
    group.add_argument(
        "--separator_token",
        type=str_or_none,
        default=None,
        help="Override speaker separator token (default: auto-detect <sc> or ????)",
    )

    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
