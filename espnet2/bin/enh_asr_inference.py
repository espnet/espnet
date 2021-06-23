#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.enh_asr import EnhASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


class Speech2Text:
    """Speech2Text class.

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]
    """

    def __init__(
        self,
        joint_train_config: Union[Path, str],
        joint_model_file: Union[Path, str] = None,
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
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
    ):
        assert check_argument_types()

        # 1. Build Joint model
        scorers = {}
        joint_model, joint_train_args = EnhASRTask.build_model_from_file(
            joint_train_config, joint_model_file, device
        )
        joint_model.eval()

        decoder = joint_model.asr_subclass.decoder
        ctc = CTCPrefixScorer(ctc=joint_model.ctc, eos=joint_model.eos)
        token_list = joint_model.token_list
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
            scorers["lm"] = lm.lm

        # 3. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=joint_model.sos,
            eos=joint_model.eos,
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

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = joint_train_args.token_type
        if bpemodel is None:
            bpemodel = joint_train_args.bpemodel

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

        self.joint_model = joint_model
        self.joint_train_args = joint_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self,
        speech_mix: Union[torch.Tensor, np.ndarray],
        speech_refs: List[Union[torch.Tensor, np.ndarray]] = None,
    ) -> List[
        List[Tuple[Optional[float], Optional[str], List[str], List[int], Hypothesis]]
    ]:
        """Inference.

        Args:
            data: Input speech data
        Returns:
            sdr, text, token, token_int, hyp
        """
        assert check_argument_types()
        speech = speech_mix

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        if speech_refs is not None and isinstance(speech_refs[0], np.ndarray):
            speech_refs = [torch.from_numpy(srf) for srf in speech_refs]

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)
        feats, f_lens = self.joint_model.enh_subclass.encoder(
            batch["speech"], batch["speech_lengths"]
        )
        feats, _, _ = self.joint_model.enh_subclass.separator(feats, f_lens)
        speech_pre_lengths = batch["speech_lengths"]
        speech_pre = [
            self.joint_model.enh_subclass.decoder(f, speech_pre_lengths)[0]
            for f in feats
        ]
        if self.joint_model.cal_enh_loss:
            ref = np.array(torch.stack(speech_refs, dim=0).squeeze())  # nspk,T
            inf = np.array(torch.stack(speech_pre, dim=1).squeeze())
            sdr, sir, sar, perm = bss_eval_sources(ref, inf, compute_permutation=True)
        else:
            sdr, perm = None, np.arange(0, len(speech_pre))

        # b. Forward Encoder
        results_list = []
        # For each predicted spk
        for idx, p in enumerate(perm):
            speech_spk = speech_pre[int(p)]
            # enc, _ = self.joint_model.encode(speech_spk, batch['speech_lengths'])
            enc, _ = self.joint_model.asr_subclass.encode(
                speech_spk, speech_pre_lengths
            )
            assert len(enc) == 1, len(enc)

            # c. Passed the encoder result and the beam search
            nbest_hyps = self.beam_search(
                x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
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
                if sdr is not None:
                    sdr_rslt = float(sdr[idx])
                else:
                    sdr_rslt = None

                results.append((sdr_rslt, text, token, token_int, hyp))

            results_list.append(results)

        assert check_return_type(results_list)
        return results_list


def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    fs: int,
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
    joint_train_config: str,
    joint_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    normalize_output_wav: bool,
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
    speech2text = Speech2Text(
        joint_train_config=joint_train_config,
        joint_model_file=joint_model_file,
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
    )

    # 3. Build data-iterator
    loader = EnhASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=EnhASRTask.build_preprocess_fn(
            speech2text.joint_train_args, False
        ),
        collate_fn=EnhASRTask.build_collate_fn(speech2text.joint_train_args, False),
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
            speech_refs = [v[0] for k, v in batch.items() if k.startwith("speech_ref")]

            # N-best list of (text, token, token_int, hyp_object)
            logging.info(f"keys: {keys}")
            results_list = speech2text(batch["speech_mix"], speech_refs)

            for spk_idx, results in enumerate(results_list):
                # Only supporting batch_size==1
                key = keys[0]
                for n, (sdr, text, token, token_int, hyp) in zip(
                    range(1, nbest + 1), results
                ):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog_spk{spk_idx+1}"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text

                    if sdr is not None:
                        writer[f"SDR_spk{spk_idx + 1}"][key] = str(sdr)


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

    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
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
    group.add_argument("--joint_train_config", type=str, required=True)
    group.add_argument("--joint_model_file", type=str, required=True)
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
        default=0.2,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")

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

    group = parser.add_argument_group("Output wav related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
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
