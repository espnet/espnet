#!/usr/bin/env python3

"""Script to run the inference of speech-to-speech translation model."""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.s2st import S2STTask
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
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args


class Speech2Speech:
    """Speech2Speech class."""

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        st_subtask_maxlenratio: float = 1.5,
        st_subtask_minlenratio: float = 0.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        nbest: int = 1,
        beam_size: int = 5,
        penalty: float = 0.0,
        st_subtask_beam_size: int = 5,
        st_subtask_penalty: float = 0.0,
        st_subtask_nbest: int = 1,
        st_subtask_token_type: str = None,
        st_subtask_bpemodel: str = None,
        vocoder_config: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        prefer_normalized_feats: bool = False,
    ):
        """Initialize Speech2Speech module."""
        assert check_argument_types()

        # setup model
        model, train_args = S2STTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.s2st_type = self.model.s2st_type
        self.preprocess_fn = S2STTask.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.st_subtask_maxlenratio = st_subtask_maxlenratio
        self.st_subtask_minlenratio = st_subtask_minlenratio
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        self.vocoder = None
        self.prefer_normalized_feats = prefer_normalized_feats
        if self.model.require_vocoder:
            vocoder = S2STTask.build_vocoder_from_file(
                vocoder_config, vocoder_file, model, device
            )
            if isinstance(vocoder, torch.nn.Module):
                vocoder.to(dtype=getattr(torch, dtype)).eval()
            self.vocoder = vocoder
        logging.info(f"S2ST:\n{self.model}")
        if self.vocoder is not None:
            logging.info(f"Vocoder:\n{self.vocoder}")

        # setup decoding config
        self.decode_conf = {}  # use for specotrogram-based decoding
        scorers = {}  # use for beam-search decoding
        st_subtask_scorers = {}  # use for beam-search in st_subtask
        if self.s2st_type == "translatotron":
            self.decode_conf.update(
                threshold=threshold,
                maxlenratio=maxlenratio,
                minlenratio=minlenratio,
                use_att_constraint=use_att_constraint,
                forward_window=forward_window,
                backward_window=backward_window,
                use_teacher_forcing=use_teacher_forcing,
            )
        elif self.s2st_type == "discrete_unit" or self.s2st_type == "unity":
            decoder = model.synthesizer
            token_list = model.unit_token_list
            scorers.update(decoder=decoder, length_bonus=LengthBonus(len(token_list)))
            weights = dict(
                decoder=1.0,
                length_bonus=penalty,
            )

            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=model.unit_sos,
                eos=model.unit_eos,
                vocab_size=len(token_list),
                token_list=None,  # No need to print out the lengthy discrete unit
                pre_beam_score_key="full",
            )

            # TODO(karita): make all scorers batchfied
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

            self.beam_search = beam_search

            # Further define st_subtask decoder
            if self.s2st_type == "unity":
                st_subtask_decoder = model.st_decoder
                st_subtask_scorers.update(
                    decoder=st_subtask_decoder,
                    length_bonus=LengthBonus(len(model.tgt_token_list)),
                )
                st_subtask_weights = {
                    "decoder": 1.0,
                    "length_bonus": st_subtask_penalty,
                }
                logging.info("model sos eos: {}".format(model.eos))
                st_subtask_beam_search = BeamSearch(
                    beam_size=st_subtask_beam_size,
                    weights=st_subtask_weights,
                    scorers=st_subtask_scorers,
                    sos=model.sos,
                    eos=model.eos,
                    vocab_size=len(model.tgt_token_list),
                    pre_beam_score_key="full",
                    return_hs=True,
                )
                # TODO(karita): make all st_subtask_scorers batchfied
                non_batch = [
                    k
                    for k, v in st_subtask_beam_search.full_scorers.items()
                    if not isinstance(v, BatchScorerInterface)
                ]
                if len(non_batch) == 0:
                    st_subtask_beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
                else:
                    logging.warning(
                        f"As non-batch st_subtask_scorers {non_batch} are found, "
                        f"fall back to non-batch implementation."
                    )
                st_subtask_beam_search.to(
                    device=device, dtype=getattr(torch, dtype)
                ).eval()
                for st_subtask_scorers in st_subtask_scorers.values():
                    if isinstance(st_subtask_scorers, torch.nn.Module):
                        st_subtask_scorers.to(
                            device=device, dtype=getattr(torch, dtype)
                        ).eval()
                logging.info(f"st_subtask Beam_search: {st_subtask_beam_search}")
                logging.info(f"st_subtask Decoding device={device}, dtype={dtype}")

                self.st_subtask_beam_search = st_subtask_beam_search

                # NOTE(jiatong): we here regard the st_subtask as target text
                #                but it may also be source text
                if st_subtask_token_type is None:
                    st_subtask_token_type = train_args.tgt_token_type
                elif st_subtask_token_type == "bpe":
                    if st_subtask_tokenizer is not None:
                        self.st_subtask_tokenizer = build_tokenizer(
                            token_type=st_subtask_token_type,
                            bpemodel=st_subtask_bpemodel,
                        )
                    else:
                        self.st_subtask_tokenizer = None
                else:
                    self.st_subtask_tokenizer = build_tokenizer(
                        token_type=st_subtask_token_type
                    )
                self.st_subtask_converter = TokenIDConverter(
                    token_list=self.model.tgt_token_list
                )

        else:
            raise NotImplementedError(
                "Not recognized s2st type of {}".format(self.s2st_type)
            )

    @torch.no_grad()
    def __call__(
        self,
        src_speech: Union[torch.Tensor, np.ndarray],
        src_speech_lengths: Union[torch.Tensor, np.ndarray],
        tgt_speech: Union[torch.Tensor, np.ndarray] = None,
        tgt_speech_lengths: Union[torch.Tensor, np.ndarray] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        sids: Union[torch.Tensor, np.ndarray] = None,
        lids: Union[torch.Tensor, np.ndarray] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run speech-to-speech."""
        assert check_argument_types()

        # check inputs
        if self.use_speech and tgt_speech is None:
            raise RuntimeError("Missing required argument: 'tgt_speech'")
        if self.use_sids and sids is None:
            raise RuntimeError("Missing required argument: 'sids'")
        if self.use_lids and lids is None:
            raise RuntimeError("Missing required argument: 'lids'")
        if self.use_spembs and spembs is None:
            raise RuntimeError("Missing required argument: 'spembs'")

        # prepare batch
        batch = dict(src_speech=src_speech, src_speech_lengths=src_speech_lengths)
        if tgt_speech is not None:
            batch.update(tgt_speech=tgt_speech)
            batch.update(tgt_speech_lengths=tgt_speech_lengths)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        batch = to_device(batch, self.device)

        # overwrite the decode configs if provided
        cfg = self.decode_conf
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)

        if self.s2st_type == "translatotron":
            output_dict = self.model.inference(**batch, **cfg)
            # apply vocoder (mel-to-wav)
            if self.vocoder is not None:
                if (
                    self.prefer_normalized_feats
                    or output_dict.get("feat_gen_denorm") is None
                ):
                    input_feat = output_dict["feat_gen"]
                else:
                    input_feat = output_dict["feat_gen_denorm"]
                wav = self.vocoder(input_feat)
                output_dict.update(wav=wav)
        elif self.s2st_type == "discrete_unit":
            output_dict = {}
            # Forward Encoder
            enc, _ = self.model.encode(batch["src_speech"], batch["src_speech_lengths"])
            nbest_hyps = self.beam_search(
                x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
            # TODO(jiatong): get nbest list instead of just best hyp
            best_hyp = nbest_hyps[0]  # just use the best
            # remove sos/eos and get results
            token_int = np.array(best_hyp.yseq[1:-1].tolist())
            output_dict.update(feat_gen=torch.tensor(token_int))

            logging.info("token_int: {}".format(token_int))

            if self.vocoder is not None:
                if len(token_int) == 0:
                    output_dict.update(wav=torch.tensor([0] * 100))
                else:
                    input_discrete_unit = to_device(
                        torch.tensor(token_int).view(-1, 1), device=self.device
                    )
                    # NOTE(jiatong): we default take the last token in the token list as <unk>
                    # see scripts/feats/performa_kemans.sh for details
                    input_discrete_unit = input_discrete_unit[
                        input_discrete_unit != self.model.unit_vocab_size - 1
                    ].view(-1, 1)
                    wav = self.vocoder(input_discrete_unit)
                    output_dict.update(wav=wav)

        elif self.s2st_type == "unity":
            output_dict = {}
            # Forward Encoder
            enc, _ = self.model.encode(batch["src_speech"], batch["src_speech_lengths"])

            st_subtask_nbest_hyps = self.st_subtask_beam_search(
                x=enc[0],
                maxlenratio=self.st_subtask_maxlenratio,
                minlenratio=self.st_subtask_minlenratio,
            )

            logging.info(
                "st_subtask_token_int: {}".format(
                    st_subtask_nbest_hyps[0].yseq[1:-1].tolist()
                )
            )

            st_subtask_result = []
            for hyp in st_subtask_nbest_hyps:
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                if isinstance(hyp.hs, List):
                    st_subtask_hs = torch.stack(hyp.hs)
                else:
                    st_subtask_hs = hyp.hs

                st_subtask_token_int = hyp.yseq[1:-1].tolist()
                st_subtask_token = self.st_subtask_converter.ids2tokens(
                    st_subtask_token_int
                )
                if self.st_subtask_tokenizer is not None:
                    st_subtask_hyp_text = self.st_subtask_tokenizer.tokens2text(
                        st_subtask_token
                    )
                else:
                    st_subtask_hyp_text = None

                st_subtask_result.append(
                    (
                        st_subtask_hyp_text,
                        st_subtask_token,
                        st_subtask_token_int,
                        st_subtask_hs,
                    )
                )

            if self.st_subtask_tokenizer is not None:
                (
                    st_subtask_hyp_text,
                    st_subtask_token,
                    st_subtask_token_int,
                    _,
                ) = st_subtask_result[0]
                logging.info("st_subtask_text: {}".format(st_subtask_result[0][0]))
                output_dict.update(st_subtask_text=st_subtask_hyp_text)
                output_dict.update(st_subtask_token=st_subtask_token)
                output_dict.update(st_subtask_token_int=st_subtask_token_int)

            # encoder 1best st_subtask result
            st_subtask_hs = st_subtask_result[0][-1].unsqueeze(0)
            st_subtask_hs = to_device(st_subtask_hs, device=self.device)
            st_subtask_hs_lengths = st_subtask_hs.new_full(
                [1], dtype=torch.long, fill_value=st_subtask_hs.size(1)
            )
            md_enc, _, _ = self.model.unit_encoder(st_subtask_hs, st_subtask_hs_lengths)
            nbest_hyps = self.beam_search(
                md_enc[0],
                maxlenratio=self.maxlenratio * 100,
                minlenratio=self.minlenratio,
            )

            # TODO(jiatong): get nbest list instead of just best hyp
            best_hyp = nbest_hyps[0]  # just use the best
            # remove sos/eos and get results
            token_int = np.array(best_hyp.yseq[1:-1].tolist())
            output_dict.update(feat_gen=torch.tensor(token_int))

            logging.info("token_int: {}".format(token_int))

            if self.vocoder is not None:
                if len(token_int) == 0:
                    output_dict.update(wav=torch.tensor([0] * 100))
                else:
                    input_discrete_unit = to_device(
                        torch.tensor(token_int).view(-1, 1), device=self.device
                    )
                    # NOTE(jiatong): we default take the last token in the token list as <unk>
                    # see scripts/feats/performa_kemans.sh for details
                    input_discrete_unit = input_discrete_unit[
                        input_discrete_unit != self.model.unit_vocab_size - 1
                    ].view(-1, 1)
                    wav = self.vocoder(input_discrete_unit)
                    output_dict.update(wav=wav)

        return output_dict

    @property
    def fs(self) -> Optional[int]:
        """Return sampling rate."""
        if hasattr(self.vocoder, "fs"):
            return self.vocoder.fs
        elif hasattr(self.model.synthesizer, "fs"):
            return self.model.synthesizer.fs
        else:
            return None

    @property
    def use_speech(self) -> bool:
        """Return speech is needed or not in the inference."""
        return self.use_teacher_forcing

    @property
    def use_sids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.model.synthesizer.spks is not None

    @property
    def use_lids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.model.synthesizer.langs is not None

    @property
    def use_spembs(self) -> bool:
        """Return spemb is needed or not in the inference."""
        return self.model.synthesizer.spk_embed_dim is not None

    @staticmethod
    def from_pretrained(
        vocoder_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Text2Speech instance from the pretrained model.

        Args:
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            Text2Speech: Text2Speech instance.

        """

        if vocoder_tag is not None:
            if vocoder_tag.startswith("parallel_wavegan/"):
                try:
                    from parallel_wavegan.utils import download_pretrained_model

                except ImportError:
                    logging.error(
                        "`parallel_wavegan` is not installed. "
                        "Please install via `pip install -U parallel_wavegan`."
                    )
                    raise

                from parallel_wavegan import __version__

                # NOTE(kan-bayashi): Filelock download is supported from 0.5.2
                assert V(__version__) > V("0.5.1"), (
                    "Please install the latest parallel_wavegan "
                    "via `pip install -U parallel_wavegan`."
                )
                vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
                vocoder_file = download_pretrained_model(vocoder_tag)
                vocoder_config = Path(vocoder_file).parent / "config.yml"
                kwargs.update(vocoder_config=vocoder_config, vocoder_file=vocoder_file)

            else:
                raise ValueError(f"{vocoder_tag} is unsupported format.")

        return Speech2Speech(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    threshold: float,
    minlenratio: float,
    maxlenratio: float,
    st_subtask_minlenratio: float,
    st_subtask_maxlenratio: float,
    use_teacher_forcing: bool,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    always_fix_seed: bool,
    nbest: int,
    beam_size: int,
    penalty: float,
    st_subtask_nbest: int,
    st_subtask_beam_size: int,
    st_subtask_penalty: float,
    st_subtask_token_type: Optional[str],
    st_subtask_bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    vocoder_config: Optional[str],
    vocoder_file: Optional[str],
    vocoder_tag: Optional[str],
):
    """Run text-to-speech inference."""
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
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

    # 2. Build model
    speech2speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        threshold=threshold,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        st_subtask_maxlenratio=st_subtask_maxlenratio,
        st_subtask_minlenratio=st_subtask_minlenratio,
        use_teacher_forcing=use_teacher_forcing,
        use_att_constraint=use_att_constraint,
        backward_window=backward_window,
        forward_window=forward_window,
        nbest=nbest,
        beam_size=beam_size,
        penalty=penalty,
        st_subtask_nbest=st_subtask_nbest,
        st_subtask_beam_size=st_subtask_beam_size,
        st_subtask_penalty=st_subtask_penalty,
        st_subtask_token_type=st_subtask_token_type,
        st_subtask_bpemodel=st_subtask_bpemodel,
        vocoder_config=vocoder_config,
        vocoder_file=vocoder_file,
        dtype=dtype,
        device=device,
        seed=seed,
        always_fix_seed=always_fix_seed,
    )
    speech2speech = Speech2Speech.from_pretrained(
        vocoder_tag=vocoder_tag,
        **speech2speech_kwargs,
    )

    # 3. Build data-iterator
    loader = S2STTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=S2STTask.build_preprocess_fn(speech2speech.train_args, False),
        collate_fn=S2STTask.build_collate_fn(speech2speech.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 6. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "speech_shape").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)
    (output_dir / "probs").mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_rates").mkdir(parents=True, exist_ok=True)

    # Lazy load to avoid the backend error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with NpyScpWriter(
        output_dir / "norm",
        output_dir / "norm/feats.scp",
    ) as norm_writer, NpyScpWriter(
        output_dir / "denorm", output_dir / "denorm/feats.scp"
    ) as denorm_writer, open(
        output_dir / "speech_shape/speech_shape", "w"
    ) as shape_writer, open(
        output_dir / "focus_rates/focus_rates", "w"
    ) as focus_rate_writer, DatadirWriter(
        output_dir / "st_subtask"
    ) as st_subtask_wrtier:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # # Change to single sequence and remove *_length
            # # because inference() requires 1-seq, not mini-batch.
            # batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

            start_time = time.perf_counter()
            output_dict = speech2speech(**batch)

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            # standard speech2mel model case
            feat_gen = output_dict["feat_gen"]
            logging.info(
                "inference speed = {:.1f} frames / sec.".format(
                    int(feat_gen.size(0)) / (time.perf_counter() - start_time)
                )
            )
            logging.info(f"{key} (size:{insize}->{feat_gen.size(0)})")
            if feat_gen.size(0) == insize * maxlenratio:
                logging.warning(f"output length reaches maximum length ({key}).")

            norm_writer[key] = output_dict["feat_gen"].cpu().numpy()
            shape_writer.write(
                f"{key} " + ",".join(map(str, output_dict["feat_gen"].shape)) + "\n"
            )
            if output_dict.get("feat_gen_denorm") is not None:
                denorm_writer[key] = output_dict["feat_gen_denorm"].cpu().numpy()

            if output_dict.get("focus_rate") is not None:
                focus_rate_writer.write(
                    f"{key} {float(output_dict['focus_rate']):.5f}\n"
                )

            if output_dict.get("att_w") is not None:
                # Plot attention weight
                att_w = output_dict["att_w"].cpu().numpy()

                if att_w.ndim == 3:
                    logging.warning(
                        "Cannot plot attn due to dim mismatch (for multihead)"
                    )
                    output_dict["att_w"] = None
                else:
                    if att_w.ndim == 2:
                        att_w = att_w[None][None]
                    elif att_w.ndim != 4:
                        raise RuntimeError(f"Must be 2 or 4 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(att_w.shape[0] / att_w.shape[1])
                    fig = plt.Figure(
                        figsize=(
                            w * 1.3 * min(att_w.shape[0], 2.5),
                            h * 1.3 * min(att_w.shape[1], 2.5),
                        )
                    )
                    fig.suptitle(f"{key}")
                    axes = fig.subplots(att_w.shape[0], att_w.shape[1])
                    if len(att_w) == 1:
                        axes = [[axes]]
                    for ax, att_w in zip(axes, att_w):
                        for ax_, att_w_ in zip(ax, att_w):
                            ax_.imshow(att_w_.astype(np.float32), aspect="auto")
                            ax_.set_xlabel("Input")
                            ax_.set_ylabel("Output")
                            ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax_.yaxis.set_major_locator(MaxNLocator(integer=True))

                    fig.set_tight_layout({"rect": [0, 0.03, 1, 0.95]})
                    fig.savefig(output_dir / f"att_ws/{key}.png")
                    fig.clf()

            if output_dict.get("prob") is not None:
                # Plot stop token prediction
                prob = output_dict["prob"].cpu().numpy()

                fig = plt.Figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(prob)
                ax.set_title(f"{key}")
                ax.set_xlabel("Output")
                ax.set_ylabel("Stop probability")
                ax.set_ylim(0, 1)
                ax.grid(which="both")

                fig.set_tight_layout(True)
                fig.savefig(output_dir / f"probs/{key}.png")
                fig.clf()

            if output_dict.get("wav") is not None:
                # TODO(kamo): Write scp
                logging.info("wav {}".format(output_dict["wav"].size()))
                sf.write(
                    f"{output_dir}/wav/{key}.wav",
                    output_dict["wav"].cpu().numpy(),
                    speech2speech.fs,
                    "PCM_16",
                )

            if output_dict.get("st_subtask_token") is not None:
                writer["token"][key] = " ".join(output_dict["st_subtask_token"])
                writer["token_int"][key] == " ".join(
                    map(str, output_dict["st_subtask_token_int"])
                )
                if output_dict.get("st_subtask_text") is not None:
                    writer["text"][key] = output_dict["st_subtask_text"]

    # remove files if those are not included in output dict
    if output_dict.get("feat_gen") is None:
        shutil.rmtree(output_dir / "norm")
    if output_dict.get("feat_gen_denorm") is None:
        shutil.rmtree(output_dir / "denorm")
    if output_dict.get("att_w") is None:
        shutil.rmtree(output_dir / "att_ws")
    if output_dict.get("focus_rate") is None:
        shutil.rmtree(output_dir / "focus_rates")
    if output_dict.get("prob") is None:
        shutil.rmtree(output_dir / "probs")
    if output_dict.get("wav") is None:
        shutil.rmtree(output_dir / "wav")
    if output_dict.get("st_subtask_token") is not None:
        shutil.rmtree(output_dict / "st_subtask")


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="S2ST inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
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
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--st_subtask_maxlenratio",
        type=float,
        default=1.5,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--st_subtask_minlenratio",
        type=float,
        default=0.1,
        help="Minimum length ratio in decoding",
    )

    group = parser.add_argument_group("Spectrogram-based generation related")
    group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value in decoding",
    )
    group.add_argument(
        "--use_att_constraint",
        type=str2bool,
        default=False,
        help="Whether to use attention constraint",
    )
    group.add_argument(
        "--backward_window",
        type=int,
        default=1,
        help="Backward window value in attention constraint",
    )
    group.add_argument(
        "--forward_window",
        type=int,
        default=3,
        help="Forward window value in attention constraint",
    )
    group.add_argument(
        "--use_teacher_forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
    )

    group = parser.add_argument_group("Beam-search (discrete unit/multi-pass) related")
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--st_subtask_nbest",
        type=int,
        default=1,
        help="Output N-best hypotheses for st subtask",
    )
    group.add_argument(
        "--st_subtask_beam_size", type=int, default=5, help="Beam size for st subtask"
    )
    group.add_argument(
        "--st_subtask_penalty",
        type=float,
        default=0.0,
        help="Insertion penalty for st subtask",
    )

    group = parser.add_argument_group("Vocoder related")
    group.add_argument(
        "--vocoder_config",
        type=str_or_none,
        help="Vocoder configuration file",
    )
    group.add_argument(
        "--vocoder_file",
        type=str_or_none,
        help="Vocoder parameter file",
    )
    group.add_argument(
        "--vocoder_tag",
        type=str,
        help="Pretrained vocoder tag. If specify this option, vocoder_config and "
        "vocoder_file will be overwritten",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--st_subtask_token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ST model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--st_subtask_bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    return parser


def main(cmd=None):
    """Run S2ST model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
