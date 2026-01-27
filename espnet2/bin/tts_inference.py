#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""

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
from typeguard import typechecked

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_tts.vits import VITS
from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.tts import TTSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.tts.utils import DurationCalculator
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none


class Text2Speech:
    """Text2Speech class.

    Examples:
        >>> from espnet2.bin.tts_inference import Text2Speech
        >>> # Case 1: Load the local model and use Griffin-Lim vocoder
        >>> text2speech = Text2Speech(
        >>>     train_config="/path/to/config.yml",
        >>>     model_file="/path/to/model.pth",
        >>> )
        >>> # Case 2: Load the local model and the pretrained vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     train_config="/path/to/config.yml",
        >>>     model_file="/path/to/model.pth",
        >>>     vocoder_tag="kan-bayashi/ljspeech_tacotron2",
        >>> )
        >>> # Case 3: Load the pretrained model and use Griffin-Lim vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     model_tag="kan-bayashi/ljspeech_tacotron2",
        >>> )
        >>> # Case 4: Load the pretrained model and the pretrained vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     model_tag="kan-bayashi/ljspeech_tacotron2",
        >>>     vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v1",
        >>> )
        >>> # Run inference and save as wav file
        >>> import soundfile as sf
        >>> wav = text2speech("Hello, World")["wav"]
        >>> sf.write("out.wav", wav.numpy(), text2speech.fs, "PCM_16")

    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        speed_control_alpha: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        vocoder_config: Union[Path, str, None] = None,
        vocoder_file: Union[Path, str, None] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        prefer_normalized_feats: bool = False,
    ):
        """Initialize Text2Speech module."""

        # setup model
        model, train_args = TTSTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.tts = model.tts
        self.normalize = model.normalize
        self.feats_extract = model.feats_extract
        self.duration_calculator = DurationCalculator()
        self.preprocess_fn = TTSTask.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        self.vocoder = None
        self.prefer_normalized_feats = prefer_normalized_feats
        if self.tts.require_vocoder:
            vocoder = TTSTask.build_vocoder_from_file(
                vocoder_config, vocoder_file, model, device
            )
            if isinstance(vocoder, torch.nn.Module):
                vocoder.to(dtype=getattr(torch, dtype)).eval()
            self.vocoder = vocoder
        logging.info(f"Extractor:\n{self.feats_extract}")
        logging.info(f"Normalizer:\n{self.normalize}")
        logging.info(f"TTS:\n{self.tts}")
        if self.vocoder is not None:
            logging.info(f"Vocoder:\n{self.vocoder}")

        # setup decoding config
        decode_conf = {}
        decode_conf.update(use_teacher_forcing=use_teacher_forcing)
        if isinstance(self.tts, (Tacotron2, Transformer)):
            decode_conf.update(
                threshold=threshold,
                maxlenratio=maxlenratio,
                minlenratio=minlenratio,
            )
        if isinstance(self.tts, Tacotron2):
            decode_conf.update(
                use_att_constraint=use_att_constraint,
                forward_window=forward_window,
                backward_window=backward_window,
            )
        if isinstance(self.tts, (FastSpeech, FastSpeech2, VITS)):
            decode_conf.update(alpha=speed_control_alpha)
        if isinstance(self.tts, VITS):
            decode_conf.update(
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
            )
        self.decode_conf = decode_conf

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        text: Union[str, torch.Tensor, np.ndarray],
        speech: Union[torch.Tensor, np.ndarray, None] = None,
        durations: Union[torch.Tensor, np.ndarray, None] = None,
        spembs: Union[torch.Tensor, np.ndarray, None] = None,
        sids: Union[torch.Tensor, np.ndarray, None] = None,
        lids: Union[torch.Tensor, np.ndarray, None] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run text-to-speech."""

        # check inputs
        if self.use_speech and speech is None:
            raise RuntimeError("Missing required argument: 'speech'")
        if self.use_sids and sids is None:
            raise RuntimeError("Missing required argument: 'sids'")
        if self.use_lids and lids is None:
            raise RuntimeError("Missing required argument: 'lids'")
        if self.use_spembs and spembs is None:
            raise RuntimeError("Missing required argument: 'spembs'")

        # prepare batch
        if isinstance(text, str):
            text = self.preprocess_fn("<dummy>", dict(text=text))["text"]
        batch = dict(text=text)
        if speech is not None:
            batch.update(speech=speech)
        if durations is not None:
            batch.update(durations=durations)
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
        output_dict = self.model.inference(**batch, **cfg)

        # calculate additional metrics
        if output_dict.get("att_w") is not None:
            duration, focus_rate = self.duration_calculator(output_dict["att_w"])
            output_dict.update(duration=duration, focus_rate=focus_rate)

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

        return output_dict

    @torch.no_grad()
    def batch_call(
        self,
        text: Union[List[str], torch.Tensor, np.ndarray],
        speech: Union[torch.Tensor, np.ndarray, None] = None,
        speech_lengths: Optional[Union[torch.Tensor, np.ndarray]] = None,
        durations: Union[torch.Tensor, np.ndarray, None] = None,
        durations_lengths: Optional[Union[torch.Tensor, np.ndarray]] = None,
        spembs: Union[torch.Tensor, np.ndarray, None] = None,
        sids: Union[torch.Tensor, np.ndarray, None] = None,
        lids: Union[torch.Tensor, np.ndarray, None] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run batch text-to-speech for FastSpeech/FastSpeech2 models.

        This method enables batch inference for non-autoregressive TTS models.

        Args:
            text: List of input text strings (will be converted to list of tensors).
            speech: Batched speech tensor for teacher forcing (B, T_wav).
            speech_lengths: Length tensor for batched speech (B,).
            durations: Batched duration tensor (B, T_text).
            durations_lengths: Length tensor for batched durations (B,).
            spembs: Batched speaker embedding tensor (B, D).
            sids: Batched speaker ID tensor (B,).
            lids: Batched language ID tensor (B,).
            decode_conf: Optional decoding configuration overrides.

        Returns:
            Dict containing:
                - feat_gen: List of generated feature tensors
                - feat_gen_lengths: Tensor of output lengths (B,)
                - duration: Batched duration tensor (B, T_text)
                - wav: List of generated waveforms (if vocoder is available)
                And other model-specific outputs.

        """
        # Check if batch inference is supported
        if not hasattr(self.model, "batch_inference"):
            raise NotImplementedError(
                "Batch inference is only supported FastSpeech2 models"
            )

        # check inputs
        if self.use_speech and speech is None:
            raise RuntimeError("Missing required argument: 'speech'")
        if self.use_sids and sids is None:
            raise RuntimeError("Missing required argument: 'sids'")
        if self.use_lids and lids is None:
            raise RuntimeError("Missing required argument: 'lids'")
        if self.use_spembs and spembs is None:
            raise RuntimeError("Missing required argument: 'spembs'")

        # prepare batch - handle list of strings
        if isinstance(text, list):
            if all(isinstance(t, str) for t in text):
                # Convert list of strings to list of tensors (no padding here)
                text = [
                    torch.tensor(
                        self.preprocess_fn("<dummy>", dict(text=t))["text"],
                        dtype=torch.long,
                    )
                    for t in text
                ]
                # Pass as list of tensors - model will handle padding and EOS
            elif all(isinstance(t, np.ndarray) for t in text):
                text = [torch.from_numpy(t) for t in text]
            elif all(isinstance(t, torch.Tensor) for t in text):
                text = text
            else:
                raise ValueError(
                    "batch_call expects text as a list of strings or tensors."
                )
        else:
            raise ValueError("batch_call expects text as a list of strings or tensors.")

        batch = dict(text=text)
        if speech is not None:
            if isinstance(speech, np.ndarray):
                speech = torch.from_numpy(speech)
            batch.update(speech=speech)
            if speech_lengths is not None:
                if isinstance(speech_lengths, np.ndarray):
                    speech_lengths = torch.from_numpy(speech_lengths)
                batch.update(speech_lengths=speech_lengths)
        if durations is not None:
            if isinstance(durations, np.ndarray):
                durations = torch.from_numpy(durations)
            batch.update(durations=durations)
            if durations_lengths is not None:
                if isinstance(durations_lengths, np.ndarray):
                    durations_lengths = torch.from_numpy(durations_lengths)
                batch.update(durations_lengths=durations_lengths)
        if spembs is not None:
            if isinstance(spembs, np.ndarray):
                spembs = torch.from_numpy(spembs)
            batch.update(spembs=spembs)
        if sids is not None:
            if isinstance(sids, np.ndarray):
                sids = torch.from_numpy(sids)
            batch.update(sids=sids)
        if lids is not None:
            if isinstance(lids, np.ndarray):
                lids = torch.from_numpy(lids)
            batch.update(lids=lids)
        batch = to_device(batch, self.device)

        # overwrite the decode configs if provided
        cfg = self.decode_conf.copy()
        if decode_conf is not None:
            cfg.update(decode_conf)

        # batch inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)
        output_dict = self.model.batch_inference(**batch, **cfg)

        # apply vocoder (mel-to-wav) to each sample
        if self.vocoder is not None:
            feat_gen_list = output_dict.get("feat_gen")
            feat_gen_denorm_list = output_dict.get("feat_gen_denorm")

            if feat_gen_list is not None:
                wav_list = []
                for i, feat_gen in enumerate(feat_gen_list):
                    if self.prefer_normalized_feats or feat_gen_denorm_list is None:
                        input_feat = feat_gen
                    else:
                        input_feat = feat_gen_denorm_list[i]
                    wav = self.vocoder(input_feat)
                    wav_list.append(wav)
                output_dict.update(wav=wav_list)

        return output_dict

    @property
    def fs(self) -> Optional[int]:
        """Return sampling rate."""
        if hasattr(self.vocoder, "fs"):
            return self.vocoder.fs
        elif hasattr(self.tts, "fs"):
            return self.tts.fs
        else:
            return None

    @property
    def use_speech(self) -> bool:
        """Return speech is needed or not in the inference."""
        return self.use_teacher_forcing or getattr(self.tts, "use_gst", False)

    @property
    def use_sids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.tts.spks is not None

    @property
    def use_lids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.tts.langs is not None

    @property
    def use_spembs(self) -> bool:
        """Return spemb is needed or not in the inference."""
        return self.tts.spk_embed_dim is not None

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        vocoder_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Text2Speech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            Text2Speech: Text2Speech instance.

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

        return Text2Speech(**kwargs)


@typechecked
def inference(
    output_dir: Union[Path, str],
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
    model_tag: Optional[str],
    threshold: float,
    minlenratio: float,
    maxlenratio: float,
    use_teacher_forcing: bool,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    speed_control_alpha: float,
    noise_scale: float,
    noise_scale_dur: float,
    always_fix_seed: bool,
    allow_variable_data_keys: bool,
    vocoder_config: Optional[str],
    vocoder_file: Optional[str],
    vocoder_tag: Optional[str],
):
    """Run text-to-speech inference."""
    # Check if batch inference is supported based on model type
    # (will be checked later when model is loaded)
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
    text2speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        threshold=threshold,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        use_teacher_forcing=use_teacher_forcing,
        use_att_constraint=use_att_constraint,
        backward_window=backward_window,
        forward_window=forward_window,
        speed_control_alpha=speed_control_alpha,
        noise_scale=noise_scale,
        noise_scale_dur=noise_scale_dur,
        vocoder_config=vocoder_config,
        vocoder_file=vocoder_file,
        dtype=dtype,
        device=device,
        seed=seed,
        always_fix_seed=always_fix_seed,
    )
    text2speech = Text2Speech.from_pretrained(
        model_tag=model_tag,
        vocoder_tag=vocoder_tag,
        **text2speech_kwargs,
    )

    # 3. Build data-iterator
    if not text2speech.use_speech:
        data_path_and_name_and_type = list(
            filter(lambda x: x[1] != "speech", data_path_and_name_and_type)
        )
    loader = TTSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=TTSTask.build_preprocess_fn(text2speech.train_args, False),
        collate_fn=TTSTask.build_collate_fn(text2speech.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "speech_shape").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)
    (output_dir / "probs").mkdir(parents=True, exist_ok=True)
    (output_dir / "durations").mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_rates").mkdir(parents=True, exist_ok=True)

    # Lazy load to avoid the backend error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Check if model supports batch inference
    supports_batch = hasattr(text2speech.model, "batch_inference") and hasattr(
        text2speech.tts, "batch_inference"
    )
    if batch_size > 1 and not supports_batch:
        raise NotImplementedError(
            f"batch decoding is not supported for {type(text2speech.tts).__name__}. "
            "Only FastSpeech and FastSpeech2 support batch inference."
        )

    with (
        NpyScpWriter(
            output_dir / "norm",
            output_dir / "norm/feats.scp",
        ) as norm_writer,
        NpyScpWriter(
            output_dir / "denorm", output_dir / "denorm/feats.scp"
        ) as denorm_writer,
        open(output_dir / "speech_shape/speech_shape", "w") as shape_writer,
        open(output_dir / "durations/durations", "w") as duration_writer,
        open(output_dir / "focus_rates/focus_rates", "w") as focus_rate_writer,
    ):
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))

            start_time = time.perf_counter()

            if batch_size > 1 and _bs > 1 and supports_batch:
                # Batch inference path for FastSpeech/FastSpeech2
                output_dict = text2speech.batch_call(**batch)

                # Process each sample in the batch
                feat_gen_list = output_dict.get("feat_gen")
                feat_gen_denorm_list = output_dict.get("feat_gen_denorm")
                wav_list = output_dict.get("wav")
                duration = output_dict.get("duration")

                total_frames = 0
                for i, key in enumerate(keys):
                    if feat_gen_list is not None:
                        feat_gen = feat_gen_list[i]
                        total_frames += feat_gen.size(0)
                        insize = batch["text"][i].size(0) + 1

                        logging.info(f"{key} (size:{insize}->{feat_gen.size(0)})")
                        if feat_gen.size(0) == insize * maxlenratio:
                            logging.warning(
                                f"output length reaches maximum length ({key})."
                            )

                        norm_writer[key] = feat_gen.cpu().numpy()
                        shape_writer.write(
                            f"{key} " + ",".join(map(str, feat_gen.shape)) + "\n"
                        )
                        if feat_gen_denorm_list is not None:
                            denorm_writer[key] = feat_gen_denorm_list[i].cpu().numpy()
                    elif wav_list is not None:
                        wav = wav_list[i]
                        total_frames += wav.size(0)
                        insize = batch["text"][i].size(0) + 1
                        logging.info(f"{key} (size:{insize}->{wav.size(0)})")

                    if duration is not None:
                        duration_writer.write(
                            f"{key} "
                            + " ".join(map(str, duration[i].long().cpu().numpy()))
                            + "\n"
                        )

                    if wav_list is not None:
                        sf.write(
                            f"{output_dir}/wav/{key}.wav",
                            wav_list[i].cpu().numpy(),
                            text2speech.fs,
                            "PCM_16",
                        )

                logging.info(
                    "batch inference speed = {:.1f} frames / sec.".format(
                        total_frames / (time.perf_counter() - start_time)
                    )
                )
            else:
                # Single sample inference path (original behavior)
                assert _bs == 1, f"Expected batch size 1, got {_bs}"

                # Change to single sequence and remove *_length
                # because inference() requires 1-seq, not mini-batch.
                batch_single = {
                    k: v[0] for k, v in batch.items() if not k.endswith("_lengths")
                }

                output_dict = text2speech(**batch_single)

                key = keys[0]
                insize = next(iter(batch_single.values())).size(0) + 1
                if output_dict.get("feat_gen") is not None:
                    # standard text2mel model case
                    feat_gen = output_dict["feat_gen"]
                    logging.info(
                        "inference speed = {:.1f} frames / sec.".format(
                            int(feat_gen.size(0)) / (time.perf_counter() - start_time)
                        )
                    )
                    logging.info(f"{key} (size:{insize}->{feat_gen.size(0)})")
                    if feat_gen.size(0) == insize * maxlenratio:
                        logging.warning(
                            f"output length reaches maximum length ({key})."
                        )

                    norm_writer[key] = output_dict["feat_gen"].cpu().numpy()
                    shape_writer.write(
                        f"{key} "
                        + ",".join(map(str, output_dict["feat_gen"].shape))
                        + "\n"
                    )
                    if output_dict.get("feat_gen_denorm") is not None:
                        denorm_writer[key] = (
                            output_dict["feat_gen_denorm"].cpu().numpy()
                        )
                else:
                    # end-to-end text2wav model case
                    wav = output_dict["wav"]
                    logging.info(
                        "inference speed = {:.1f} points / sec.".format(
                            int(wav.size(0)) / (time.perf_counter() - start_time)
                        )
                    )
                    logging.info(f"{key} (size:{insize}->{wav.size(0)})")

                if output_dict.get("duration") is not None:
                    # Save duration and fucus rates
                    duration_writer.write(
                        f"{key} "
                        + " ".join(
                            map(str, output_dict["duration"].long().cpu().numpy())
                        )
                        + "\n"
                    )

                if output_dict.get("focus_rate") is not None:
                    focus_rate_writer.write(
                        f"{key} {float(output_dict['focus_rate']):.5f}\n"
                    )

                if output_dict.get("att_w") is not None:
                    # Plot attention weight
                    att_w = output_dict["att_w"].cpu().numpy()

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
                    sf.write(
                        f"{output_dir}/wav/{key}.wav",
                        output_dict["wav"].cpu().numpy(),
                        text2speech.fs,
                        "PCM_16",
                    )

    # remove files if those are not included in output dict
    if output_dict.get("feat_gen") is None:
        shutil.rmtree(output_dir / "norm")
    if output_dict.get("feat_gen_denorm") is None:
        shutil.rmtree(output_dir / "denorm")
    if output_dict.get("att_w") is None:
        shutil.rmtree(output_dir / "att_ws")
    if output_dict.get("duration") is None:
        shutil.rmtree(output_dir / "durations")
    if output_dict.get("focus_rate") is None:
        shutil.rmtree(output_dir / "focus_rates")
    if output_dict.get("prob") is None:
        shutil.rmtree(output_dir / "probs")
    if output_dict.get("wav") is None:
        shutil.rmtree(output_dir / "wav")


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="TTS inference",
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
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
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
    parser.add_argument(
        "--speed_control_alpha",
        type=float,
        default=1.0,
        help="Alpha in FastSpeech to change the speed of generated speech",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.667,
        help="Noise scale parameter for the flow in vits",
    )
    parser.add_argument(
        "--noise_scale_dur",
        type=float,
        default=0.8,
        help="Noise scale parameter for the stochastic duration predictor in vits",
    )
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
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
    return parser


def main(cmd=None):
    """Run TTS model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
