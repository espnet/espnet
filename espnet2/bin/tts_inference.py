#!/usr/bin/env python3

"""TTS mode decoding."""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.tts import TTSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tts.duration_calculator import DurationCalculator
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class Text2Speech:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> text2speech = Text2Speech("config.yml", "model.pth")
        >>> wav = text2speech("Hello World")[0]
        >>> soundfile.write("out.wav", wav.numpy(), text2speech.fs, "PCM_16")

    """

    def __init__(
        self,
        train_config: Optional[Union[Path, str]],
        model_file: Optional[Union[Path, str]] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        speed_control_alpha: float = 1.0,
        vocoder_conf: dict = None,
        dtype: str = "float32",
        device: str = "cpu",
    ):
        assert check_argument_types()

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

        logging.info(f"Normalization:\n{self.normalize}")
        logging.info(f"TTS:\n{self.tts}")

        decode_config = {}
        if isinstance(self.tts, (Tacotron2, Transformer)):
            decode_config.update(
                {
                    "threshold": threshold,
                    "maxlenratio": maxlenratio,
                    "minlenratio": minlenratio,
                }
            )
        if isinstance(self.tts, Tacotron2):
            decode_config.update(
                {
                    "use_att_constraint": use_att_constraint,
                    "forward_window": forward_window,
                    "backward_window": backward_window,
                }
            )
        if isinstance(self.tts, (FastSpeech, FastSpeech2)):
            decode_config.update({"alpha": speed_control_alpha})
        decode_config.update({"use_teacher_forcing": use_teacher_forcing})

        self.decode_config = decode_config

        if vocoder_conf is None:
            vocoder_conf = {}
        if self.feats_extract is not None:
            vocoder_conf.update(self.feats_extract.get_parameters())
        if (
            "n_fft" in vocoder_conf
            and "n_shift" in vocoder_conf
            and "fs" in vocoder_conf
        ):
            self.spc2wav = Spectrogram2Waveform(**vocoder_conf)
            logging.info(f"Vocoder: {self.spc2wav}")
        else:
            self.spc2wav = None
            logging.info("Vocoder is not used because vocoder_conf is not sufficient")

    @torch.no_grad()
    def __call__(
        self,
        text: Union[str, torch.Tensor, np.ndarray],
        speech: Union[torch.Tensor, np.ndarray] = None,
        durations: Union[torch.Tensor, np.ndarray] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        speed_control_alpha: Optional[float] = None,
    ):
        assert check_argument_types()

        if self.use_speech and speech is None:
            raise RuntimeError("missing required argument: 'speech'")

        if isinstance(text, str):
            # str -> np.ndarray
            text = self.preprocess_fn("<dummy>", {"text": text})["text"]
        batch = {"text": text}
        if speech is not None:
            batch["speech"] = speech
        if durations is not None:
            batch["durations"] = durations
        if spembs is not None:
            batch["spembs"] = spembs

        cfg = self.decode_config
        if speed_control_alpha is not None and isinstance(
            self.tts, (FastSpeech, FastSpeech2)
        ):
            cfg = self.decode_config.copy()
            cfg.update({"alpha": speed_control_alpha})

        batch = to_device(batch, self.device)
        outs, outs_denorm, probs, att_ws = self.model.inference(**batch, **cfg)

        if att_ws is not None:
            duration, focus_rate = self.duration_calculator(att_ws)
        else:
            duration, focus_rate = None, None

        if self.spc2wav is not None:
            wav = torch.tensor(self.spc2wav(outs_denorm.cpu().numpy()))
        else:
            wav = None

        return wav, outs, outs_denorm, probs, att_ws, duration, focus_rate

    @property
    def fs(self) -> Optional[int]:
        if self.spc2wav is not None:
            return self.spc2wav.fs
        else:
            return None

    @property
    def use_speech(self) -> bool:
        """Check whether to require speech in inference.

        Returns:
            bool: True if speech is required else False.

        """
        return self.use_teacher_forcing or getattr(self.tts, "use_gst", False)


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
    use_teacher_forcing: bool,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    speed_control_alpha: float,
    allow_variable_data_keys: bool,
    vocoder_conf: dict,
):
    """Perform TTS model decoding."""
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
    text2speech = Text2Speech(
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
        vocoder_conf=vocoder_conf,
        dtype=dtype,
        device=device,
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

    # 6. Start for-loop
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
        output_dir / "durations/durations", "w"
    ) as duration_writer, open(
        output_dir / "focus_rates/focus_rates", "w"
    ) as focus_rate_writer:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            start_time = time.perf_counter()
            wav, outs, outs_denorm, probs, att_ws, duration, focus_rate = text2speech(
                **batch
            )

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            logging.info(
                "inference speed = {:.1f} frames / sec.".format(
                    int(outs.size(0)) / (time.perf_counter() - start_time)
                )
            )
            logging.info(f"{key} (size:{insize}->{outs.size(0)})")
            if outs.size(0) == insize * maxlenratio:
                logging.warning(f"output length reaches maximum length ({key}).")

            norm_writer[key] = outs.cpu().numpy()
            shape_writer.write(f"{key} " + ",".join(map(str, outs.shape)) + "\n")

            denorm_writer[key] = outs_denorm.cpu().numpy()

            if duration is not None:
                # Save duration and fucus rates
                duration_writer.write(
                    f"{key} " + " ".join(map(str, duration.cpu().numpy())) + "\n"
                )
                focus_rate_writer.write(f"{key} {float(focus_rate):.5f}\n")

                # Plot attention weight
                att_ws = att_ws.cpu().numpy()

                if att_ws.ndim == 2:
                    att_ws = att_ws[None][None]
                elif att_ws.ndim != 4:
                    raise RuntimeError(f"Must be 2 or 4 dimension: {att_ws.ndim}")

                w, h = plt.figaspect(att_ws.shape[0] / att_ws.shape[1])
                fig = plt.Figure(
                    figsize=(
                        w * 1.3 * min(att_ws.shape[0], 2.5),
                        h * 1.3 * min(att_ws.shape[1], 2.5),
                    )
                )
                fig.suptitle(f"{key}")
                axes = fig.subplots(att_ws.shape[0], att_ws.shape[1])
                if len(att_ws) == 1:
                    axes = [[axes]]
                for ax, att_w in zip(axes, att_ws):
                    for ax_, att_w_ in zip(ax, att_w):
                        ax_.imshow(att_w_.astype(np.float32), aspect="auto")
                        ax_.set_xlabel("Input")
                        ax_.set_ylabel("Output")
                        ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax_.yaxis.set_major_locator(MaxNLocator(integer=True))

                fig.set_tight_layout({"rect": [0, 0.03, 1, 0.95]})
                fig.savefig(output_dir / f"att_ws/{key}.png")
                fig.clf()

            if probs is not None:
                # Plot stop token prediction
                probs = probs.cpu().numpy()

                fig = plt.Figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(probs)
                ax.set_title(f"{key}")
                ax.set_xlabel("Output")
                ax.set_ylabel("Stop probability")
                ax.set_ylim(0, 1)
                ax.grid(which="both")

                fig.set_tight_layout(True)
                fig.savefig(output_dir / f"probs/{key}.png")
                fig.clf()

            # TODO(kamo): Write scp
            if wav is not None:
                sf.write(
                    f"{output_dir}/wav/{key}.wav", wav.numpy(), text2speech.fs, "PCM_16"
                )

    # remove duration related files if attention is not provided
    if att_ws is None:
        shutil.rmtree(output_dir / "att_ws")
        shutil.rmtree(output_dir / "durations")
        shutil.rmtree(output_dir / "focus_rates")
    if probs is None:
        shutil.rmtree(output_dir / "probs")


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="TTS Decode",
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
        help="Training configuration file.",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file.",
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

    group = parser.add_argument_group("Grriffin-Lim related")
    group.add_argument(
        "--vocoder_conf",
        action=NestedDictAction,
        default=get_default_kwargs(Spectrogram2Waveform),
        help="The configuration for Grriffin-Lim",
    )
    return parser


def main(cmd=None):
    """Run TTS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
