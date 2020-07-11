#!/usr/bin/env python3

"""TTS mode decoding."""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Dict
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
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.duration_calculator import DurationCalculator
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def check_use_speech_in_inference(tts: AbsTTS, decode_config: Dict) -> bool:
    """Check whether to require speech in inference.

    Args:
        tts (AbsTTS): TTS model instance.
        decode_config (Dict): Decoding config dictionary.

    Returns:
        bool: True if speech is required else False.

    """
    inferece_options_required_speech = ["use_teacher_forcing"]
    tts_options_required_speech = ["use_gst"]
    for option in inferece_options_required_speech:
        if decode_config.get(option, False):
            return True
    for option in tts_options_required_speech:
        if getattr(tts, option, False):
            return True
    return False


@torch.no_grad()
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
    model, train_args = TTSTask.build_model_from_file(train_config, model_file, device)
    model.to(dtype=getattr(torch, dtype)).eval()
    tts = model.tts
    normalize = model.normalize
    feats_extract = model.feats_extract
    duration_calculator = DurationCalculator()
    logging.info(f"Normalization:\n{normalize}")
    logging.info(f"TTS:\n{tts}")

    # 3. Build decoding config
    decode_config = {}
    if isinstance(tts, (Tacotron2, Transformer)):
        decode_config.update(
            {
                "threshold": threshold,
                "maxlenratio": maxlenratio,
                "minlenratio": minlenratio,
                "use_teacher_forcing": use_teacher_forcing,
            }
        )
    if isinstance(tts, Tacotron2):
        decode_config.update(
            {
                "use_att_constraint": use_att_constraint,
                "forward_window": forward_window,
                "backward_window": backward_window,
            }
        )
    if isinstance(tts, FastSpeech):
        decode_config.update({"alpha": speed_control_alpha})
    use_speech = check_use_speech_in_inference(tts, decode_config)

    # 4. Build data-iterator
    if not use_speech:
        data_path_and_name_and_type = list(
            filter(lambda x: x[1] != "speech", data_path_and_name_and_type)
        )
    loader = TTSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=TTSTask.build_preprocess_fn(train_args, False),
        collate_fn=TTSTask.build_collate_fn(train_args),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 5. Build converter from spectrogram to waveform
    if feats_extract is not None:
        vocoder_conf.update(feats_extract.get_parameters())
    if "n_fft" in vocoder_conf and "n_shift" in vocoder_conf and "fs" in vocoder_conf:
        spc2wav = Spectrogram2Waveform(**vocoder_conf)
        logging.info(f"Vocoder: {spc2wav}")
    else:
        spc2wav = None
        logging.info("Vocoder is not used because vocoder_conf is not sufficient")

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
        output_dir / "norm", output_dir / "norm/feats.scp",
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
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = to_device(batch, device)

            # Extract features if speech is needed
            if use_speech:
                if feats_extract is not None:
                    _speech = (v for k, v in batch.items() if k.startswith("speech"))
                    speech, speech_lengths = normalize(*feats_extract(*_speech))
                else:
                    speech, speech_lengths = normalize(*_speech)
                batch.update(speech=speech, speech_lengths=speech_lengths)

            key = keys[0]
            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            _data = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            start_time = time.perf_counter()
            outs, probs, att_ws = tts.inference(**_data, **decode_config)
            insize = next(iter(_data.values())).size(0) + 1
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

            # NOTE: normalize.inverse is in-place operation
            outs_denorm = normalize.inverse(outs[None])[0][0]
            denorm_writer[key] = outs_denorm.cpu().numpy()

            if att_ws is not None:
                # Save duration and fucus rates
                duration, focus_rate = duration_calculator(att_ws)
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
            if spc2wav is not None:
                wav = spc2wav(outs_denorm.cpu().numpy())
                sf.write(f"{output_dir}/wav/{key}.wav", wav, spc2wav.fs, "PCM_16")

    # remove duration related files if attention is not provided
    if att_ws is None:
        shutil.rmtree(output_dir / "att_ws")
        shutil.rmtree(output_dir / "probs")
        shutil.rmtree(output_dir / "durations")
        shutil.rmtree(output_dir / "focus_rates")


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
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed",
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
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file", type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys", type=str2bool, default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config", type=str, help="Training configuration file.",
    )
    group.add_argument(
        "--model_file", type=str, help="Model parameter file.",
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
        "--threshold", type=float, default=0.5, help="Threshold value in decoding",
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
