#!/usr/bin/env python3

"""TTS mode decoding."""

import logging
from pathlib import Path
import sys
import time
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
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
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


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
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
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
    logging.info(f"Normalization:\n{normalize}")
    logging.info(f"TTS:\n{tts}")

    # 3. Build data-iterator
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

    # 4. Build converter from spectrogram to waveform
    if model.feats_extract is not None:
        vocoder_conf.update(model.feats_extract.get_parameters())
    if "n_fft" in vocoder_conf and "n_shift" in vocoder_conf and "fs" in vocoder_conf:
        spc2wav = Spectrogram2Waveform(**vocoder_conf)
        logging.info(f"Vocoder: {spc2wav}")
    else:
        spc2wav = None
        logging.info("Vocoder is not used because vocoder_conf is not sufficient")

    # 5. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)
    (output_dir / "probs").mkdir(parents=True, exist_ok=True)

    with NpyScpWriter(
        output_dir / "norm", output_dir / "norm/feats.scp",
    ) as f, NpyScpWriter(output_dir / "denorm", output_dir / "denorm/feats.scp") as g:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = to_device(batch, device)

            key = keys[0]
            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            _data = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            start_time = time.perf_counter()

            _decode_conf = {
                "threshold": threshold,
                "maxlenratio": maxlenratio,
                "minlenratio": minlenratio,
            }
            if isinstance(tts, Tacotron2):
                _decode_conf.update(
                    {
                        "use_att_constraint": use_att_constraint,
                        "forward_window": forward_window,
                        "backward_window": backward_window,
                    }
                )
            outs, probs, att_ws = tts.inference(**_data, **_decode_conf)
            insize = next(iter(_data.values())).size(0) + 1
            logging.info(
                "inference speed = {:.1f} frames / sec.".format(
                    int(outs.size(0)) / (time.perf_counter() - start_time)
                )
            )
            logging.info(f"{key} (size:{insize}->{outs.size(0)})")
            if outs.size(0) == insize * maxlenratio:
                logging.warning(f"output length reaches maximum length ({key}).")
            f[key] = outs.cpu().numpy()

            # NOTE: normalize.inverse is in-place operation
            outs_denorm = normalize.inverse(outs[None])[0][0]
            g[key] = outs_denorm.cpu().numpy()

            # Lazy load to avoid the backend error
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator

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

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(output_dir / f"att_ws/{key}.png")
            fig.clf()

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

            fig.tight_layout()
            fig.savefig(output_dir / f"probs/{key}.png")
            fig.clf()

            # TODO(kamo): Write scp
            if spc2wav is not None:
                wav = spc2wav(outs_denorm.cpu().numpy())
                sf.write(f"{output_dir}/wav/{key}.wav", wav, spc2wav.fs, "PCM_16")


def get_parser():
    """Get argument parser."""
    parser = configargparse.ArgumentParser(
        description="TTS Decode",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--config", is_config_file=True, help="config file path",
    )

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
