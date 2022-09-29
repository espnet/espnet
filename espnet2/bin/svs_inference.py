#!/usr/bin/env python3

"""Script to run the inference of singing-voice-synthesis model."""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.svs import SVSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed

from espnet2.svs.naive_rnn.naive_rnn import NaiveRNN
from espnet2.svs.naive_rnn.naive_rnn_dp import NaiveRNNDP

# from espnet2.svs.glu_transformer.glu_transformer import GLU_Transformer
# from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing
# from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing_noDP

from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str2triple_str, str_or_none


class SingingGenerate:
    """SingingGenerate class

    Examples:
        >>> import soundfile
        >>> svs = SingingGenerate("config.yml", "model.pth")
        >>> wav = svs("Hello World")[0]
        >>> soundfile.write("out.wav", wav.numpy(), svs.fs, "PCM_16")
    """

    def __init__(
        self,
        train_config: Optional[Union[Path, str]],
        model_file: Optional[Union[Path, str]] = None,
        use_teacher_forcing: bool = False,
        vocoder_config: Union[Path, str] = None,
        vocoder_checkpoint: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
    ):
        assert check_argument_types()

        model, train_args = SVSTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.svs = model.svs
        self.normalize = model.normalize
        self.feats_extract = model.feats_extract
        # self.duration_calculator = DurationCalculator() # TODO
        self.preprocess_fn = SVSTask.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing

        self.vocoder = None
        if vocoder_checkpoint is not None:
            vocoder = SVSTask.build_vocoder_from_file(
                vocoder_config, vocoder_checkpoint, model, device
            )
            if isinstance(vocoder, torch.nn.Module):
                vocoder.to(dtype=getattr(torch, dtype)).eval()
            self.vocoder = vocoder

        logging.info(f"Extractor:\n{self.feats_extract}")
        logging.info(f"Normalizer:\n{self.normalize}")
        logging.info(f"SVS:\n{self.svs}")

        decode_config = {}
        decode_config.update({"use_teacher_forcing": use_teacher_forcing})

        self.decode_config = decode_config

    @torch.no_grad()
    def __call__(
        self,
        text: torch.Tensor,
        singing: torch.Tensor = None,
        label_lab: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
    ):
        assert check_argument_types()

        # check inputs
        if self.use_sids and sids is None:
            raise RuntimeError("Missing required argument: 'sids'")
        if self.use_lids and lids is None:
            raise RuntimeError("Missing required argument: 'lids'")
        if self.use_spembs and spembs is None:
            raise RuntimeError("Missing required argument: 'spembs'")

        batch = dict(
            text=text,
        )
        if label_lab is not None:
            batch.update(label_lab=label_lab)
        if label_xml is not None:
            batch.update(label_xml=label_xml)
        if midi_lab is not None:
            batch.update(midi_lab=midi_lab)
        if midi_xml is not None:
            batch.update(midi_xml=midi_xml)
        if pitch is not None:
            batch.update(pitch=pitch)
        if tempo_lab is not None:
            batch.update(tempo_lab=tempo_lab)
        if tempo_xml is not None:
            batch.update(tempo_xml=tempo_xml)
        if beat_lab is not None:
            batch.update(beat_lab=beat_lab)
        if beat_xml is not None:
            batch.update(beat_xml=beat_xml)
        if energy is not None:
            batch.update(energy=energy)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        batch = to_device(batch, self.device)

        cfg = self.decode_config
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)

        batch = to_device(batch, self.device)
        outs, outs_denorm, probs, att_ws = self.model.inference(**batch, **cfg)

        if att_ws is not None:
            duration, focus_rate = self.duration_calculator(att_ws)
        else:
            duration, focus_rate = None, None

        assert outs.shape[0] == 1
        outs = outs.squeeze(0)
        outs_denorm = outs_denorm.squeeze(0)
        if self.vocoder is not None:
            if self.vocoder.normalize_before:
                wav = self.vocoder(outs_denorm)
            else:
                wav = self.vocoder(outs)
        else:
            wav = None

        return wav, outs, outs_denorm, probs, att_ws, duration, focus_rate

    @property
    def fs(self) -> Optional[int]:
        """Return sampling rate."""
        if hasattr(self.vocoder, "fs"):
            return self.vocoder.fs
        elif hasattr(self.svs, "fs"):
            return self.svs.fs
        else:
            return None

    @property
    def use_speech(self) -> bool:
        """Return speech is needed or not in the inference."""
        return self.use_teacher_forcing or getattr(self.svs, "use_gst", False)

    @property
    def use_sids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.svs.spks is not None

    @property
    def use_lids(self) -> bool:
        """Return sid is needed or not in the inference."""
        return self.svs.langs is not None

    @property
    def use_spembs(self) -> bool:
        """Return spemb is needed or not in the inference."""
        return self.svs.spk_embed_dim is not None


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
    use_teacher_forcing: bool,
    allow_variable_data_keys: bool,
    vocoder_config: Optional[str] = None,
    vocoder_checkpoint: Optional[str] = None,
):
    """Perform SVS model decoding."""
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
    singingGenerate = SingingGenerate(
        train_config=train_config,
        model_file=model_file,
        use_teacher_forcing=use_teacher_forcing,
        vocoder_config=vocoder_config,
        vocoder_checkpoint=vocoder_checkpoint,
        dtype=dtype,
        device=device,
    )

    # 3. Build data-iterator
    loader = SVSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SVSTask.build_preprocess_fn(singingGenerate.train_args, False),
        collate_fn=SVSTask.build_collate_fn(singingGenerate.train_args, False),
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

            logging.info(f"batch: {batch}")
            logging.info(f"keys: {keys}")

            start_time = time.perf_counter()
            (
                wav,
                outs,
                outs_denorm,
                probs,
                att_ws,
                duration,
                focus_rate,
            ) = singingGenerate(**batch)

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            logging.info(
                "inference speed = {:.1f} frames / sec.".format(
                    int(outs.size(0)) / (time.perf_counter() - start_time)
                )
            )
            logging.info(f"{key} (size:{insize}->{outs.size(0)})")

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
                    f"{output_dir}/wav/{key}.wav",
                    wav.numpy(),
                    singingGenerate.fs,
                    "PCM_16",
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
        description="SVS Decode",
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
        "--use_teacher_forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )

    group = parser.add_argument_group("Vocoder related")
    group.add_argument(
        "--vocoder_checkpoint",
        default="/data5/gs/vocoder_peter/hifigan-vocoder/exp/train_hifigan.v1_train_nodev_clean_libritts_hifigan-2.v1/checkpoint-50000steps.pkl",
        type=str_or_none,
        help="checkpoint file to be loaded.",
    )
    group.add_argument(
        "--vocoder_config",
        default=None,
        type=str_or_none,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )

    return parser


def main(cmd=None):
    """Run SVS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
