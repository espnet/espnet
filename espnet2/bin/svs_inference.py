#!/usr/bin/env python3

"""Script to run the inference of singing-voice-synthesis model."""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.gan_svs.vits import VITS
from espnet2.svs.singing_tacotron.singing_tacotron import singing_tacotron
from espnet2.tasks.svs import SVSTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tts.utils import DurationCalculator
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


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
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        use_dynamic_filter: bool = False,
        backward_window: int = 2,
        forward_window: int = 4,
        speed_control_alpha: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        vocoder_config: Union[Path, str] = None,
        vocoder_checkpoint: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        prefer_normalized_feats: bool = False,
    ):
        """Initialize SingingGenerate module."""
        assert check_argument_types()

        # setup model
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
        self.duration_calculator = DurationCalculator()
        self.preprocess_fn = SVSTask.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        self.vocoder = None
        self.prefer_normalized_feats = prefer_normalized_feats
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
        if self.vocoder is not None:
            logging.info(f"Vocoder:\n{self.vocoder}")

        # setup decoding config
        decode_conf = {}
        decode_conf.update({"use_teacher_forcing": use_teacher_forcing})
        if isinstance(self.svs, VITS):
            decode_conf.update(
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
            )
        if isinstance(self.svs, singing_tacotron):
            decode_conf.update(
                threshold=threshold,
                maxlenratio=maxlenratio,
                minlenratio=minlenratio,
                use_att_constraint=use_att_constraint,
                use_dynamic_filter=use_dynamic_filter,
                forward_window=forward_window,
                backward_window=backward_window,
            )
        self.decode_conf = decode_conf

    @torch.no_grad()
    def __call__(
        self,
        text: Union[Dict[str, Tuple], torch.Tensor, np.ndarray],
        singing: Union[torch.Tensor, np.ndarray] = None,
        label: Union[torch.Tensor, np.ndarray] = None,
        midi: Union[torch.Tensor, np.ndarray] = None,
        duration_phn: Union[torch.Tensor, np.ndarray] = None,
        duration_ruled_phn: Union[torch.Tensor, np.ndarray] = None,
        duration_syb: Union[torch.Tensor, np.ndarray] = None,
        phn_cnt: Union[torch.Tensor, np.ndarray] = None,
        slur: Union[torch.Tensor, np.ndarray] = None,
        pitch: Union[torch.Tensor, np.ndarray] = None,
        energy: Union[torch.Tensor, np.ndarray] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        sids: Union[torch.Tensor, np.ndarray] = None,
        lids: Union[torch.Tensor, np.ndarray] = None,
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

        # prepare batch
        if isinstance(text, Dict):
            data = self.preprocess_fn(
                "<dummy>", dict(label=text["label"], score=text["score"])
            )
            label = data["label"]
            midi = data["midi"]
            duration_phn = data["duration_phn"]
            duration_ruled_phn = data["duration_ruled_phn"]
            duration_syb = data["duration_syb"]
            phn_cnt = data["phn_cnt"]
            slur = data["slur"]
            batch = dict(text=data["label"])
        else:
            batch = dict(text=text)

        if singing is not None:
            batch.update(singing=singing)
        if label is not None:
            batch.update(label=label)
        if midi is not None:
            batch.update(midi=midi)
        if duration_phn is not None:
            batch.update(duration_phn=duration_phn)
        if duration_ruled_phn is not None:
            batch.update(duration_ruled_phn=duration_ruled_phn)
        if duration_syb is not None:
            batch.update(duration_syb=duration_syb)
        if pitch is not None:
            batch.update(pitch=pitch)
        if phn_cnt is not None:
            batch.update(phn_cnt=phn_cnt)
        if slur is not None:
            batch.update(slur=slur)
        if energy is not None:
            batch.update(energy=energy)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        batch = to_device(batch, self.device)

        cfg = self.decode_conf
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)
        output_dict = self.model.inference(**batch, **cfg)

        if output_dict.get("att_w") is not None:
            duration, focus_rate = self.duration_calculator(output_dict["att_w"])
            output_dict.update(duration=duration, focus_rate=focus_rate)
        else:
            output_dict.update(duration=None, focus_rate=None)

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

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        vocoder_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build SingingGenerate instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            SingingGenerate: SingingGenerate instance.

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

        return SingingGenerate(**kwargs)


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
    noise_scale: float,
    noise_scale_dur: float,
    allow_variable_data_keys: bool,
    vocoder_config: Optional[str] = None,
    vocoder_checkpoint: Optional[str] = None,
    vocoder_tag: Optional[str] = None,
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
        noise_scale=noise_scale,
        noise_scale_dur=noise_scale_dur,
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
            output_dict = singingGenerate(**batch)

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            if output_dict.get("feat_gen") is not None:
                # standard text2mel model case
                feat_gen = output_dict["feat_gen"]
                logging.info(
                    "inference speed = {:.1f} frames / sec.".format(
                        int(feat_gen.size(0)) / (time.perf_counter() - start_time)
                    )
                )
                logging.info(f"{key} (size:{insize}->{feat_gen.size(0)})")

                norm_writer[key] = output_dict["feat_gen"].cpu().numpy()
                shape_writer.write(
                    f"{key} " + ",".join(map(str, output_dict["feat_gen"].shape)) + "\n"
                )
                if output_dict.get("feat_gen_denorm") is not None:
                    denorm_writer[key] = output_dict["feat_gen_denorm"].cpu().numpy()
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
                    + " ".join(map(str, output_dict["duration"].long().cpu().numpy()))
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
            # TODO(kamo): Write scp
            if output_dict.get("wav") is not None:
                sf.write(
                    f"{output_dir}/wav/{key}.wav",
                    output_dict["wav"].cpu().numpy(),
                    singingGenerate.fs,
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

    group = parser.add_argument_group("Vocoder related")
    group.add_argument(
        "--vocoder_checkpoint",
        default="None",
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
