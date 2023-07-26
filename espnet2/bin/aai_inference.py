#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""

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
from packaging.version import parse as V
from typeguard import check_argument_types
from espnet2.fileio.datadir_writer import DatadirWriter

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.aai import AAITask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class AAI:
    def __init__(
        self,
        aai_train_config: Union[Path, str] = None,
        aai_model_file: Union[Path, str] = None,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
    ):
        """Initialize Text2Speech module."""
        assert check_argument_types()

        # setup model
        model, train_args = AAITask.build_model_from_file(
            aai_train_config, aai_model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.aai_model = model
        self.aai_train_args = train_args
        decode_conf = {}
        self.decode_conf = decode_conf

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run text-to-speech."""
        assert check_argument_types()

        # check inputs
        if speech is None:
            raise RuntimeError("Missing required argument: 'speech'")

        # prepare batch
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # overwrite the decode configs if provided
        cfg = self.decode_conf
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)
        output = self.aai_model.encode(**batch)
        output = self.aai_model.decoder(output)
        print(output.shape, speech.shape)

        

       

        return output_dict

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
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

        return AAI(**kwargs)


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
    aai_train_config: Optional[str],
    aai_model_file: Optional[str],
    model_tag: Optional[str],
    minlenratio: float,
    maxlenratio: float,
    allow_variable_data_keys: bool,
    decoder_weight=1.0,

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
    aai_kwargs = dict(
        aai_train_config=aai_train_config,
        aai_model_file=aai_model_file,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    aai = AAI.from_pretrained(
        model_tag=model_tag,
        **aai_kwargs,
    )

    # 3. Build data-iterator

    loader = AAITask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=AAITask.build_preprocess_fn(aai.train_args, False),
        collate_fn=AAITask.build_collate_fn(aai.train_args, False),
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with DatadirWriter(output_dir) as ema_writer:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            start_time = time.perf_counter()
            output_dict = aai(**batch)

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
                if feat_gen.size(0) == insize * maxlenratio:
                    logging.warning(f"output length reaches maximum length ({key}).")

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
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--aai_train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--aai_model_file",
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
        "--decoder_weight",
        type=float,
        default=1.0,
    )
    return parser


def main(cmd=None):
    """Run AAI model inference."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
