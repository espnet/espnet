#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import yaml
from tqdm import trange
from typeguard import check_argument_types

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

EPS = torch.finfo(torch.get_default_dtype()).eps


def get_train_config(train_config, model_file=None):
    if train_config is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'train_config' is not specified."
        )
        train_config = Path(model_file).parent / "config.yaml"
    else:
        train_config = Path(train_config)
    return train_config


def recursive_dict_update(dict_org, dict_patch, verbose=False, log_prefix=""):
    """Update `dict_org` with `dict_patch` in-place recursively."""
    for key, value in dict_patch.items():
        if key not in dict_org:
            if verbose:
                logging.info(
                    "Overwriting config: [{}{}]: None -> {}".format(
                        log_prefix, key, value
                    )
                )
            dict_org[key] = value
        elif isinstance(value, dict):
            recursive_dict_update(
                dict_org[key], value, verbose=verbose, log_prefix=f"{key}."
            )
        else:
            if verbose and dict_org[key] != value:
                logging.info(
                    "Overwriting config: [{}{}]: {} -> {}".format(
                        log_prefix, key, dict_org[key], value
                    )
                )
            dict_org[key] = value


def build_model_from_args_and_file(task, args, model_file, device):
    model = task.build_model(args)
    if not isinstance(model, AbsESPnetModel):
        raise RuntimeError(
            f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
        )
    model.to(device)
    if model_file is not None:
        if device == "cuda":
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            device = f"cuda:{torch.cuda.current_device()}"
        model.load_state_dict(torch.load(model_file, map_location=device))
    return model


class SeparateSpeech:
    """SeparateSpeech class

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeech("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separate_speech(audio)
        [separated_audio1, separated_audio2, ...]

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        inference_config: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        ref_channel: Optional[int] = None,
        normalize_output_wav: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
        enh_s2t_task: bool = False,
    ):
        assert check_argument_types()

        task = EnhancementTask if not enh_s2t_task else EnhS2TTask

        # 1. Build Enh model

        if inference_config is None:
            enh_model, enh_train_args = task.build_model_from_file(
                train_config, model_file, device
            )
        else:
            # Overwrite model attributes
            train_config = get_train_config(train_config, model_file=model_file)
            with train_config.open("r", encoding="utf-8") as f:
                train_args = yaml.safe_load(f)

            with Path(inference_config).open("r", encoding="utf-8") as f:
                infer_args = yaml.safe_load(f)

            if enh_s2t_task:
                arg_list = ("enh_encoder", "enh_separator", "enh_decoder")
            else:
                arg_list = ("encoder", "separator", "decoder")
            supported_keys = list(chain(*[[k, k + "_conf"] for k in arg_list]))
            for k in infer_args.keys():
                if k not in supported_keys:
                    raise ValueError(
                        "Only the following top-level keys are supported: %s"
                        % ", ".join(supported_keys)
                    )

            recursive_dict_update(train_args, infer_args, verbose=True)
            enh_train_args = argparse.Namespace(**train_args)
            enh_model = build_model_from_args_and_file(
                task, enh_train_args, model_file, device
            )

        if enh_s2t_task:
            enh_model = enh_model.enh_model
        enh_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.enh_model = enh_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar

        self.num_spk = enh_model.num_spk
        task = "enhancement" if self.num_spk == 1 else "separation"

        # reference channel for processing multi-channel speech
        if ref_channel is not None:
            logging.info(
                "Overwrite enh_model.separator.ref_channel with {}".format(ref_channel)
            )
            enh_model.separator.ref_channel = ref_channel
            self.ref_channel = ref_channel
        else:
            self.ref_channel = enh_model.ref_channel

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech %s" % task)
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech %s on the input" % task)

    @torch.no_grad()
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray], fs: int = 8000
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [separated_audio1, separated_audio2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        # lengths: (B,)
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]  # B x T [x C]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Enhancement/Separation Forward
                feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
                feats, _, _ = self.enh_model.separator(feats, f_lens)
                processed_wav = [
                    self.enh_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                if speech_seg.dim() > 2:
                    # multi-channel speech
                    speech_seg_ = speech_seg[:, self.ref_channel]
                else:
                    speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    # normalize the scale to match the input mixture scale
                    mix_energy = torch.sqrt(
                        torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
                    )
                    enh_energy = torch.sqrt(
                        torch.mean(
                            sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True
                        )
                    )
                    processed_wav = [
                        w * (mix_energy / enh_energy) for w in processed_wav
                    ]
                # List[torch.Tensor(num_spk, B, T)]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            # c. Stitch the enhanced segments together
            waves = enh_waves[0]
            for i in range(1, num_segments):
                # permutation between separated streams in last and current segments
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                # repermute separated streams in current segment
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                # overlap-and-add (average over the overlapped part)
                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            # ensure the stitched length is same as input
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            # b. Enhancement/Separation Forward
            feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
            feats, _, _ = self.enh_model.separator(feats, f_lens)
            waves = [self.enh_model.decoder(f, lengths)[0] for f in feats]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]  # list[(batch, sample)]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return waves

    @torch.no_grad()
    def cal_permumation(self, ref_wavs, enh_wavs, criterion="si_snr"):
        """Calculate the permutation between seaprated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """

        criterion_class = {"si_snr": SISNRLoss, "mse": FrequencyDomainMSE}[criterion]

        pit_solver = PITSolver(criterion=criterion_class())

        _, _, others = pit_solver(ref_wavs, enh_wavs)
        perm = others["perm"]
        return perm

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any],
    ):
        """Build SeparateSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            SeparateSpeech: SeparateSpeech instance.

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

        return SeparateSpeech(**kwargs)


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    inference_config: Optional[str],
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    show_progressbar: bool,
    ref_channel: Optional[int],
    normalize_output_wav: bool,
    enh_s2t_task: bool,
):
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

    # 2. Build separate_speech
    separate_speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        inference_config=inference_config,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        ref_channel=ref_channel,
        normalize_output_wav=normalize_output_wav,
        device=device,
        dtype=dtype,
        enh_s2t_task=enh_s2t_task,
    )
    separate_speech = SeparateSpeech.from_pretrained(
        model_tag=model_tag, **separate_speech_kwargs,
    )

    # 3. Build data-iterator
    loader = EnhancementTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=EnhancementTask.build_preprocess_fn(
            separate_speech.enh_train_args, False
        ),
        collate_fn=EnhancementTask.build_collate_fn(
            separate_speech.enh_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    output_dir = Path(output_dir).expanduser().resolve()
    writers = []
    for i in range(separate_speech.num_spk):
        writers.append(
            SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        )

    for i, (keys, batch) in enumerate(loader):
        logging.info(f"[{i}] Enhancing {keys}")
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        waves = separate_speech(**batch)
        for (spk, w) in enumerate(waves):
            for b in range(batch_size):
                writers[spk][keys[b]] = fs, w[b]

    for writer in writers:
        writer.close()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
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
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
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

    group = parser.add_argument_group("Output data related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config", type=str, help="Training configuration file",
    )
    group.add_argument(
        "--model_file", type=str, help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )
    group.add_argument(
        "--inference_config",
        type=str_or_none,
        default=None,
        help="Optional configuration file for overwriting enh model attributes "
        "during inference",
    )
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="enhancement and asr joint model",
    )

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    group = parser.add_argument_group("SeparateSpeech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=None,
        help="Segment length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--hop_size",
        type=float,
        default=None,
        help="Hop length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--normalize_segment_scale",
        type=str2bool,
        default=False,
        help="Whether to normalize the energy of the separated streams in each segment",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speech "
        "enhancement/separation",
    )
    group.add_argument(
        "--ref_channel",
        type=int,
        default=None,
        help="If not None, this will overwrite the ref_channel defined in the "
        "separator module (for multi-channel speech processing)",
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
