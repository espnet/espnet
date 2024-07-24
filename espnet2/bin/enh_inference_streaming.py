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
from typeguard import typechecked

from espnet2.bin.enh_inference import (
    build_model_from_args_and_file,
    get_train_config,
    recursive_dict_update,
)
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args

EPS = torch.finfo(torch.get_default_dtype()).eps


class SeparateSpeechStreaming:
    """SeparateSpeechStreaming class. Separate a small audio chunk in streaming.

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeechStreaming("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> lengths = torch.LongTensor(audio.shape[-1])
        >>> speech_sim_chunks = separate_speech.frame(wav)
        >>> output_chunks = [[] for ii in range(separate_speech.num_spk)]
        >>>
        >>> for chunk in speech_sim_chunks:
        >>>     output = separate_speech(chunk)
        >>>     for spk in range(separate_speech.num_spk):
        >>>         output_chunks[spk].append(output[spk])
        >>>
        >>> separate_speech.reset()
        >>> waves = [
        >>>     separate_speech.merge(chunks, length)
        >>>     for chunks in output_chunks ]
    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        inference_config: Union[Path, str, None] = None,
        ref_channel: Optional[int] = None,
        device: str = "cpu",
        dtype: str = "float32",
        enh_s2t_task: bool = False,
    ):

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

        self.streaming_states = None

    def frame(self, audio):
        return self.enh_model.encoder.streaming_frame(audio)

    def merge(self, chunks, ilens=None):
        return self.enh_model.decoder.streaming_merge(chunks, ilens=ilens)

    def reset(self):
        self.streaming_states = None

    @torch.no_grad()
    @typechecked
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

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)

        # b. Enhancement/Separation Forward
        # frame_feature: (B, 1, F)
        frame_feature = self.enh_model.encoder.forward_streaming(speech_mix)

        # frame_separated: list of num_spk [(B, 1, F)]
        (
            frame_separated,
            self.streaming_states,
            _,
        ) = self.enh_model.separator.forward_streaming(
            frame_feature, self.streaming_states
        )

        # frame_separated: list of num_spk [(B, frame_size)]
        waves = [self.enh_model.decoder.forward_streaming(f) for f in frame_separated]

        assert len(waves) == self.num_spk, (len(waves), self.num_spk)
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)

        return waves

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
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

        return SeparateSpeechStreaming(**kwargs)


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


@typechecked
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
    ref_channel: Optional[int],
    output_format: str,
    enh_s2t_task: bool,
):
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
        ref_channel=ref_channel,
        device=device,
        dtype=dtype,
        enh_s2t_task=enh_s2t_task,
    )
    separate_speech = SeparateSpeechStreaming.from_pretrained(
        model_tag=model_tag,
        **separate_speech_kwargs,
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

    # 4. Start dataset for-loop
    output_dir = Path(output_dir).expanduser().resolve()
    writers = []
    for i in range(separate_speech.num_spk):
        writers.append(
            SoundScpWriter(
                f"{output_dir}/wavs/{i + 1}",
                f"{output_dir}/spk{i + 1}.scp",
                format=output_format,
            )
        )

    import tqdm

    for i, (keys, batch) in tqdm.tqdm(enumerate(loader)):
        logging.info(f"[{i}] Enhancing {keys}")
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        speech = batch["speech_mix"]
        lengths = speech.new_full(
            [batch_size], dtype=torch.long, fill_value=speech.size(1)
        )

        # split continuous speech into small chunks to simulate streaming
        speech_sim_chunks = separate_speech.frame(speech)
        output_chunks = [[] for ii in range(separate_speech.num_spk)]

        # the main loop for streaming processing
        for chunk in speech_sim_chunks:
            # process a single chunk
            output = separate_speech(chunk, fs=fs)
            for channel in range(separate_speech.num_spk):
                # append processed chunks to ouput channels
                output_chunks[channel].append(output[channel])

        # reset separator states after processing
        separate_speech.reset()

        # merge chunks
        waves = [separate_speech.merge(chunks, lengths) for chunks in output_chunks]

        waves = [
            (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy() for w in waves
        ]

        for spk, w in enumerate(waves):
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
        "--output_format",
        type=str,
        default="wav",
        help="Output format for the separated speech",
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
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group = parser.add_argument_group("SeparateSpeech related")
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
