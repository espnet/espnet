#!/usr/bin/env python3
import argparse
import logging
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.vad import VADTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class VoiceActivityDetect:
    """VoiceActivityDetect class

    Examples:
        >>> import soundfile
        >>> voiceactivitydetect = VoiceActivityDetect("vad_config.yml", "vad.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> voiceactivitydetect(audio)
        [(text), ...]

    """

    def __init__(
        self,
        vad_train_config: Union[Path, str] = None,
        vad_model_file: Union[Path, str] = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
        quantize_vad_model: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        threshold: float = 0.5,
        silence_to_speech_thresh: int = 3,
        speech_to_silence_thresh: int = 15,
    ):
        assert check_argument_types()

        task = VADTask

        if quantize_vad_model:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion(
                "1.5.0"
            ):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype = getattr(torch, quantize_dtype)

        # 1. Build VAD model
        scorers = {}
        vad_model, vad_train_args = task.build_model_from_file(
            vad_train_config, vad_model_file, device
        )
        vad_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_vad_model:
            logging.info("Use quantized vad model for decoding.")

            vad_model = torch.quantization.quantize_dynamic(
                vad_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )

        self.vad_model = vad_model
        self.vad_train_args = vad_train_args
        self.device = device
        self.dtype = dtype
        self.threshold = threshold
        self.silence_to_speech_thresh = silence_to_speech_thresh
        self.speech_to_silence_thresh = speech_to_silence_thresh

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[List[float]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            timestamps: List of timestamps for speech segments

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.vad_model.encode(**batch)
        enc = torch.softmax(enc, dim=-1).detach().cpu().numpy()

        # c. Passed the encoder result to do post-processing
        results = self.vad_post_processing(
            enc[0][:, 1],
            self.threshold,
            self.silence_to_speech_thresh,
            self.speech_to_silence_thresh,
        )
        assert check_return_type(results)

        return results

    def vad_post_processing(
        self,
        enc: np.ndarray,
        threshold: float,
        silence_to_speech_thresh: int,
        speech_to_silence_thresh: int,
    ) -> List[List[float]]:
        state = 0 if enc[0] > threshold else 1
        silence_frame_count = 0
        speech_frame_count = 0
        ret = []
        for i in range(1, len(enc)):
            # current state is silence
            if state == 0:
                # if current frame is speech
                if enc[i] > threshold:
                    speech_frame_count += 1
                    if speech_frame_count >= silence_to_speech_thresh:
                        state = 1
                        silence_frame_count = 0
                else:
                    speech_frame_count = 0
                    silence_frame_count += 1
            # current state is speech
            elif state == 1:
                # if current frame is silence
                if enc[i] <= threshold:
                    silence_frame_count += 1
                    if silence_frame_count >= speech_to_silence_thresh:
                        state = 0
                        speech_frame_count = 0
                else:
                    silence_frame_count = 0
                    speech_frame_count += 1
            else:
                raise ValueError("Invalid state")
            ret.append(state)
        # transform the state list to numerical start point and end point
        last_state = 0
        res, start, end = [], 0, 0
        for i in range(len(ret)):
            if last_state == 0 and ret[i] == 1:
                start = i
            if last_state == 1 and ret[i] == 0:
                end = i
                res.append([float(start) / 100, float(end / 100)])
            last_state = ret[i]
        if last_state == 1:
            res.append([float(start) / 100, float(len(ret) / 100)])
        # meaning there is no speech in the audio
        if len(res) == 0:
            res.append([0.0, 0.0])
        return res

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

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

        return VoiceActivityDetect(**kwargs)


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
    vad_train_config: Optional[str],
    vad_model_file: Optional[str],
    model_tag: Optional[str],
    allow_variable_data_keys: bool,
    quantize_vad_model: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    threshold: float,
    silence_to_speech_thresh: int,
    speech_to_silence_thresh: int,
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

    # 2. Build speech2text
    voiceactivitydetect_kwargs = dict(
        vad_train_config=vad_train_config,
        vad_model_file=vad_model_file,
        device=device,
        dtype=dtype,
        quantize_vad_model=quantize_vad_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        threshold=threshold,
        silence_to_speech_thresh=silence_to_speech_thresh,
        speech_to_silence_thresh=speech_to_silence_thresh,
    )
    voiceactivitydetect = VoiceActivityDetect.from_pretrained(
        model_tag=model_tag,
        **voiceactivitydetect_kwargs,
    )

    # 3. Build data-iterator
    loader = VADTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=VADTask.build_preprocess_fn(
            voiceactivitydetect.vad_train_args, False
        ),
        collate_fn=VADTask.build_collate_fn(voiceactivitydetect.vad_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            try:
                results = voiceactivitydetect(**batch)
            except Exception as e:
                logging.warning(f"Utterance {keys} {e}")

            # Only supporting batch_size==1
            key = keys[0]

            str_result = []
            for item in results:
                str_result.append(" ".join([str(x) for x in item]))
            str_result = " ".join(str_result)
            # Create a directory: vad_result
            ibest_writer = writer[f"vad_result"]
            # Write the result to each file
            ibest_writer["segments"][key] = str_result


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="VAD Decoding",
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

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--vad_train_config",
        type=str,
        help="VAD training configuration",
    )
    group.add_argument(
        "--vad_model_file",
        type=str,
        help="VAD model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_vad_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to VAD model.",
    )
    group.add_argument(
        "--quantize_modules",
        type=str,
        nargs="*",
        default=["Linear"],
        help="""List of modules to be dynamically quantized.
        E.g.: --quantize_modules=[Linear,LSTM,GRU].
        Each specified module should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    group.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for VAD",
    )
    group.add_argument(
        "--silence_to_speech_thresh",
        type=int,
        default=3,
        help="Threshold for silence to speech",
    )
    group.add_argument(
        "--speech_to_silence_thresh",
        type=int,
        default=15,
        help="Threshold for speech to silence",
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
