#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import trange
from typeguard import check_argument_types

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.diar import DiarizationTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import (
    humanfriendly_parse_size_or_none,
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_none,
)
from espnet.utils.cli_utils import get_commandline_args


class DiarizeSpeech:
    """DiarizeSpeech class

    Examples:
        >>> import soundfile
        >>> diarization = DiarizeSpeech("diar_config.yaml", "diar.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> diarization(audio)
        [(spk_id, start, end), (spk_id2, start2, end2)]

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        num_spk: Optional[int] = None,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        assert check_argument_types()

        # 1. Build Diar model
        diar_model, diar_train_args = DiarizationTask.build_model_from_file(
            train_config, model_file, device
        )
        diar_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.diar_train_args = diar_train_args
        self.diar_model = diar_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.normalize_segment_scale = normalize_segment_scale
        self.show_progressbar = show_progressbar
        # not specifying "num_spk" in inference config file
        # will enable speaker number prediction during inference
        self.num_spk = num_spk

        self.segmenting = segment_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speaker diarization")
            logging.info("Segment length = {} sec".format(segment_size))
        else:
            logging.info("Perform direct speaker diarization on the input")

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], fs: int = 8000
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [speaker_info1, speaker_info2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)

        assert speech.dim() > 1, speech.size()
        batch_size = speech.size(0)
        speech = speech.to(getattr(torch, self.dtype))
        # lengths: (B,)
        lengths = speech.new_full(
            [batch_size], dtype=torch.long, fill_value=speech.size(1)
        )

        # a. To device
        speech = to_device(speech, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speaker diarization
            num_segments = int(np.ceil(speech.size(1) / (self.segment_size * fs)))
            t = T = int(self.segment_size * fs)
            pad_shape = speech[:, :T].shape
            diarized_wavs = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.segment_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech[:, st:en]
                else:
                    t = T
                    speech_seg = speech[:, st:en]  # B x T [x C]

                lengths_seg = speech.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Diarization Forward
                encoder_out, encoder_out_lens = self.diar_model.encode(
                    speech_seg, lengths_seg
                )
                # SA-EEND
                if self.diar_model.attractor is None:
                    assert (
                        self.num_spk is not None
                    ), 'Argument "num_spk" must be specified'
                    spk_prediction = self.diar_model.decoder(
                        encoder_out, encoder_out_lens
                    )
                # EEND-EDA
                else:
                    # if num_spk is specified, use that number
                    if self.num_spk is not None:
                        attractor, att_prob = self.diar_model.attractor(
                            encoder_out,
                            encoder_out_lens,
                            torch.zeros(
                                encoder_out.size(0),
                                self.num_spk + 1,
                                encoder_out.size(2),
                            ),
                        )
                        spk_prediction = torch.bmm(
                            encoder_out,
                            attractor[:, : self.num_spk, :].permute(0, 2, 1),
                        )
                    # else find the first att_prob[i] < 0
                    else:
                        max_num_spk = 15  # upper bound number for estimation
                        attractor, att_prob = self.diar_model.attractor(
                            encoder_out,
                            encoder_out_lens,
                            torch.zeros(
                                encoder_out.size(0),
                                max_num_spk + 1,
                                encoder_out.size(2),
                            ),
                        )
                        att_prob = torch.squeeze(att_prob)
                        for pred_num_spk in range(len(att_prob)):
                            if att_prob[pred_num_spk].item() < 0:
                                break
                        spk_prediction = torch.bmm(
                            encoder_out, attractor[:, :pred_num_spk, :].permute(0, 2, 1)
                        )
                # List[torch.Tensor(B, T, num_spks)]
                diarized_wavs.append(spk_prediction)
            # Determine maximum estimated number of speakers among the segments
            max_len = max([x.size(2) for x in diarized_wavs])
            # pad tensors in diarized_wavs with "float('-inf')" to have same size
            diarized_wavs = [
                torch.nn.functional.pad(
                    x, (0, max_len - x.size(2)), "constant", float("-inf")
                )
                for x in diarized_wavs
            ]
            spk_prediction = torch.cat(diarized_wavs, dim=1)
        else:
            # b. Diarization Forward
            encoder_out, encoder_out_lens = self.diar_model.encode(speech, lengths)
            # SA-EEND
            if self.diar_model.attractor is None:
                assert self.num_spk is not None, 'Argument "num_spk" must be specified'
                spk_prediction = self.diar_model.decoder(encoder_out, encoder_out_lens)
            # EEND-EDA
            else:
                # if num_spk is specified, use that number
                if self.num_spk is not None:
                    attractor, att_prob = self.diar_model.attractor(
                        encoder_out,
                        encoder_out_lens,
                        torch.zeros(
                            encoder_out.size(0), self.num_spk + 1, encoder_out.size(2)
                        ),
                    )
                    spk_prediction = torch.bmm(
                        encoder_out, attractor[:, : self.num_spk, :].permute(0, 2, 1)
                    )
                # else find the first att_prob[i] < 0
                else:
                    max_num_spk = 15  # upper bound number for estimation
                    attractor, att_prob = self.diar_model.attractor(
                        encoder_out,
                        encoder_out_lens,
                        torch.zeros(
                            encoder_out.size(0), max_num_spk + 1, encoder_out.size(2)
                        ),
                    )
                    att_prob = torch.squeeze(att_prob)
                    for pred_num_spk in range(len(att_prob)):
                        if att_prob[pred_num_spk].item() < 0:
                            break
                    spk_prediction = torch.bmm(
                        encoder_out, attractor[:, :pred_num_spk, :].permute(0, 2, 1)
                    )
        if self.num_spk is not None:
            assert spk_prediction.size(2) == self.num_spk, (
                spk_prediction.size(2),
                self.num_spk,
            )
        assert spk_prediction.size(0) == batch_size, (
            spk_prediction.size(0),
            batch_size,
        )
        spk_prediction = spk_prediction.cpu().numpy()
        spk_prediction = 1 / (1 + np.exp(-spk_prediction))

        return spk_prediction

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any],
    ):
        """Build DiarizeSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            DiarizeSpeech: DiarizeSpeech instance.

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

        return DiarizeSpeech(**kwargs)


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
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    show_progressbar: bool,
    num_spk: Optional[int],
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
    diarize_speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        segment_size=segment_size,
        show_progressbar=show_progressbar,
        num_spk=num_spk,
        device=device,
        dtype=dtype,
    )
    diarize_speech = DiarizeSpeech.from_pretrained(
        model_tag=model_tag, **diarize_speech_kwargs,
    )

    # 3. Build data-iterator
    loader = DiarizationTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=DiarizationTask.build_preprocess_fn(
            diarize_speech.diar_train_args, False
        ),
        collate_fn=DiarizationTask.build_collate_fn(
            diarize_speech.diar_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    writer = NpyScpWriter(f"{output_dir}/predictions", f"{output_dir}/diarize.scp")

    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        spk_predictions = diarize_speech(**batch)
        for b in range(batch_size):
            writer[keys[b]] = spk_predictions[b]

    writer.close()


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker Diarization inference",
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
        "--fs",
        type=humanfriendly_parse_size_or_none,
        default=8000,
        help="Sampling rate",
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
        "--train_config", type=str, help="Diarization training configuration",
    )
    group.add_argument(
        "--model_file", type=str, help="Diarization model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    group = parser.add_argument_group("Diarize speech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=None,
        help="Segment length in seconds for segment-wise speaker diarization",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speaker "
        "diarization",
    )
    group.add_argument(
        "--num_spk",
        type=int_or_none,
        default=None,
        help="Predetermined number of speakers for inference",
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
