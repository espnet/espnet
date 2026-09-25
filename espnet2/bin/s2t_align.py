#!/usr/bin/env python3
# Copyright 2021, Ludwig Kürzinger; Kamo Naoyuki; Yifan Peng
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC segmentation to align utterances within audio files using OWSM-CTC.

The four alignment modules in espnet2.bin, and what each is::

    align.py        forced alignment over any CTC head; what `espnet align`
                    and the MCP server run
    asr_align.py    CTC segmentation with an ASR model, and its script
    s2t_align.py    CTC segmentation with an OWSM-CTC model, and its script
    ctc_segment.py  the CTC segmentation algorithm those two share; not an
                    entry point

The algorithm is `espnet2.bin.ctc_segment`, shared with
`espnet2.bin.asr_align`. What is here is how an OWSM-CTC model produces the
CTC posteriors - in windows, with a language and a task symbol in front of
each - and this module is what such a model is aligned from::

    from espnet2.bin.s2t_align import CTCSegmentation
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# CTCSegmentationTask is re-exported: it has been importable from this
# module since 2024, and the tests and recipes that use it say so.
from espnet2.bin.ctc_segment import (  # noqa: F401
    AbsCTCSegmentation,
    CTCSegmentationTask,
    align_file,
    build_parser,
)
from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.s2t_ctc import S2TTask
from espnet2.torch_utils.device_funcs import to_device


class CTCSegmentation(AbsCTCSegmentation):
    """Align text to audio with an OWSM-CTC model, using CTC segmentation.

    Usage:
        Initialize with given OWSM-CTC model and parameters.
        If needed, parameters for CTC segmentation can be set with ``set_config(·)``.
        Then call the instance as function to align text within an audio file.

    Example:
        >>> # example file included in the ESPnet repository
        >>> import soundfile
        >>> speech, fs = soundfile.read("test_utils/ctc_align_test.wav")
        >>> # an OWSM-CTC model, by the tag its model card gives
        >>> aligner = CTCSegmentation.from_pretrained("espnet/owsm_ctc_v4_1B")
        >>> text=["utt1 THE SALE OF THE HOTELS", "utt2 ON PROPERTY MANAGEMENT"]
        >>> aligner.set_config( gratis_blank=True )
        >>> segments = aligner( speech, text, fs=fs )
        >>> print( segments )
        utt1 utt 0.27 1.72 -0.1663 THE SALE OF THE HOTELS
        utt2 utt 4.54 6.10 -4.9646 ON PROPERTY MANAGEMENT

    The algorithm, the parameters it takes and how to split it across
    processes are documented in `espnet2.bin.ctc_segment`.

    """

    def __init__(
        self,
        s2t_train_config: Union[Path, str] = None,
        s2t_model_file: Union[Path, str] = None,
        fs: int = 16000,
        ngpu: int = 0,
        batch_size: int = 1,
        dtype: str = "float32",
        kaldi_style_text: bool = True,
        text_converter: str = "tokenize",
        time_stamps: str = "auto",
        lang_sym: str = "<eng>",
        task_sym: str = "<asr>",
        context_len_in_secs: float = 4,
        # last, and after every argument this class already had: a caller
        # passing them positionally must keep binding what it bound before
        device: Optional[str] = None,
        **ctc_segmentation_args,
    ):
        """Initialize the CTCSegmentation module.

        Args:
            s2t_train_config: S2T model config file (yaml).
            s2t_model_file: S2T model file (pth).
            fs: Sample rate of audio file.
            device: Where to run, as torch spells it: "cpu", "cuda",
                "cuda:1", "mps". Overrides `ngpu`, which cannot say which.
            ngpu: Number of GPUs. Set 0 for processing on CPU, set to 1 for
                processing on GPU. Multi-GPU aligning is currently not
                implemented. Default: 0.
            batch_size: Currently, only batch size == 1 is implemented.
            dtype: Data type used for inference. Set dtype according to
                the ASR model.
            kaldi_style_text: A kaldi-style text file includes the name of the
                utterance at the start of the line. If True, the utterance name
                is expected as first word at each line. If False, utterance
                names are automatically generated. Set this option according to
                your input data. Default: True.
            text_converter: How CTC segmentation handles text.
                "tokenize": Use ESPnet 2 preprocessing to tokenize the text.
                "classic": The text is preprocessed as in ESPnet 1 which takes
                token length into account. If the ASR model has longer tokens,
                this option may yield better results. Default: "tokenize".
            time_stamps: Choose the method how the time stamps are
                calculated. While "fixed" and "auto" use both the sample rate,
                the ratio of samples to one frame is either automatically
                determined for each inference or fixed at a certain ratio that
                is initially determined by the module, but can be changed via
                the parameter ``samples_to_frames_ratio``. Recommended for
                longer audio files: "auto".
            **ctc_segmentation_args: Parameters for CTC segmentation.
        """

        # Basic settings
        device = self._resolve_device(device, ngpu)

        # Prepare ASR model
        s2t_model, s2t_train_args = S2TTask.build_model_from_file(
            s2t_train_config, s2t_model_file, device
        )
        s2t_model.to(dtype=getattr(torch, dtype)).eval()
        self.preprocess_fn = S2TTask.build_preprocess_fn(s2t_train_args, False)

        # Warn for nets with high memory consumption on long audio files
        if hasattr(s2t_model, "encoder"):
            encoder_module = s2t_model.encoder.__class__.__module__
        else:
            encoder_module = "Unknown"
        logging.info(f"Encoder module: {encoder_module}")
        logging.info(f"CTC module:     {s2t_model.ctc.__class__.__module__}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.device = device
        self.dtype = dtype
        self.ctc = s2t_model.ctc

        self.kaldi_style_text = kaldi_style_text
        self.token_list = s2t_model.token_list
        # Apply configuration
        self.set_config(
            fs=fs,
            time_stamps=time_stamps,
            kaldi_style_text=kaldi_style_text,
            text_converter=text_converter,
            **ctc_segmentation_args,
        )
        self.config.char_list = s2t_model.token_list

        self.batch_size = batch_size
        self.lang_sym = lang_sym
        self.task_sym = task_sym
        self.context_len_in_secs = context_len_in_secs

        subsample_dict = {
            "conv2d1": 1,
            "conv2d2": 2,
            "conv2d": 4,
            "conv2d6": 6,
            "conv2d8": 8,
        }
        subsample_factor = subsample_dict[s2t_train_args.encoder_conf["input_layer"]]
        self.samples_to_frames_ratio = (
            s2t_train_args.frontend_conf["hop_length"] * subsample_factor
        )
        self.frames_per_sec = fs / self.samples_to_frames_ratio

    @torch.no_grad()
    def get_lpz(self, speech: Union[torch.Tensor, np.ndarray]):
        """Obtain CTC posterior log probabilities for given speech data.

        Args:
            speech: Speech input.

        Returns:
            lpz: Numpy vector with CTC log posterior probabilities.
        """

        lang_id = self.token_list.index(self.lang_sym)
        task_id = self.token_list.index(self.task_sym)
        context_len_in_secs = self.context_len_in_secs
        sample_rate = self.fs
        frames_per_sec = self.frames_per_sec
        batch_size = self.batch_size

        buffer_len_in_secs = self.s2t_train_args.preprocessor_conf["speech_length"]
        chunk_len_in_secs = buffer_len_in_secs - 2 * context_len_in_secs
        buffer_len = int(sample_rate * buffer_len_in_secs)
        chunk_len = int(sample_rate * chunk_len_in_secs)

        speech = np.pad(
            speech,
            (
                int(sample_rate * context_len_in_secs),
                int(sample_rate * context_len_in_secs),
            ),
        )
        buffer_list = []
        for i in range(0, len(speech), chunk_len):
            cur_buffer = speech[i : i + buffer_len]
            if len(cur_buffer) < buffer_len:
                buffer_list.append(
                    np.pad(cur_buffer, (0, buffer_len - len(cur_buffer)))
                )
                break
            else:
                buffer_list.append(cur_buffer)

        speech = torch.tensor(np.array(buffer_list)).to(getattr(torch, self.dtype))
        buffer_frames = int(frames_per_sec * buffer_len_in_secs)  # noqa
        context_frames = int(frames_per_sec * context_len_in_secs)

        valid_speech_samples = speech.size(0) * chunk_len

        unmerged = []
        for idx in range(0, speech.size(0), batch_size):
            cur_speech = speech[idx : idx + batch_size]
            cur_speech_lengths = cur_speech.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=cur_speech.size(1)
            )

            text_prev = torch.tensor([self.s2t_model.na], dtype=torch.long).repeat(
                cur_speech.size(0), 1
            )
            text_prev_lengths = text_prev.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=text_prev.size(1)
            )

            prefix = torch.tensor([lang_id, task_id], dtype=torch.long).repeat(
                cur_speech.size(0), 1
            )
            prefix_lengths = prefix.new_full(
                [cur_speech.size(0)], dtype=torch.long, fill_value=prefix.size(-1)
            )

            batch = {
                "speech": cur_speech,
                "speech_lengths": cur_speech_lengths,
                "text_prev": text_prev,
                "text_prev_lengths": text_prev_lengths,
                "prefix": prefix,
                "prefix_lengths": prefix_lengths,
            }

            # a. To device
            batch = to_device(batch, device=self.device)

            # b. Forward Encoder
            enc, enc_olens = self.s2t_model.encode(**batch)

            intermediate_outs = None
            if isinstance(enc, tuple):
                enc, intermediate_outs = enc

            # enc: (B, T, D), T is 376 in the default setup
            # The first two frames are language and task symbols
            enc = enc[:, 2:]  # (B, T', D), T'=buffer_frames-1

            # Remove left and right context
            enc = enc[:, context_frames:-context_frames]

            batched_log_p = self.ctc.log_softmax(enc).detach()  # (B, T'', V)

            unmerged.append(batched_log_p.reshape(-1, batched_log_p.size(-1)).cpu())

        lpz = torch.cat(unmerged, dim=0).numpy()  # (time, V)
        return lpz, valid_speech_samples

    def _lpz_and_length(self, speech):
        """The posteriors, and how many samples they cover.

        This model decodes padded windows, so what the posteriors cover is a
        round number of windows rather than the input length.
        """
        return self.get_lpz(speech)

    def _tokenize(self, text: str):
        """Text to token ids, as this model's preprocessing does it.

        The preprocessor of an S2T model expects a language and a task
        symbol around the text, so its pieces are called one by one instead.
        """
        text = self.preprocess_fn.text_cleaner(text)
        tokens = self.preprocess_fn.tokenizer.text2tokens(text)
        text_ints = self.preprocess_fn.token_id_converter.tokens2ids(tokens)
        return np.array(text_ints, dtype=np.int64)


def ctc_align(**kwargs):
    """Provide the scripting interface to align text to audio."""
    align_file(CTCSegmentation, **kwargs)


def get_parser():
    """Obtain an argument-parser for the script interface."""
    return build_parser("s2t", "CTC segmentation, with an OWSM-CTC model")


def main(cmd=None):
    """Parse arguments and start the alignment in ctc_align(·)."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    ctc_align(**kwargs)


if __name__ == "__main__":
    main()
