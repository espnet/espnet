#!/usr/bin/env python3
# Copyright 2021, Ludwig Kürzinger; Kamo Naoyuki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC segmentation to align utterances within audio files.

The four alignment modules in espnet2.bin, and what each is::

    align.py        forced alignment over any CTC head; what `espnet align`
                    and the MCP server run
    asr_align.py    CTC segmentation with an ASR model, and its script
    s2t_align.py    CTC segmentation with an OWSM-CTC model, and its script
    ctc_segment.py  the CTC segmentation algorithm those two share; not an
                    entry point

The algorithm is `espnet2.bin.ctc_segment`, shared with
`espnet2.bin.s2t_align`. What is here is how an ASR model produces the CTC
posteriors, and this module is what an ASR model is aligned from::

    from espnet2.bin.asr_align import CTCSegmentation
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from typeguard import typechecked

# CTCSegmentationTask is re-exported: it has been importable from this
# module since 2021, and the tests and recipes that use it say so.
from espnet2.bin.ctc_segment import (  # noqa: F401
    AbsCTCSegmentation,
    CTCSegmentationTask,
    align_file,
    build_parser,
)
from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device


class CTCSegmentation(AbsCTCSegmentation):
    """Align text to audio with an ASR model, using CTC segmentation.

    Usage:
        Initialize with given ASR model and parameters.
        If needed, parameters for CTC segmentation can be set with ``set_config(·)``.
        Then call the instance as function to align text within an audio file.

    Example:
        >>> # example file included in the ESPnet repository
        >>> import soundfile
        >>> speech, fs = soundfile.read("test_utils/ctc_align_test.wav")
        >>> # an ASR model, by the tag its model card gives
        >>> aligner = CTCSegmentation.from_pretrained(
        ...     "espnet/kamo-naoyuki_wsj_transformer2"
        ... )
        >>> text=["utt1 THE SALE OF THE HOTELS", "utt2 ON PROPERTY MANAGEMENT"]
        >>> aligner.set_config( gratis_blank=True )
        >>> segments = aligner( speech, text, fs=fs )
        >>> print( segments )
        utt1 utt 0.27 1.72 -0.1663 THE SALE OF THE HOTELS
        utt2 utt 4.54 6.10 -4.9646 ON PROPERTY MANAGEMENT

    The algorithm, the parameters it takes and how to split it across
    processes are documented in `espnet2.bin.ctc_segment`.

    """

    @typechecked
    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str, None] = None,
        fs: int = 16000,
        ngpu: int = 0,
        batch_size: int = 1,
        dtype: str = "float32",
        kaldi_style_text: bool = True,
        text_converter: str = "tokenize",
        time_stamps: str = "auto",
        # last, and after every argument this class already had: a caller
        # passing them positionally must keep binding what it bound before
        device: Optional[str] = None,
        **ctc_segmentation_args,
    ):
        """Initialize the CTCSegmentation module.

        Args:
            asr_train_config: ASR model config file (yaml).
            asr_model_file: ASR model file (pth).
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
        if batch_size > 1:
            raise NotImplementedError("Batch decoding is not implemented")
        device = self._resolve_device(device, ngpu)

        # Prepare ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()
        self.preprocess_fn = ASRTask.build_preprocess_fn(asr_train_args, False)

        # Warn for nets with high memory consumption on long audio files
        if hasattr(asr_model, "encoder"):
            encoder_module = asr_model.encoder.__class__.__module__
        else:
            encoder_module = "Unknown"
        logging.info(f"Encoder module: {encoder_module}")
        logging.info(f"CTC module:     {asr_model.ctc.__class__.__module__}")
        if "rnn" not in encoder_module.lower():
            logging.warning("No RNN model detected; memory consumption may be high.")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.ctc = asr_model.ctc

        self.kaldi_style_text = kaldi_style_text
        self.token_list = asr_model.token_list
        # Apply configuration
        self.set_config(
            fs=fs,
            time_stamps=time_stamps,
            kaldi_style_text=kaldi_style_text,
            text_converter=text_converter,
            **ctc_segmentation_args,
        )
        # last token "<sos/eos>", not needed
        self.config.char_list = asr_model.token_list[:-1]

    @torch.no_grad()
    def get_lpz(self, speech: Union[torch.Tensor, np.ndarray]):
        """Obtain CTC posterior log probabilities for given speech data.

        Args:
            speech: Speech audio input.

        Returns:
            lpz: Numpy vector with CTC log posterior probabilities.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        batch = to_device(batch, device=self.device)
        # Encode input
        enc, _ = self.asr_model.encode(**batch)
        assert len(enc) == 1, len(enc)
        # Apply ctc layer to obtain log character probabilities
        lpz = self.ctc.log_softmax(enc).detach()
        #  Shape should be ( <time steps>, <classes> )
        lpz = lpz.squeeze(0).cpu().numpy()
        return lpz


def ctc_align(**kwargs):
    """Provide the scripting interface to align text to audio."""
    align_file(CTCSegmentation, **kwargs)


def get_parser():
    """Obtain an argument-parser for the script interface."""
    return build_parser("asr", "CTC segmentation, with an ASR model")


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
