#!/usr/bin/env python3

# Copyright 2021, Ludwig Kürzinger; Kamo Naoyuki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC segmentation to align utterances within audio files."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import TextIO
from typing import Union

import numpy as np
import soundfile
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

# imports for inference
from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

# imports for CTC segmentation
from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text
from ctc_segmentation import prepare_tokenized_text


class CTCSegmentationResult:
    """Result object for CTC segmentation.

    When formatted with str(·), this object returns
    results in a kaldi-style segments file formatting.
    The human-readable output can be configured with
    the printing options.

    Properties:
        text: Utterance texts, separated by line. But without the utterance
            name at the beginning of the line (as in kaldi-style text).
        ground_truth_mat: Ground truth matrix (CTC segmentation).
        utt_begin_indices: Utterance separator for the Ground truth matrix.
        timings: Time marks of the corresponding chars.
        state_list: Estimated alignment of chars/tokens.
        segments: Calculated segments as: (start, end, confidence score).
        config: CTC Segmentation configuration object.
        name: Name of aligned audio file (Optional). If given, name is
            considered when generating the text.
        utt_ids: The list of utterance names (Optional). This list should
            have the same length as the number of utterances.
        lpz: CTC posterior log probabilities (Optional).

    Properties for printing:
        print_confidence_score: Includes the confidence score.
        print_utterance_text: Includes utterance text.

    """

    text = None
    ground_truth_mat = None
    utt_begin_indices = None
    timings = None
    char_probs = None
    state_list = None
    segments = None
    config = None
    # Optional
    name = "utt"
    utt_ids = None
    lpz = None
    # Printing
    print_confidence_score = True
    print_utterance_text = True

    def __init__(self, **kwargs):
        """Initialize the module."""
        self.set(**kwargs)

    def set(self, **kwargs):
        """Update CTCSegmentationResult properties.

        Args:
            **kwargs: Key-value dict that contains all properties
                with their new values. Unknown properties are ignored.
        """
        for key in kwargs:
            if key.startswith("_"):
                raise ValueError(f"Don't touch {key}!")
            if hasattr(self, key) and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def __str__(self):
        """Return a kaldi-style ``segments`` file (string)."""
        output = ""
        num_utts = len(self.segments)
        if self.utt_ids is None:
            utt_names = [f"{self.name}_{i:04}" for i in range(num_utts)]
        else:
            # ensure correct mapping of segments to utterance ids
            assert num_utts == len(self.utt_ids)
            utt_names = self.utt_ids
        for i, boundary in enumerate(self.segments):
            # utterance name and file name
            utt_entry = f"{utt_names[i]} {self.name} "
            # segment start and end
            utt_entry += f"{boundary[0]:.2f} {boundary[1]:.2f}"
            # confidence score
            if self.print_confidence_score:
                utt_entry += f" {boundary[2]:3.4f}"
            # utterance ground truth
            if self.print_utterance_text:
                utt_entry += f" {self.text[i]}"
            output += utt_entry + "\n"
        return output


class CTCSegmentation:
    """Align text to audio using CTC segmentation.

    Usage:
        Initialize with given ASR model.
        If needed, parameters for CTC segmentation can be set with ``set_config(·)``.

    Parameters:
        fs: Sample rate.
        ngpu: Number of GPUs. Set 0 for processing on CPU, set to 1 for
            processing on GPU. Multi-GPU aligning is currentlly not
            implemented. Default: 0.
        dtype: Set dtype according to the ASR model.
        kaldi_style_text: A kaldi-style text file includes the name of the
            utterance at the start of the line. Set this option to True if
            this is the case with your input data. Default: False.
        tokenized_text: If given text is tokenized. Default: False.

    Selected CTC segmentation parameters:
        gratis_blank: If True, the transition cost of blank is set to zero.
            Useful for long preambles or if there are large unrelated segments
            between utterances. Default: False.
        replace_spaces_with_blanks: Inserts blanks between words, which is
            useful for handling long pauses between words. Default: False.
        min_window_size: Minimum number of frames considered for a single
            utterance. The current default value of 8000 should be OK in most
            cases. If your utterances are further apart, increase this value,
            or decrease it for smaller audio files.
        set_blank: Index of blank in token list. Default: 0.
        scoring_length: Block length to calculate confidence score. The
            default value of 30 should be OK in most cases.

    Examples:
        >>> # example file included in the ESPnet repository
        >>> import soundfile
        >>> speech, fs = soundfile.read("test_utils/ctc_align_test.wav")
        >>> # load an ASR model
        >>> from espnet_model_zoo.downloader import ModelDownloader
        >>> d = ModelDownloader()
        >>> wsjmodel = d.download_and_unpack( "kamo-naoyuki/wsj" )
        >>> # Apply CTC segmentation
        >>> aligner = CTCSegmentation( **wsjmodel )
        >>> text=["THE SALE OF THE HOTELS", "ON PROPERTY MANAGEMENT"]
        >>> aligner.set_config( gratis_blank=True )
        >>> segments = aligner( speech, text, fs=fs )
        >>> print( segments )
        utt_0000 utt 0.27 1.70 -5.4395 THE SALE OF THE HOTELS
        utt_0001 utt 4.52 6.10 -9.1153 ON PROPERTY MANAGEMENT

    """

    samples_to_frames_ratio = None
    warned_about_misconfiguration = False
    config = CtcSegmentationParameters()

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        fs: int = 16000,
        ngpu: int = 0,
        batch_size: int = 1,
        dtype: str = "float32",
        kaldi_style_text: bool = False,
        tokenized_text: bool = False,
        **ctc_segmentation_args,
    ):
        assert check_argument_types()

        # Basic settings
        if batch_size > 1:
            raise NotImplementedError("Batch decoding is not implemented")
        device = "cpu"
        if ngpu == 1:
            device = "cuda"
        elif ngpu > 1:
            logging.error("Multi-GPU not yet implemented.")
            raise NotImplementedError("Only single GPU decoding is supported")

        # Prepare ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

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

        tokenized_text_choices = {False: prepare_text, True: prepare_tokenized_text}
        self._prepare_text = tokenized_text_choices[tokenized_text]
        self.kaldi_style_text = bool(kaldi_style_text)
        self.token_list = asr_model.token_list
        # Apply configuration
        self.set_config(fs=fs, **ctc_segmentation_args)
        # last token "<sos/eos>", not needed
        self.config.char_list = asr_model.token_list[:-1]

    def set_config(self, **kwargs):
        """Set CTC segmentation parameters."""
        if "fs" in kwargs:
            assert isinstance(kwargs["fs"], int)
            self.config.fs = kwargs["fs"]
            self.config.subsampling_factor = self.determine_sample_to_encoded_ratio()
        if "subsampling_factor" in kwargs:
            logging.warning("subsampling_factor is deprecated. Use fs instead.")
            assert isinstance(kwargs["subsampling_factor"], int)
            self.config.subsampling_factor = kwargs["subsampling_factor"]
        if "frame_duration" in kwargs:
            logging.warning("frame_duration is deprecated. Use fs instead.")
            assert isinstance(kwargs["frame_duration"], int)
            self.config.frame_duration_ms = kwargs["frame_duration"]
        if "min_window_size" in kwargs:
            assert isinstance(kwargs["min_window_size"], int)
            self.config.min_window_size = kwargs["min_window_size"]
        if "max_window_size" in kwargs:
            assert isinstance(kwargs["max_window_size"], int)
            self.config.max_window_size = kwargs["max_window_size"]
        if "set_blank" in kwargs:
            assert isinstance(kwargs["set_blank"], int)
            self.config.blank = kwargs["set_blank"]
        if "scoring_length" in kwargs:
            assert isinstance(kwargs["scoring_length"], int)
            self.config.score_min_mean_over_L = kwargs["scoring_length"]
        if "replace_spaces_with_blanks" in kwargs:
            self.config.replace_spaces_with_blanks = bool(
                kwargs["replace_spaces_with_blanks"]
            )
        if "gratis_blank" in kwargs:
            self.config.blank_transition_cost_zero = bool(kwargs["gratis_blank"])
        if (
            self.config.blank_transition_cost_zero
            and self.config.replace_spaces_with_blanks
            and not self.warned_about_misconfiguration
        ):
            logging.error(
                "Blanks are inserted between words, and also the transition cost of"
                " blank is zero. This configuration may lead to misalignments!"
            )
            self.warned_about_misconfiguration = True

    def determine_sample_to_encoded_ratio(self):
        """Determine the ratio of encoded frames to sample points.

        This method helps to determine the time a single encoded frame occupies.
        As the sample rate already gave the number of samples, only the ratio
        of samples per encoded frame are needed. This function estimates them by
        doing one inference, which is only needed once.
        """
        if self.samples_to_frames_ratio is None:
            audio_len = 2048 * 3 * 5 * 7
            random_input = torch.rand(audio_len)
            lpz = self.get_lpz(random_input)
            encoder_out_len = lpz.shape[0]
            # Most frontends (DefaultFrontend, SlidingWindow) discard trailing data
            encoder_out_len = encoder_out_len + 1
            self.samples_to_frames_ratio = audio_len // encoder_out_len
        return self.samples_to_frames_ratio

    @torch.no_grad()
    def get_lpz(self, speech: torch.Tensor):
        """Obtain CTC posterior log probabilities for given speech data."""
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        batch = to_device(batch, device=self.device)
        # Encode input
        enc, _ = self.asr_model.encode(**batch)
        assert len(enc) == 1, len(enc)
        # Apply ctc layer to obtain log character probabilities
        lpz = self.ctc.log_softmax(enc.unsqueeze(0)).detach()
        #  Shape should be ( <time steps>, <classes> )
        lpz = lpz.squeeze(0).squeeze(0).cpu().numpy()
        return lpz

    def get_segments(self, text, lpz):
        """Obtain segments for given utterance texts and CTC log posteriors."""
        utt_ids = None
        # handle multiline strings
        if isinstance(text, str):
            text = text.splitlines()
        # remove empty lines
        text = list(filter(len, text))
        # handle kaldi-style text format
        if self.kaldi_style_text:
            utt_ids = [utt.split(" ", 1)[0] for utt in text]
            text = [utt.split(" ", 1)[1] for utt in text]
        # Prepare the text for aligning
        ground_truth_mat, utt_begin_indices = self._prepare_text(self.config, text)
        # Align using CTC segmentation
        timings, char_probs, state_list = ctc_segmentation(
            self.config, lpz, ground_truth_mat
        )
        # Obtain list of utterances with time intervals and confidence score
        segments = determine_utterance_segments(
            self.config, utt_begin_indices, char_probs, timings, text
        )
        result = CTCSegmentationResult()
        result.set(
            config=self.config,
            text=text,
            ground_truth_mat=ground_truth_mat,
            utt_begin_indices=utt_begin_indices,
            timings=timings,
            char_probs=char_probs,
            state_list=state_list,
            segments=segments,
            utt_ids=utt_ids,
            lpz=lpz,
        )
        return result

    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text: Union[List[str], str],
        fs: Optional[int] = None,
        name: Optional[str] = None,
    ) -> CTCSegmentationResult:
        """Align utterances.

        Args:
            speech: Audio file.
            text: List or multiline-string with utterance ground truths.
            fs: Sample rate in Hz. Optional, as this can be given when
                the module is initialized.
            name: Name of the file. Utterance names are derived from it.

        Returns:
            CTCSegmentationResult object with segments.
        """
        assert check_argument_types()
        if fs is not None:
            self.set_config(fs=fs)
        # Get log CTC posterior probabilities
        lpz = self.get_lpz(speech)
        # Apply CTC segmentation
        result = self.get_segments(text, lpz)
        if name is not None:
            result.name = name
        assert check_return_type(result)
        return result


def ctc_align(
    log_level: Union[int, str],
    asr_train_config: str,
    asr_model_file: str,
    audio: Path,
    text: TextIO,
    output: TextIO,
    print_utt_text: bool = True,
    print_utt_score: bool = True,
    **kwargs,
):
    assert check_argument_types()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Ignore configuration values that are set to None (from parser).
    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

    # Prepare CTC segmentation
    model = {
        "asr_train_config": asr_train_config,
        "asr_model_file": asr_model_file,
    }
    aligner = CTCSegmentation(**model, **kwargs)

    # load audio file
    assert audio.name != ""
    name = audio.stem
    speech, fs = soundfile.read(str(audio))
    # load text file
    transcripts = text.read()

    aligned = aligner(speech=speech, text=transcripts, fs=fs, name=name)
    # Write to "segments" file
    aligned.print_utterance_text = print_utt_text
    aligned.print_confidence_score = print_utt_score
    segments_str = str(aligned)
    output.write(segments_str)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
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

    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    group = parser.add_argument_group("Model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    group = parser.add_argument_group("CTC segmentation related")
    group.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling Frequency."
        " The sampling frequency (in Hz) is needed to correctly determine the"
        " starting and ending time of aligned segments.",
    )
    group.add_argument(
        "--min_window_size",
        type=int,
        default=None,
        help="Minimum window size considered for utterance.",
    )
    group.add_argument(
        "--max_window_size",
        type=int,
        default=None,
        help="Maximum window size considered for utterance.",
    )
    group.add_argument(
        "--set_blank",
        type=int,
        default=None,
        help="Index of model dictionary for blank token.",
    )
    group.add_argument(
        "--gratis_blank",
        type=str2bool,
        default=False,
        help="Set the transition cost of the blank token to zero. Audio sections"
        " labeled with blank tokens can then be skipped without penalty. Useful"
        " if there are unrelated audio segments between utterances.",
    )
    group.add_argument(
        "--replace_spaces_with_blanks",
        type=str2bool,
        default=False,
        help="Fill blanks in between words to better model pauses between words."
        " Segments can be misaligned if this option is combined with"
        " --gratis-blank.",
    )
    group.add_argument(
        "--scoring_length",
        type=int,
        default=None,
        help="Changes partitioning length L for calculation of the confidence score.",
    )

    group = parser.add_argument_group("Input/output arguments")
    group.add_argument(
        "--kaldi_style_text",
        type=str2bool,
        default=False,
        help="Assume that the input text file is kaldi-style formatted, i.e., the"
        " utterance name is at the beginning of each line.",
    )
    group.add_argument(
        "--print_utt_text",
        type=str2bool,
        default=True,
        help="Include the utterance text in the segments output.",
    )
    group.add_argument(
        "--print_utt_score",
        type=str2bool,
        default=True,
        help="Include the confidence score in the segments output.",
    )
    group.add_argument(
        "-a",
        "--audio",
        type=Path,
        required=True,
        help="Input audio file.",
    )
    group.add_argument(
        "-t",
        "--text",
        type=argparse.FileType("r"),
        required=True,
        help="Input text file."
        " Each line contains the ground truth of a single utterance."
        " Kaldi-style text files include the name of the utterance as"
        " the first word in the line.",
    )
    group.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default="-",
        help="Output in the form of a `segments` file."
        " If not given, output is written to stdout.",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    ctc_align(**kwargs)


if __name__ == "__main__":
    main()
