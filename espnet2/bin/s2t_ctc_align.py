#!/usr/bin/env python3
# Copyright 2021, Ludwig Kürzinger; Kamo Naoyuki; Yifan Peng
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC segmentation to align utterances within audio files using OWSM-CTC."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, TextIO, Union

import numpy as np
import soundfile
import torch

# imports for CTC segmentation
from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
    prepare_token_list,
)
from typeguard import typechecked

from espnet2.tasks.s2t_ctc import S2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str_or_none

# imports for inference
from espnet.utils.cli_utils import get_commandline_args


class CTCSegmentationTask:
    """Task object for CTC segmentation.

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
    done = False
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
        """Update properties.

        Args:
            **kwargs: Key-value dict that contains all properties
                with their new values. Unknown properties are ignored.
        """
        for key in kwargs:
            if (
                not key.startswith("_")
                and hasattr(self, key)
                and kwargs[key] is not None
            ):
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
        Initialize with given ASR model and parameters.
        If needed, parameters for CTC segmentation can be set with ``set_config(·)``.
        Then call the instance as function to align text within an audio file.

    Example:
        >>> # example file included in the ESPnet repository
        >>> import soundfile
        >>> speech, fs = soundfile.read("test_utils/ctc_align_test.wav")
        >>> # load an ASR model
        >>> from espnet_model_zoo.downloader import ModelDownloader
        >>> d = ModelDownloader()
        >>> wsjmodel = d.download_and_unpack( "kamo-naoyuki/wsj" )
        >>> # Apply CTC segmentation
        >>> aligner = CTCSegmentation( **wsjmodel )
        >>> text=["utt1 THE SALE OF THE HOTELS", "utt2 ON PROPERTY MANAGEMENT"]
        >>> aligner.set_config( gratis_blank=True )
        >>> segments = aligner( speech, text, fs=fs )
        >>> print( segments )
        utt1 utt 0.27 1.72 -0.1663 THE SALE OF THE HOTELS
        utt2 utt 4.54 6.10 -4.9646 ON PROPERTY MANAGEMENT

    On multiprocessing:
        To parallelize the computation with multiprocessing, these three steps
        can be separated:
        (1) ``get_lpz``: obtain the lpz,
        (2) ``prepare_segmentation_task``: prepare the task, and
        (3) ``get_segments``: perform CTC segmentation.
        Note that the function `get_segments` is a staticmethod and therefore
        independent of an already initialized CTCSegmentation object.

    References:
        CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition
        2020, Kürzinger, Winkelbauer, Li, Watzel, Rigoll
        https://arxiv.org/abs/2007.09127

    More parameters are described in https://github.com/lumaku/ctc-segmentation

    """

    fs = 16000
    samples_to_frames_ratio = None
    time_stamps = "auto"
    choices_time_stamps = ["auto", "fixed"]
    text_converter = "tokenize"
    choices_text_converter = ["tokenize", "classic"]
    warned_about_misconfiguration = False
    config = CtcSegmentationParameters()

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
        **ctc_segmentation_args,
    ):
        """Initialize the CTCSegmentation module.

        Args:
            s2t_train_config: S2T model config file (yaml).
            s2t_model_file: S2T model file (pth).
            fs: Sample rate of audio file.
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
        device = "cpu"
        if ngpu == 1:
            device = "cuda"
        elif ngpu > 1:
            logging.error("Multi-GPU not yet implemented.")
            raise NotImplementedError("Only single GPU decoding is supported")

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

    def set_config(self, **kwargs):
        """Set CTC segmentation parameters.

        Parameters for timing:
            time_stamps: Select method how CTC index duration is estimated, and
                thus how the time stamps are calculated.
            fs: Sample rate.
            samples_to_frames_ratio: If you want to directly determine the
                ratio of samples to CTC frames, set this parameter, and
                set ``time_stamps`` to "fixed".
                Note: If you want to calculate the time stamps as in
                ESPnet 1, set this parameter to:
                ``subsampling_factor * frame_duration / 1000``.

        Parameters for text preparation:
            set_blank: Index of blank in token list. Default: 0.
            replace_spaces_with_blanks: Inserts blanks between words, which is
                useful for handling long pauses between words. Only used in
                ``text_converter="classic"`` preprocessing mode. Default: False.
            kaldi_style_text: Determines whether the utterance name is expected
                as fist word of the utterance. Set at module initialization.
            text_converter: How CTC segmentation handles text.
                Set at module initialization.

        Parameters for alignment:
            min_window_size: Minimum number of frames considered for a single
                utterance. The current default value of 8000 corresponds to
                roughly 4 minutes (depending on ASR model) and should be OK in
                most cases. If your utterances are further apart, increase
                this value, or decrease it for smaller audio files.
            max_window_size: Maximum window size. It should not be necessary
                to change this value.
            gratis_blank: If True, the transition cost of blank is set to zero.
                Useful for long preambles or if there are large unrelated segments
                between utterances. Default: False.

        Parameters for calculation of confidence score:
            scoring_length: Block length to calculate confidence score. The
                default value of 30 should be OK in most cases.
        """
        # Parameters for timing
        if "time_stamps" in kwargs:
            if kwargs["time_stamps"] not in self.choices_time_stamps:
                raise NotImplementedError(
                    f"Parameter ´time_stamps´ has to be one of "
                    f"{list(self.choices_time_stamps)}",
                )
            self.time_stamps = kwargs["time_stamps"]
        if "fs" in kwargs:
            self.fs = float(kwargs["fs"])
        if "samples_to_frames_ratio" in kwargs:
            self.samples_to_frames_ratio = float(kwargs["samples_to_frames_ratio"])
        # Parameters for text preparation
        if "set_blank" in kwargs:
            assert isinstance(kwargs["set_blank"], int)
            self.config.blank = kwargs["set_blank"]
        if "replace_spaces_with_blanks" in kwargs:
            self.config.replace_spaces_with_blanks = bool(
                kwargs["replace_spaces_with_blanks"]
            )
        if "kaldi_style_text" in kwargs:
            assert isinstance(kwargs["kaldi_style_text"], bool)
            self.kaldi_style_text = kwargs["kaldi_style_text"]
        if "text_converter" in kwargs:
            if kwargs["text_converter"] not in self.choices_text_converter:
                raise NotImplementedError(
                    f"Parameter ´text_converter´ has to be one of "
                    f"{list(self.choices_text_converter)}",
                )
            self.text_converter = kwargs["text_converter"]
        # Parameters for alignment
        if "min_window_size" in kwargs:
            assert isinstance(kwargs["min_window_size"], int)
            self.config.min_window_size = kwargs["min_window_size"]
        if "max_window_size" in kwargs:
            assert isinstance(kwargs["max_window_size"], int)
            self.config.max_window_size = kwargs["max_window_size"]
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
        # Parameter for calculation of confidence score
        if "scoring_length" in kwargs:
            assert isinstance(kwargs["scoring_length"], int)
            self.config.score_min_mean_over_L = kwargs["scoring_length"]

    def get_timing_config(self, speech_len=None, lpz_len=None):
        """Obtain parameters to determine time stamps."""
        timing_cfg = {
            "index_duration": self.config.index_duration,
        }
        # As the parameter ctc_index_duration vetoes the other
        if self.time_stamps == "fixed":
            index_duration = self.samples_to_frames_ratio / self.fs
        else:
            assert self.time_stamps == "auto"
            samples_to_frames_ratio = speech_len / lpz_len
            index_duration = samples_to_frames_ratio / self.fs
        timing_cfg["index_duration"] = index_duration
        return timing_cfg

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

    def _split_text(self, text):
        """Convert text to list and extract utterance IDs."""
        utt_ids = None
        # Handle multiline strings
        if isinstance(text, str):
            text = text.splitlines()
        # Remove empty lines
        text = list(filter(len, text))
        # Handle kaldi-style text format
        if self.kaldi_style_text:
            utt_ids_and_text = [utt.split(" ", 1) for utt in text]
            # remove utterances with empty text
            utt_ids_and_text = filter(lambda ui: len(ui) == 2, utt_ids_and_text)
            utt_ids_and_text = list(utt_ids_and_text)
            utt_ids = [utt[0] for utt in utt_ids_and_text]
            text = [utt[1] for utt in utt_ids_and_text]
        return utt_ids, text

    def prepare_segmentation_task(self, text, lpz, name=None, speech_len=None):
        """Preprocess text, and gather text and lpz into a task object.

        Text is pre-processed and tokenized depending on configuration.
        If ``speech_len`` is given, the timing configuration is updated.
        Text, lpz, and configuration is collected in a CTCSegmentationTask
        object. The resulting object can be serialized and passed in a
        multiprocessing computation.

        A minimal amount of text processing is done, i.e., splitting the
        utterances in ``text`` into a list and applying ``text_cleaner``.
        It is recommended that you normalize the text beforehand, e.g.,
        change numbers into their spoken equivalent word, remove special
        characters, and convert UTF-8 characters to chars corresponding to
        your ASR model dictionary.

        The text is tokenized based on the ``text_converter`` setting:

        The "tokenize" method is more efficient and the easiest for models
        based on latin or cyrillic script that only contain the main chars,
        ["a", "b", ...] or for Japanese or Chinese ASR models with ~3000
        short Kanji / Hanzi tokens.

        The "classic" method improves the the accuracy of the alignments
        for models that contain longer tokens, but with a greater complexity
        for computation. The function scans for partial tokens which may
        improve time resolution.
        For example, the word "▁really" will be broken down into
        ``['▁', '▁r', '▁re', '▁real', '▁really']``. The alignment will be
        based on the most probable activation sequence given by the network.

        Args:
            text: List or multiline-string with utterance ground truths.
            lpz: Log CTC posterior probabilities obtained from the CTC-network;
                numpy array shaped as ( <time steps>, <classes> ).
            name: Audio file name. Choose a unique name, or the original audio
                file name, to distinguish multiple audio files. Default: None.
            speech_len: Number of sample points. If given, the timing
                configuration is automatically derived from length of fs, length
                of speech and length of lpz. If None is given, make sure the
                timing parameters are correct, see time_stamps for reference!
                Default: None.

        Returns:
            task: CTCSegmentationTask object that can be passed to
                ``get_segments()`` in order to obtain alignments.
        """
        config = self.config

        # Update timing parameters, if needed
        if speech_len is not None:
            lpz_len = lpz.shape[0]
            timing_cfg = self.get_timing_config(speech_len, lpz_len)
            config.set(**timing_cfg)

        # `text` is needed in the form of a list.
        utt_ids, text = self._split_text(text)

        # Obtain utterance & label sequence from text
        if self.text_converter == "tokenize":

            def _tokenize(text):
                text = self.preprocess_fn.text_cleaner(text)
                tokens = self.preprocess_fn.tokenizer.text2tokens(text)
                text_ints = self.preprocess_fn.token_id_converter.tokens2ids(tokens)
                text_ints = np.array(text_ints, dtype=np.int64)
                return text_ints

            # list of str --tokenize--> list of np.array
            token_list = [_tokenize(utt) for utt in text]

            # filter out any instances of the <unk> token
            unk = config.char_list.index("<unk>")
            token_list = [utt[utt != unk] for utt in token_list]
            ground_truth_mat, utt_begin_indices = prepare_token_list(config, token_list)

        else:
            assert self.text_converter == "classic"
            text = [self.preprocess_fn.text_cleaner(utt) for utt in text]
            token_list = [
                "".join(self.preprocess_fn.tokenizer.text2tokens(utt)) for utt in text
            ]
            token_list = [utt.replace("<unk>", "") for utt in token_list]
            ground_truth_mat, utt_begin_indices = prepare_text(config, token_list)

        task = CTCSegmentationTask(
            config=config,
            name=name,
            text=text,
            ground_truth_mat=ground_truth_mat,
            utt_begin_indices=utt_begin_indices,
            utt_ids=utt_ids,
            lpz=lpz,
        )
        return task

    @staticmethod
    def get_segments(task: CTCSegmentationTask):
        """Obtain segments for given utterance texts and CTC log posteriors.

        Args:
            task: CTCSegmentationTask object that contains ground truth and
                CTC posterior probabilities.

        Returns:
            result: Dictionary with alignments. Combine this with the task
                object to obtain a human-readable segments representation.
        """
        assert task.config is not None
        config = task.config
        lpz = task.lpz
        ground_truth_mat = task.ground_truth_mat
        utt_begin_indices = task.utt_begin_indices
        text = task.text
        # Align using CTC segmentation
        timings, char_probs, state_list = ctc_segmentation(
            config, lpz, ground_truth_mat
        )
        # Obtain list of utterances with time intervals and confidence score
        segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, text
        )
        # Store results
        result = {
            "name": task.name,
            "timings": timings,
            "char_probs": char_probs,
            "state_list": state_list,
            "segments": segments,
            "done": True,
        }
        return result

    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        text: Union[List[str], str],
        fs: Optional[int] = None,
        name: Optional[str] = None,
    ) -> CTCSegmentationTask:
        """Align utterances.

        Args:
            speech: Audio file.
            text: List or multiline-string with utterance ground truths.
            fs: Sample rate in Hz. Optional, as this can be given when
                the module is initialized.
            name: Name of the file. Utterance names are derived from it.

        Returns:
            CTCSegmentationTask object with segments.

        """

        if fs is not None:
            self.set_config(fs=fs)
        # Get log CTC posterior probabilities
        lpz, valid_speech_samples = self.get_lpz(speech)
        # Conflate text & lpz & config as a segmentation task object
        task = self.prepare_segmentation_task(text, lpz, name, valid_speech_samples)
        # Apply CTC segmentation
        segments = self.get_segments(task)
        task.set(**segments)
        return task


@typechecked
def ctc_align(
    log_level: Union[int, str],
    s2t_train_config: str,
    s2t_model_file: str,
    audio: Path,
    text: TextIO,
    output: TextIO,
    print_utt_text: bool = True,
    print_utt_score: bool = True,
    **kwargs,
):
    """Provide the scripting interface to align text to audio."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Ignore configuration values that are set to None (from parser).
    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

    # Prepare CTC segmentation module
    model = {
        "s2t_train_config": s2t_train_config,
        "s2t_model_file": s2t_model_file,
    }
    aligner = CTCSegmentation(**model, **kwargs)

    # load audio file
    assert audio.name != ""
    name = audio.stem
    speech, fs = soundfile.read(str(audio))
    # load text file
    transcripts = text.read()

    # perform inference and CTC segmentation
    segments = aligner(speech=speech, text=transcripts, fs=fs, name=name)

    # Write to "segments" file or stdout
    segments.print_utterance_text = print_utt_text
    segments.print_confidence_score = print_utt_score
    segments_str = str(segments)
    output.write(segments_str)


def get_parser():
    """Obtain an argument-parser for the script interface."""
    parser = config_argparse.ArgumentParser(
        description="CTC alignment",
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
    group.add_argument("--s2t_train_config", type=str, required=True)
    group.add_argument("--s2t_model_file", type=str, required=True)

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
        " This option is only active for `--text_converter classic`."
        " Segments can be misaligned if this option is combined with"
        " --gratis-blank.",
    )
    group.add_argument(
        "--scoring_length",
        type=int,
        default=None,
        help="Changes partitioning length L for calculation of the confidence score.",
    )
    group.add_argument(
        "--time_stamps",
        type=str,
        default=CTCSegmentation.time_stamps,
        choices=CTCSegmentation.choices_time_stamps,
        help="Select method how CTC index duration is estimated, and"
        " thus how the time stamps are calculated.",
    )
    group.add_argument(
        "--text_converter",
        type=str,
        default=CTCSegmentation.text_converter,
        choices=CTCSegmentation.choices_text_converter,
        help="How CTC segmentation handles text.",
    )

    group = parser.add_argument_group("Input/output arguments")
    group.add_argument(
        "--kaldi_style_text",
        type=str2bool,
        default=True,
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
    """Parse arguments and start the alignment in ctc_align(·)."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    ctc_align(**kwargs)


if __name__ == "__main__":
    main()
