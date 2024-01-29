#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#           2020, Technische Universität München;  Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This program performs CTC segmentation to align utterances within audio files.

Inputs:
    `--data-json`:
        A json containing list of utterances and audio files
    `--model`:
        An already trained ASR model

Output:
    `--output`:
        A plain `segments` file with utterance positions in the audio files.

Selected parameters:
    `--min-window-size`:
        Minimum window size considered for a single utterance. The current default value
        should be OK in most cases. Larger values might give better results; too large
        values cause IndexErrors.
    `--subsampling-factor`:
        If the encoder sub-samples its input, the number of frames at the CTC layer is
        reduced by this factor.
    `--frame-duration`:
        This is the non-overlapping duration of a single frame in milliseconds (the
        inverse of frames per millisecond).
    `--set-blank`:
        In the rare case that the blank token has not the index 0 in the character
        dictionary, this parameter sets the index of the blank token.
    `--gratis-blank`:
        Sets the transition cost for blank tokens to zero. Useful if there are longer
        unrelated segments between segments.
    `--replace-spaces-with-blanks`:
        Spaces are replaced with blanks. Helps to model pauses between words. May
        increase length of ground truth. May lead to misaligned segments when combined
        with the option `--gratis-blank`.
"""

import json
import logging
import os
import sys

import configargparse
import torch

# imports for CTC segmentation
from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
)

# imports for inference
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.io_utils import LoadInputsAndTargets


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Align text to audio using CTC segmentation."
        "using a pre-trained speech recognition model.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Decoding config file path.")
    parser.add_argument(
        "--ngpu", type=int, default=0, help="Number of GPUs (max. 1 is supported)"
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    # task related
    parser.add_argument(
        "--data-json", type=str, help="Json of recognition data for audio and text"
    )
    parser.add_argument("--utt-text", type=str, help="Text separated into utterances")
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # ctc-segmentation related
    parser.add_argument(
        "--subsampling-factor",
        type=int,
        default=None,
        help="Subsampling factor."
        " If the encoder sub-samples its input, the number of frames at the CTC layer"
        " is reduced by this factor. For example, a BLSTMP with subsampling 1_2_2_1_1"
        " has a subsampling factor of 4.",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=None,
        help="Non-overlapping duration of a single frame in milliseconds.",
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=None,
        help="Minimum window size considered for utterance.",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=None,
        help="Maximum window size considered for utterance.",
    )
    parser.add_argument(
        "--use-dict-blank",
        type=int,
        default=None,
        help="DEPRECATED.",
    )
    parser.add_argument(
        "--set-blank",
        type=int,
        default=None,
        help="Index of model dictionary for blank token (default: 0).",
    )
    parser.add_argument(
        "--gratis-blank",
        type=int,
        default=None,
        help="Set the transition cost of the blank token to zero. Audio sections"
        " labeled with blank tokens can then be skipped without penalty. Useful"
        " if there are unrelated audio segments between utterances.",
    )
    parser.add_argument(
        "--replace-spaces-with-blanks",
        type=int,
        default=None,
        help="Fill blanks in between words to better model pauses between words."
        " Segments can be misaligned if this option is combined with --gratis-blank."
        " May increase length of ground truth.",
    )
    parser.add_argument(
        "--scoring-length",
        type=int,
        default=None,
        help="Changes partitioning length L for calculation of the confidence score.",
    )
    parser.add_argument(
        "--output",
        type=configargparse.FileType("w"),
        required=True,
        help="Output segments file",
    )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args, extra = parser.parse_known_args(args)
    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")
    # check CUDA_VISIBLE_DEVICES
    device = "cpu"
    if args.ngpu == 1:
        device = "cuda"
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
    elif args.ngpu > 1:
        logging.error("Decoding only supports ngpu=1.")
        sys.exit(1)
    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))
    # recog
    logging.info("backend = " + args.backend)
    if args.backend == "pytorch":
        ctc_align(args, device)
    else:
        raise ValueError("Only pytorch is supported.")
    sys.exit(0)


def ctc_align(args, device):
    """ESPnet-specific interface for CTC segmentation.

    Parses configuration, infers the CTC posterior probabilities,
    and then aligns start and end of utterances using CTC segmentation.
    Results are written to the output file given in the args.

    :param args: given configuration
    :param device: for inference; one of ['cuda', 'cpu']
    :return:  0 on success
    """
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        sort_in_input_length=False,
        preprocess_conf=(
            train_args.preprocess_conf
            if args.preprocess_conf is None
            else args.preprocess_conf
        ),
        preprocess_args={"train": False},
    )
    logging.info(f"Decoding device={device}")
    # Warn for nets with high memory consumption on long audio files
    if hasattr(model, "enc"):
        encoder_module = model.enc.__class__.__module__
    elif hasattr(model, "encoder"):
        encoder_module = model.encoder.__class__.__module__
    else:
        encoder_module = "Unknown"
    logging.info(f"Encoder module: {encoder_module}")
    logging.info(f"CTC module:     {model.ctc.__class__.__module__}")
    if "rnn" not in encoder_module:
        logging.warning("No BLSTM model detected; memory consumption may be high.")
    model.to(device=device).eval()
    # read audio and text json data
    with open(args.data_json, "rb") as f:
        js = json.load(f)["utts"]
    with open(args.utt_text, "r", encoding="utf-8") as f:
        lines = f.readlines()
        i = 0
        text = {}
        segment_names = {}
        for name in js.keys():
            text_per_audio = []
            segment_names_per_audio = []
            while i < len(lines) and lines[i].startswith(name):
                text_per_audio.append(lines[i][lines[i].find(" ") + 1 :])
                segment_names_per_audio.append(lines[i][: lines[i].find(" ")])
                i += 1
            text[name] = text_per_audio
            segment_names[name] = segment_names_per_audio
    # apply configuration
    config = CtcSegmentationParameters()
    subsampling_factor = 1
    frame_duration_ms = 10
    if args.subsampling_factor is not None:
        subsampling_factor = args.subsampling_factor
    if args.frame_duration is not None:
        frame_duration_ms = args.frame_duration
    # Backwards compatibility to ctc_segmentation <= 1.5.3
    if hasattr(config, "index_duration"):
        config.index_duration = frame_duration_ms * subsampling_factor / 1000
    else:
        config.subsampling_factor = subsampling_factor
        config.frame_duration_ms = frame_duration_ms
    if args.min_window_size is not None:
        config.min_window_size = args.min_window_size
    if args.max_window_size is not None:
        config.max_window_size = args.max_window_size
    config.char_list = train_args.char_list
    if args.use_dict_blank is not None:
        logging.warning(
            "The option --use-dict-blank is deprecated. If needed,"
            " use --set-blank instead."
        )
    if args.set_blank is not None:
        config.blank = args.set_blank
    if args.replace_spaces_with_blanks is not None:
        if args.replace_spaces_with_blanks:
            config.replace_spaces_with_blanks = True
        else:
            config.replace_spaces_with_blanks = False
    if args.gratis_blank:
        config.blank_transition_cost_zero = True
    if config.blank_transition_cost_zero and args.replace_spaces_with_blanks:
        logging.error(
            "Blanks are inserted between words, and also the transition cost of blank"
            " is zero. This configuration may lead to misalignments!"
        )
    if args.scoring_length is not None:
        config.score_min_mean_over_L = args.scoring_length
    logging.info(f"Frame timings: {frame_duration_ms}ms * {subsampling_factor}")
    # Iterate over audio files to decode and align
    for idx, name in enumerate(js.keys(), 1):
        logging.info("(%d/%d) Aligning " + name, idx, len(js.keys()))
        batch = [(name, js[name])]
        feat, label = load_inputs_and_targets(batch)
        feat = feat[0]
        with torch.no_grad():
            # Encode input frames
            enc_output = model.encode(torch.as_tensor(feat).to(device)).unsqueeze(0)
            # Apply ctc layer to obtain log character probabilities
            lpz = model.ctc.log_softmax(enc_output)[0].cpu().numpy()
        # Prepare the text for aligning
        ground_truth_mat, utt_begin_indices = prepare_text(config, text[name])
        # Align using CTC segmentation
        timings, char_probs, state_list = ctc_segmentation(
            config, lpz, ground_truth_mat
        )
        logging.debug(f"state_list = {state_list}")
        # Obtain list of utterances with time intervals and confidence score
        segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, text[name]
        )
        # Write to "segments" file
        for i, boundary in enumerate(segments):
            utt_segment = (
                f"{segment_names[name][i]} {name} {boundary[0]:.2f}"
                f" {boundary[1]:.2f} {boundary[2]:.9f}\n"
            )
            args.output.write(utt_segment)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
