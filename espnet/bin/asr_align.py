#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#           2020, Technische Universität München;  Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This program performs CTC segmentation to align utterances within audio files.

Inputs:
    `--data-json`: A json containing list of utterances and audio files
    `--model`: An already trained ASR model

Output:
    `--output`: A plain `segments` file with utterance positions in the audio files.

Selected parameters:
    `--min-window-size`: Minimum window size considered for a single utterance. The
        current default value should be OK in most cases. Larger values might
        give better results; too large values cause IndexErrors.
    `--subsampling-factor`: If the encoder sub-samples its input, the number of
        frames at the CTC layer is reduced by this factor.
    `--frame-duration`: This is the non-overlapping duration of a single frame in
        milliseconds (the inverse of frames per millisecond).
    `--use-dict-blank`: Use the Blank character from the model. Useful if in the
        model dictionary e.g. "<blank>" instead of the default "_" is used.
"""

import configargparse
import logging
import os
import sys

# imports for inference
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.io_utils import LoadInputsAndTargets
import json
import torch

# imports for CTC segmentation
from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text


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
        "If the encoder sub-samples its input, the number of frames at the CTC layer is"
        " reduced by this factor. For example, a BLSTMP with subsampling 1_2_2_1_1"
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
        help="Use the Blank character of the model dictionary.",
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
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
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
    if args.subsampling_factor is not None:
        config.subsampling_factor = args.subsampling_factor
    if args.frame_duration is not None:
        config.frame_duration_ms = args.frame_duration
    if args.min_window_size is not None:
        config.min_window_size = args.min_window_size
    if args.max_window_size is not None:
        config.max_window_size = args.max_window_size
    char_list = train_args.char_list
    if args.use_dict_blank:
        config.blank = char_list[0]
        logging.info(f"Blank char was set to >{config.blank}<")
    else:
        logging.info(f"Blank char >{config.blank}< (align) >{char_list[0]}< (model)")
        if config.blank != char_list[0]:
            logging.error("Blank char mismatch; this can result in an IndexError.")
            logging.error("Pass the parameter --use-dict-blank to asr_align.py")
    logging.info(
        f"Frame timings: {config.frame_duration_ms}ms * {config.subsampling_factor}"
    )
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
        ground_truth_mat, utt_begin_indices = prepare_text(
            config, text[name], char_list
        )
        # Align using CTC segmentation
        timings, char_probs, state_list = ctc_segmentation(
            config, lpz, ground_truth_mat
        )
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
