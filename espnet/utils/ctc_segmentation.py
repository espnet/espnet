#!/usr/bin/env false
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#           2020, Technische Universität München, Authors: Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
    ctc_segmentation.py
This file contains the core functions of CTC segmentation
to extract utterance alignments within an audio file with a given transcription.
For a description, see https://arxiv.org/abs/2007.09127
"""

import logging
import numpy as np
import pyximport
import json
import torch
# imports for inference
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.io_utils import LoadInputsAndTargets

# import for table of character probabilities mapped to time
pyximport.install(setup_args={"include_dirs": np.get_include()}, build_dir="build", build_in_temp=False)
from espnet.utils.ctc_segmentation_dyn import cython_fill_table


class CtcSegmentationParameters:
    """
    Default values for CTC segmentation.
    May need adjustment according to localization or ASR settings.
    """
    max_prob = -10000000000.0
    skip_prob = -10000000000.0
    min_window_size = 8000
    max_window_size = 100000
    subsampling_factor = 1
    frame_duration_ms = 10  # the non-overlapping duration
    score_min_mean_over_L = 30
    space = " "
    underscore = "▁"
    self_transition = "ε"
    start_of_ground_truth = "#"
    excluded_characters = [".", ",", "-", "?", "!", ":", "»", "«", ";", "'", "›", "‹", "(", ")"]

    @property
    def index_duration_in_seconds(self):
        return self.frame_duration_ms * self.subsampling_factor / 1000


def ctc_align(args, device):
    """
        Parses configuration,
        infers the CTC posterior probabilities,
        and then starts aligns start of end of utterances.
        Results are written to the output file given in the args

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
    model.to(device=device).eval()
    # read audio and text json data
    with open(args.data_json, "rb") as f:
        js = json.load(f)["utts"]
    with open(args.utt_text, "r") as f:
        lines = f.readlines()
        i = 0
        text = {}
        segment_names = {}
        for name in js.keys():
            text_per_audio = []
            segment_names_per_audio = []
            while i < len(lines) and lines[i].startswith(name):
                text_per_audio.append(lines[i][lines[i].find(" ") + 1:])
                segment_names_per_audio.append(lines[i][:lines[i].find(" ")])
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
        ground_truth_mat, utt_begin_indices = prepare_text(config, text[name], train_args.char_list)
        # Align using CTC segmentation
        timings, char_probs, char_list = ctc_segmentation(config, lpz, ground_truth_mat)
        # Obtain list of utterances with corresponding time intervals and confidence score
        segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text[name])
        # Write to "segments" file
        for i, boundary in enumerate(segments):
            utt_segment  = f"{segment_names[name][i]} {name} {boundary[0]:.2f} {boundary[1]:.2f} {boundary[2]:.9f}\n"
            args.output.write(utt_segment)
    return 0


def ctc_segmentation(config, lpz, ground_truth):
    """
    Extract utterance alignments using CTC-segmentation

    :param config: an instance of CtcSegmentationParameters
    :param lpz: probabilities obtained from CTC output
    :param ground_truth:  ground truth text in the form of a label sequence
    :return:
    """
    blank = 0
    audio_duration = lpz.shape[0] * config.index_duration_in_seconds
    logging.info(f"Alignment: {len(ground_truth)} chars to {audio_duration:.2f}s audio ({lpz.shape[0]} indices).")
    if len(ground_truth) > lpz.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_size = config.min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table of alignment probabilities
        table = np.zeros([min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32)
        table.fill(config.max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int)
        # Run actual alignment of utterances
        t, c = cython_fill_table(table, lpz.astype(np.float32), np.array(ground_truth), offsets, blank)
        logging.debug(f"Max. joint probability to align text to audio: {table[:, c].max()} at time index {t}")
        # Backtracking
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz.shape[0]])
        char_list = [''] * lpz.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate the possible transition probabilities towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = lpz[t + offsets[c], ground_truth[c, s]] if c > 0 else config.max_prob
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = max(lpz[t + offsets[c], blank], max_lpz_prob) if t > 0 else config.max_prob
                est_stay_prob = table[t, c] - table[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (offsets[c] + t) * config.index_duration_in_seconds
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_list[offsets[c] + t] = char_list[ground_truth[c, min_s]]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    char_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            logging.warning("IndexError: Backtracking was not successful, the window size might be too small.")
            window_size *= 2
            if window_size < config.max_window_size:
                logging.warning("Increasing the window size to: " + str(window_size))
                continue
            else:
                logging.error("Maximum window size reached.")
                raise
        break
    return timings, char_probs, char_list


def prepare_text(config, text, char_list):
    """
    Prepares the given text for CTC segmentation
        creates a matrix of character symbols to represent the given text
        then creates list of char indices depending on the models char list

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if ground_truth[-1] != config.space:
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace():
                if ground_truth[-1] != config.space:
                    ground_truth += config.space
            elif char in char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if ground_truth[-1] != config.space:
        ground_truth += config.space
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s:i + 1]
            span = span.replace(config.space, config.underscore)
            if span in char_list:
                ground_truth_mat[i, s] = char_list.index(span)
    return ground_truth_mat, utt_begin_indices


def determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text):
    """
        utterance-wise alignments from char-wise alignments

    :param config: an instance of CtcSegmentationParameters
    :param utt_begin_indices: list of time indices of utterance start
    :param char_probs:  character positioned probabilities obtained from backtracking
    :param timings:  mapping of time indices to seconds
    :param text: list of utterances
    :return: segments, a list of: utterance start and end in seconds, and its confidence score
    """

    def compute_time(index, align_type):
        """
        Compute start and end time of utterance.
        :param index:  frame index value
        :param align_type:  one of ["begin", "end"]
        :return: start/end time of utterance in seconds
        """
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)

    segments = []
    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start / config.index_duration_in_seconds))
        end_t = int(round(end / config.index_duration_in_seconds))
        # Compute confidence score by using the min mean probability after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t == start_t:
            min_avg = 0
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = 0
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t:t + n].mean())
        segments.append((start, end, min_avg))
    return segments

