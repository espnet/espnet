#!/usr/bin/env false
# encoding: utf-8
"""
2020, Technische Universität München, Authors: Dominik Winkelbauer, Ludwig Kürzinger

This file is part of CTC segmentation to extract utterance alignments within an audio file with a given transcription.
For a description, see https://arxiv.org/abs/2007.09127
"""

import logging
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, build_dir="build", build_in_temp=False)
from espnet.utils.align_core_dyn import cython_fill_table

max_prob = -10000000000.0
min_window_size = 8000
max_window_size = 100000
excluded_characters = [".", ",", "-", "?", "!", ":", "»", "«", ";", "'", "›", "‹", "(", ")"]

def align(lpz, char_list, ground_truth, utt_begin_indices, skip_prob):
    """
    Extract utterance alignments using CTC-segmentation

    :param lpz:
    :param char_list:
    :param ground_truth:
    :param utt_begin_indices:
    :param skip_prob:
    :return:
    """
    blank = 0
    logging.info("Audio length: " + str(lpz.shape[0]))
    logging.info("Text length: " + str(len(ground_truth)))
    if len(ground_truth) > lpz.shape[0] and skip_prob <= max_prob:
        raise AssertionError("Audio is shorter than text!")
    window_size = min_window_size
    # Try multiple window lengths if it fails
    while True:
        # Create table which will contain alignment probabilities
        table = np.zeros([min(window_size, lpz.shape[0]), len(ground_truth)], dtype=np.float32)
        table.fill(max_prob)
        # Use array to log window offsets per character
        offsets = np.zeros([len(ground_truth)], dtype=np.int)
        # Run actual alignment of utterances
        t, c = cython_fill_table(table, lpz.astype(np.float32), np.array(ground_truth), offsets,
                                 np.array(utt_begin_indices), blank, skip_prob)
        logging.info("Max prob: " + str(table[:, c].max()) + " at " + str(t))
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
                max_lpz_prob = max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = lpz[t + offsets[c], ground_truth[c, s]] if c > 0 else max_prob
                        est_switch_prob = table[t, c] - table[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = max(lpz[t + offsets[c], blank], max_lpz_prob) if t > 0 else max_prob
                est_stay_prob = table[t, c] - table[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (offsets[c] + t) * 10 * 4 / 1000
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_list[offsets[c] + t] = char_list[ground_truth[c, min_s]]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    char_list[offsets[c] + t] = "ε"
                    t -= 1
        except IndexError:
            # If the backtracking was not successful, this usually means the window was too small
            window_size *= 2
            logging.warning("IndexError: Increasing the window size to: " + str(window_size))
            if window_size < max_window_size:
                continue
            else:
                raise
        break
    return timings, char_probs, char_list


def prepare_text(text, char_list):
    """
    Prepares the given text for alignment
    Therefore we create a matrix of possible character symbols to represent the given text
    Create list of char indices depending on the models char list

    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return:
    """
    ground_truth = "#"
    utt_begin_indices = []
    for utt in text:
        # Only one space in-between
        if ground_truth[-1] != " ":
            ground_truth += " "
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace():
                if ground_truth[-1] != " ":
                    ground_truth += " "
            elif char in char_list and char not in excluded_characters:
                ground_truth += char
    # Add space to the end
    if ground_truth[-1] != " ":
        ground_truth += " "
    utt_begin_indices.append(len(ground_truth) - 1)

    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s:i + 1]
            span = span.replace(" ", '▁')
            if span in char_list:
                ground_truth_mat[i, s] = char_list.index(span)

    return ground_truth_mat, utt_begin_indices


def compute_utterance_boundaries(utt_begin_indices, char_probs, timings, text):
    """
        utterance-wise alignments from char-wise alignments
        the output are the segments
        result is written into given file

    :param utt_begin_indices: frame index position of utterance start
    :param char_probs:
    :param timings:
    :param text:
    :return: segments a list of: utterance start and end in seconds, and its segmentation score
    """
    boundaries = []
    def compute_time(index, type):
        # Compute start and end time of utterance.
        middle = (timings[index] + timings[index - 1]) / 2
        if type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif type == "end":
            return min(timings[index - 1] + 0.5, middle)

    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start * 1000 / 40))
        end_t = int(round(end * 1000 / 40))
        # Compute confidence score by using the min mean probability after splitting into segments of 30 frames
        n = 30
        if end_t == start_t:
            min_avg = 0
        elif end_t - start_t <= n:
            min_avg = char_probs[start_t:end_t].mean()
        else:
            min_avg = 0
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t:t + n].mean())
        boundaries.append((start, end, min_avg))
    return boundaries


def align_run(enc, text, char_list):
    """

    :param enc:
    :param text:
    :param char_list:
    :return:
    """
    ground_truth_mat, utt_begin_indices = prepare_text(text, char_list)
    char_list = char_list[:]
    char_list[0] = "ε"
    char_list = [c.lower() for c in char_list]
    timings, char_probs, char_list = align(enc, char_list, ground_truth_mat, utt_begin_indices, max_prob)
    return compute_utterance_boundaries(utt_begin_indices, char_probs, timings, text)
