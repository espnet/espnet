#!/usr/bin/env false
# encoding: utf-8

# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test functions for CTC segmentation."""
import numpy as np
import torch

from ctc_segmentation import ctc_segmentation
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import determine_utterance_segments
from ctc_segmentation import prepare_text


def test_ctc_segmentation():
    """Test CTC segmentation.

    This is a minimal example for the function.
    Only executes CTC segmentation, does not check its result.
    """
    config = CtcSegmentationParameters()
    config.min_window_size = 10
    char_list = [config.blank, "a", "c", "d", "g", "o", "s", "t"]
    text = ["catzz#\n", "dogs!!\n"]
    lpz = torch.nn.functional.log_softmax(torch.rand(30, 8) * 10, dim=0).numpy()
    ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
    timings, char_probs, state_list = ctc_segmentation(config, lpz, ground_truth_mat)


def test_determine_utterance_segments():
    """Test the generation of segments from aligned utterances.

    This is a function that is used after a completed CTC segmentation.
    Results are checked and compared with test vectors.
    """
    config = CtcSegmentationParameters()
    config.frame_duration_ms = 1000
    config.score_min_mean_over_L = 2
    utt_begin_indices = [1, 4, 9]
    text = ["catzz#\n", "dogs!!\n"]
    char_probs = np.array([-0.5] * 10)
    timings = np.array(list(range(10))) + 0.5
    segments = determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, text
    )
    correct_segments = [(2.0, 4.0, -0.5), (5.0, 9.0, -0.5)]
    for i, boundary in enumerate(segments):
        utt_segment = f"{i} {boundary[0]:.2f} {boundary[1]:.2f} {boundary[2]:.2f}"
        print(utt_segment)
    for i in [0, 1]:
        for j in [0, 1, 2]:
            assert segments[i][j] == correct_segments[i][j]


def test_prepare_text():
    """Test the prepare_text function for CTC segmentation.

    Results are checked and compared with test vectors.
    """
    config = CtcSegmentationParameters()
    text = ["catzz#\n", "dogs!!\n"]
    char_list = [config.blank, "a", "c", "d", "g", "o", "s", "t"]
    ground_truth_mat, utt_begin_indices = prepare_text(config, text, char_list)
    correct_begin_indices = np.array([1, 5, 10])
    assert (utt_begin_indices == correct_begin_indices).all()
    gtm = list(ground_truth_mat.shape)
    assert gtm[0] == 11
    assert gtm[1] == 1
