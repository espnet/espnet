#!/usr/bin/env false
# encoding: utf-8

# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CTC segmentation.

This file is part of CTC segmentation to extract utterance alignments
within an audio file using dynamic programming.
For a description, see https://arxiv.org/abs/2007.09127
"""

import logging
import numpy as np
cimport numpy as np


def cython_fill_table(np.ndarray[np.float32_t, ndim=2] table,
                      np.ndarray[np.float32_t, ndim=2] lpz,
                      np.ndarray[np.int_t, ndim=2] ground_truth,
                      np.ndarray[np.int_t, ndim=1] offsets,
                      int blank):
    """Fill the table of transition probabilities.

    :param table: table filled with maximum joint probabilities k_{t,j}
    :param lpz: character probabilities of each time frame
    :param ground_truth: label sequence
    :param offsets: window offsets per character (given as array of zeros)
    :param blank: label ID of the blank symbol, usually 0
    :return:
    """
    cdef int c
    cdef int t
    cdef int offset = 0
    cdef float mean_offset
    cdef int offset_sum = 0
    cdef int lower_offset
    cdef int higher_offset
    cdef float switch_prob, stay_prob, skip_prob
    cdef float prob_max = -1000000000
    cdef float lastMax
    cdef int lastArgMax
    cdef np.ndarray[np.int_t, ndim=1] cur_offset = np.zeros([ground_truth.shape[1]], np.int) - 1
    cdef float max_lpz_prob
    cdef float p
    cdef int s

    # Compute the mean offset between two window positions
    mean_offset = (lpz.shape[0] - table.shape[0]) / float(table.shape[1])
    logging.debug(f"Average character duration: {mean_offset} (indices)")
    lower_offset = int(mean_offset)
    higher_offset = lower_offset + 1
    # calculation of the trellis diagram table
    table[0, 0] = 0
    for c in range(table.shape[1]):
        if c > 0:
            # Compute next window offset
            offset = min(max(0, last_arg_max - table.shape[0] // 2),
                         min(higher_offset, (lpz.shape[0] - table.shape[0]) - offset_sum))
            # Compute relative offset to previous columns
            for s in range(ground_truth.shape[1] - 1):
                cur_offset[s + 1] = cur_offset[s] + offset
            cur_offset[0] = offset
            # Apply offset and move window one step further
            offset_sum += offset
        # Log offset
        offsets[c] = offset_sum
        last_arg_max = -1
        last_max = 0
        # Go through all rows of the current column
        for t in range((1 if c == 0 else 0), table.shape[0]):
            # Compute max switch probability
            switch_prob = prob_max
            max_lpz_prob = prob_max
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] != -1:
                    if t >= table.shape[0] - (cur_offset[s] - 1) or t - 1 + cur_offset[s] < 0 or c == 0:
                        p = prob_max
                    else:
                        p = table[t - 1 + cur_offset[s], c - (s + 1)] + lpz[t + offset_sum, ground_truth[c, s]]
                    switch_prob = max(switch_prob, p)
                    max_lpz_prob = max(max_lpz_prob, lpz[t + offset_sum, ground_truth[c, s]])
            # Compute stay probability
            if t - 1 < 0:
                stay_prob = prob_max
            elif c == 0:
                stay_prob = 0
            else:
                stay_prob = table[t - 1, c] + max(lpz[t + offset_sum, blank], max_lpz_prob)
            # Use max of stay and switch prob
            table[t, c] = max(switch_prob, stay_prob)
            # Remember the row with the max prob
            if last_arg_max == -1 or last_max < table[t, c]:
                last_max = table[t, c]
                last_arg_max = t
    # Return cell index with max prob in last column
    c = table.shape[1] - 1
    t = table[:, c].argmax()
    return t, c
