#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import logging
import numpy as np
import six
import sys


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


# TODO(takaaki-hori): add different smoothing methods
def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    """Obtain label distribution for loss smoothing

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    """
    if transcript is not None:
        with open(transcript, 'rb') as f:
            trans_json = json.load(f)['utts']

    if lsm_type == 'unigram':
        assert transcript is not None, 'transcript is required for %s label smoothing' % lsm_type
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v['output'][0]['tokenid'].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error(
            "Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist


def get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # number of channels


def expand_elayers(elayers, etype, warn=False):
    """Expands the elayers representation and return the corrected etype if necessary

    The elayers string is formatted as a sequence of "count"x"units-dropout"_"proj_units-dropout_proj" or simply "units"
    separated by commas
    Examples : 6x300-0.2_300-0.4 ; 3x500_300-0.5,2x700 ; 300_200,300,300_200,300,300 ; 3x500,300,500 ; ...

    :param str elayers: The layers configuration
    :param str etype: The chosen etype
    :param bool warn: Whether to warn the user if incompatible etypes or not
    :rtype: tuple[list[tuple[int,float,int,float]],str]
    :return: (expanded layers, new_etype)
    """
    expanded_elayers = []
    layers_group = elayers.split(",")
    # Split on layers
    for layers in layers_group:
        layer_proj = layers.strip().split("_")
        layer_tuple = layer_proj[0].strip().split("x")
        # Check if count exists
        if len(layer_tuple) > 1:
            repetitions = int(layer_tuple[0])
            units_dropout = layer_tuple[1].strip().split("-")
            units = int(units_dropout[0])
            # Check if layer dropout exists
            if len(units_dropout) > 1:
                dropout = float(units_dropout[1])
            else:
                dropout = 0.0
        else:
            repetitions = 1
            units = int(layer_tuple[0])
            dropout = 0.0
        # Check if eprojs exists
        if len(layer_proj) > 1:
            proj_dropout = layer_proj[1].strip().split("-")
            proj = proj_dropout[0]
            # Check if projection dropout exists
            if len(proj_dropout) > 1:
                dropoutp = float(proj_dropout[1])
            else:
                dropoutp = 0
        else:
            proj = units
            dropoutp = 0

        expanded_elayers.extend(repetitions * [(units, dropout, proj, dropoutp)])

    all_same = len(set(expanded_elayers)) == 1
    if not etype.endswith('p') and not all_same:
        etype = etype + 'p'
        if warn:
            logging.warning("Adding every-layer projection to encoder due to different encoder layers")
    return expanded_elayers, etype
