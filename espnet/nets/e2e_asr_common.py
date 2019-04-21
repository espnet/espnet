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
    return int(idim) * out_channel  # numer of channels


class ASRInterface(object):
    """ASR Interface for ESPnet model implementation

    For chainer backend, __call__ method is needed.
    For pytorch backend, forward method is needed.
    """

    @staticmethod
    def add_arguments(parser):
        return parser

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        '''recognize x

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        pass

    def calculate_all_attentions(self, xs, ilens, ys):
        '''attention calculation

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        '''

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport
