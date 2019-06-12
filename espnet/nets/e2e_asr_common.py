#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import editdistance
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


def calculate_cer_wer(y_hats, y_pads, char_list, sym_space, sym_blank):
    """Calculate CER and WER for E2E_ASR models during training

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
    for i, y_hat in enumerate(y_hats):
        y_true = y_pads[i]
        eos_true = np.where(y_true == -1)[0]
        eos_true = eos_true[0] if len(eos_true) > 0 else len(y_true)
        # To avoid wrong higger WER than the one obtained from the decoding
        # eos from y_true is used to mark the eos in y_hat
        # because of that y_hats has not padded outs with -1.
        seq_hat = [char_list[int(idx)] for idx in y_hat[:eos_true]]
        seq_true = [char_list[int(idx)] for idx in y_true if int(idx) != -1]
        seq_hat_text = "".join(seq_hat).replace(sym_space, ' ')
        seq_hat_text = seq_hat_text.replace(sym_blank, '')
        seq_true_text = "".join(seq_true).replace(sym_space, ' ')
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
        hyp_chars = seq_hat_text.replace(' ', '')
        ref_chars = seq_true_text.replace(' ', '')
        char_eds.append(editdistance.eval(hyp_chars, ref_chars))
        char_ref_lens.append(len(ref_chars))
    return word_eds, word_ref_lens, char_eds, char_ref_lens
