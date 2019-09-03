#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility funcitons for the text translation task."""

import logging


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text string
    :return: recognition token string
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    if 'utt2spk' in js.keys():
        new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js['output']) > 0:
            out_dic = dict(js['output'][0].items())
        else:
            out_dic = {'name': ''}

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add source reference
        out_dic['text_src'] = js['output'][1]['text']
        out_dic['token_src'] = js['output'][1]['token']
        out_dic['tokenid_src'] = js['output'][1]['tokenid']

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])
            logging.info('source : %s' % out_dic['token_src'])

    return new_js
