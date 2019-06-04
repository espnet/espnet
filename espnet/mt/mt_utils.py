#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import logging
# matplotlib related
import os

# chainer related
from chainer.training import extension

# io related
import matplotlib


matplotlib.use('Agg')


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param CustomConverter converter: function to convert data
    :param int | torch.device device: device
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, transform, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            att_w = self.get_attention_weight(idx, att_w)
            plot = self.draw_attention_plot(att_w)
            logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        else:
            att_ws = self.att_vis_fn(**batch)
        return att_ws

    def get_attention_weight(self, idx, att_w):
        if self.reverse:
            dec_len = int(self.data[idx][1]['output'][1]['shape'][0])
            enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
        else:
            dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['output'][1]['shape'][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

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
    """Function to add N-best results to json

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
        out_dic['text.src'] = js['output'][1]['text']
        out_dic['token.src'] = js['output'][1]['token']
        out_dic['tokenid.src'] = js['output'][1]['tokenid']

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])
            logging.info('source : %s' % out_dic['token.src'])

    return new_js
