#!/usr/bin/env python3

"""
This script is used to provide utility functions designed for multi-speaker ASR.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Most functions can be directly used as in asr_utils.py:
    CompareValueTrigger, restore_snapshot, adadelta_eps_decay, chainer_load,
    torch_snapshot, torch_save, torch_resume, AttributeDict, get_model_conf.

"""

import copy
import logging
import os

from chainer.training import extension

import matplotlib

from espnet.asr.asr_utils import parse_hypothesis


matplotlib.use("Agg")


# * -------------------- chainer extension related -------------------- *
class PlotAttentionReport(extension.Extension):
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.calculate_all_attentions):
            Function of attention visualization.
        data (list[tuple(str, dict[str, dict[str, Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter):
            CustomConverter object. Function to convert data.
        device (torch.device): The destination device to send tensor.
        reverse (bool): If True, input and output length are reversed.

    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        """Initialize PlotAttentionReport."""
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        """Plot and save imaged matrix of att_ws."""
        att_ws_sd = self.get_attention_weights()
        for ns, att_ws in enumerate(att_ws_sd):
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{.updater.epoch}.output%d.png" % (
                    self.outdir,
                    self.data[idx][0],
                    ns + 1,
                )
                att_w = self.get_attention_weight(idx, att_w, ns)
                self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        """Add image files of attention matrix to tensorboard."""
        att_ws_sd = self.get_attention_weights()
        for ns, att_ws in enumerate(att_ws_sd):
            for idx, att_w in enumerate(att_ws):
                att_w = self.get_attention_weight(idx, att_w, ns)
                plot = self.draw_attention_plot(att_w)
                logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
                plot.clf()

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            arr_ws_sd (numpy.ndarray): attention weights. It's shape would be
                differ from bachend.dtype=float
                * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax). 2)
                  other case => (B, Lmax, Tmax).
                * chainer-> attention weights (B, Lmax, Tmax).

        """
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws_sd = self.att_vis_fn(*batch)
        return att_ws_sd

    def get_attention_weight(self, idx, att_w, spkr_idx):
        """Transform attention weight in regard to self.reverse."""
        if self.reverse:
            dec_len = int(self.data[idx][1]["input"][0]["shape"][0])
            enc_len = int(self.data[idx][1]["output"][spkr_idx]["shape"][0])
        else:
            dec_len = int(self.data[idx][1]["output"][spkr_idx]["shape"][0])
            enc_len = int(self.data[idx][1]["input"][0]["shape"][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        """Visualize attention weights matrix.

        Args:
            att_w(Tensor): Attention weight matrix.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
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


def add_results_to_json(js, nbest_hyps_sd, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]):
            List of hypothesis for multi_speakers (# Utts x # Spkrs).
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js["utt2spk"] = js["utt2spk"]
    num_spkrs = len(nbest_hyps_sd)
    new_js["output"] = []

    for ns in range(num_spkrs):
        tmp_js = []
        nbest_hyps = nbest_hyps_sd[ns]

        for n, hyp in enumerate(nbest_hyps, 1):
            # parse hypothesis
            rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

            # copy ground-truth
            out_dic = dict(js["output"][ns].items())

            # update name
            out_dic["name"] += "[%d]" % n

            # add recognition results
            out_dic["rec_text"] = rec_text
            out_dic["rec_token"] = rec_token
            out_dic["rec_tokenid"] = rec_tokenid
            out_dic["score"] = score

            # add to list of N-best result dicts
            tmp_js.append(out_dic)

            # show 1-best result
            if n == 1:
                logging.info("groundtruth: %s" % out_dic["text"])
                logging.info("prediction : %s" % out_dic["rec_text"])

        new_js["output"].append(tmp_js)
    return new_js
