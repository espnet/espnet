# Copyright 2021 Johns Hopkins University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Ensemble model for speech translation (pytorch)."""

from argparse import Namespace
import logging
import math

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.st_interface import STInterface


class E2E(STInterface, torch.nn.Module):
    """EnsembleE2E module (A wrapper around an ensemble of models.

    :param list models: models for ensemble inference
    """

    def __init__(self, models):
        """Construct an EnsembleE2E object.

        :param list models: models for ensemble inference
        """
        torch.nn.Module.__init__(self)
        self.model_size = len(models)
        self.single_model = models[0]
        self.sos = self.single_model.sos
        self.eos = self.single_model.eos
        self.models = torch.nn.ModuleList(models)

    def translate(
        self,
        x,
        trans_args,
        char_list=None,
    ):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        logging.info("input lengths: " + str(x.shape[0]))

        self.eval()
        enc_outputs = []
        reference_length = 0
        for m in self.models:
            if hasattr(m, "encoder_st"):  # for multi-decoder
                enc_output = m.encode(x, trans_args, char_list)
                if enc_output[1].size(1) > reference_length:
                    reference_length = enc_output[1].size(1)
                enc_outputs.append(
                    (enc_output[0].unsqueeze(0), enc_output[1].unsqueeze(0))
                )
            else:
                enc_output = m.encode(x)
                if enc_output.size(1) > reference_length:
                    reference_length = enc_output.size(1)
                enc_outputs.append(enc_output.unsqueeze(0))

        h = enc_outputs

        logging.info("encoder output lengths: " + str(reference_length))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        if trans_args.maxlenratio == 0:
            maxlen = reference_length
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * reference_length))
        minlen = int(trans_args.minlenratio * reference_length)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))
            local_scores = []

            for m in range(len(h)):
                local_scores.append(
                    self.models[m].decoder_forward_one_step(h[m], i, hyps)
                )

            avg_scores = torch.logsumexp(
                torch.stack(local_scores, dim=0), dim=0
            ) - math.log(self.model_size)

            hyps_best_kept = []
            for j, hyp in enumerate(hyps):
                local_best_scores, local_best_ids = torch.topk(
                    avg_scores[j : j + 1], beam, dim=1
                )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps
