# Beam search module for partially autoregressive decoding.
# Copyright 2024 Masao Someki
# This script is licensed under MIT license.
# This script is the upgraded version used in https://arxiv.org/abs/2309.14922
import logging
import warnings
from itertools import groupby
from typing import Dict, List

import numpy
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.text.token_id_converter import TokenIDConverter
from espnet.nets.beam_search import Hypothesis
from espnet.nets.beam_search_partially_AR import PartiallyARBeamSearch
from espnet.nets.scorer_interface import MaskParallelScorerInterface, ScorerInterface

warnings.filterwarnings("ignore", category=UserWarning)


class PartiallyARInference(torch.nn.Module):
    """Mask-CTC-based partially autoregressive inference"""

    def __init__(
        self,
        ctc: CTC,
        decoder: AbsDecoder,
        threshold_probability: float,
        sos: int = None,
        eos: int = None,
        mask_token: int = None,
        token_list: List[int] = None,
        scorers: Dict[str, ScorerInterface] = None,
        weights: Dict[str, float] = None,
        beam_size: int = 10,
        max_seq_len: int = 5,
        max_mask_parallel: int = -1,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        # check if scorer is a MaskParallelScorerInterface object
        for k, v in scorers.items():
            assert isinstance(
                v, MaskParallelScorerInterface
            ), f"{k} is not a MaskParallelScorerInterface object"

        self.ctc = ctc
        self.decoder = decoder
        self.mask_token = mask_token
        self.threshold_probability = threshold_probability
        token_list = token_list + ["<mask>"]

        self.sos = sos
        self.eos = eos
        self.max_seq_len = max_seq_len

        logging.info(f"vocab_size: {len(token_list)}")
        ctc_weight = weights["ctc"] if "ctc" in weights.keys() else 0.0
        self.converter = TokenIDConverter(token_list=token_list)
        self.beam_search = PartiallyARBeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        self.nn_dict = self.beam_search.nn_dict
        self.max_mask_parallel = max_mask_parallel
        self.primer = []

    def set_hyp_primer(self, primer: List[int]):
        self.primer = primer

    def forward(self, enc_out: torch.Tensor, *args, **kwargs) -> List[Hypothesis]:
        """Perform Semi-AR inference"""
        # greedy ctc outputs
        enc_out = enc_out.unsqueeze(0)
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(enc_out)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1).cpu()

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        probs_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item()
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))

        # mask ctc outputs based on ctc probabilities
        p_thres = self.threshold_probability
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)
        y_in = (
            torch.zeros(1, len(y_idx), dtype=torch.long).to(enc_out.device)
        ) + self.mask_token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

        if mask_num == 0:
            # pad with mask tokens to ensure compatibility with mask-ctc output
            yseq = torch.tensor(
                [self.mask_token] + y_in.tolist()[0] + [self.mask_token],
                device=y_in.device,
            )
            return [Hypothesis(yseq=yseq)]

        # partially autoregressive decoding from here
        # First, merge the masked tokens
        yseq_with_mask = (
            torch.LongTensor([x[0] for x in groupby(y_in[0])])
            .unsqueeze(0)
            .to(y_in.device)
        )
        merged_mask_len = torch.cat(
            (
                torch.LongTensor([0]),
                torch.cumsum(
                    torch.LongTensor([len(list(x[1])) for x in groupby(y_in[0])]) - 1,
                    dim=0,
                )[:-1],
            )
        )

        # prepare required variables for retrieving information from y_hat
        y_hat_tokens = y_hat[y_idx]
        mask_num = torch.sum(yseq_with_mask == self.mask_token)

        # then use `add_mask` to register masks to the beam search class,
        # run beam search, and get the best hypotheses.
        # Since we might get OOM with the too many batch size,
        # we restrict the maximum number of masks to be processed at the same time.
        if self.max_mask_parallel == -1:
            self.max_mask_parallel = mask_num + 1

        result = y_in[0].clone().tolist()
        for i in range((mask_num // self.max_mask_parallel) + 1):
            bs_iter = i * self.max_mask_parallel
            max_iter = min(self.max_mask_parallel, mask_num - bs_iter)
            self.beam_search.init_masks()

            # register masks to the beam search class
            for m in range(bs_iter, bs_iter + max_iter):
                mask_idx = self._get_mask_idx(yseq_with_mask, i)
                yhat_idx = mask_idx + merged_mask_len[mask_idx]
                prev_tokens = (
                    [self.sos] + y_hat_tokens[:yhat_idx].tolist()
                    if mask_idx > 0
                    else [self.sos]
                )
                next_token = (
                    yseq_with_mask[0, mask_idx + 1].tolist()
                    if mask_idx < len(yseq_with_mask[0]) - 1
                    else [self.eos]
                )
                self.beam_search.add_mask(self.primer + prev_tokens, next_token)

            # run beam search and save to `result`
            hypos = self.beam_search(enc_out.squeeze(0), self.max_seq_len)
            for i_hypo, hypo in enumerate(hypos):
                res_mask = self._get_mask_idx(result, 0)
                hypo_list = [
                    x[0]
                    for x in groupby(
                        hypo.yseq[len(self.beam_search.masks[i_hypo][0]) :]
                    )
                ][
                    :-1
                ]  # remove eos
                result = result[:res_mask] + hypo_list + result[res_mask + 1 :]

        # pad with mask tokens to ensure compatibility with mask-ctc output
        yseq = torch.tensor([self.mask_token] + result + [self.mask_token])
        return [Hypothesis(yseq=yseq)]

    def _get_mask_idx(self, y_in, i: int, cs: torch.Tensor = None) -> List[int]:
        if cs is None:
            if type(y_in) is not torch.Tensor:  # then y_in is a list.
                y_in = torch.tensor(y_in, device="cpu").unsqueeze(0)
            cs = torch.cumsum(y_in[0] == self.mask_token, dim=0)

        return (cs == i + 1).nonzero()[0].item()
