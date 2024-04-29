import logging
from contextlib import contextmanager
from itertools import groupby
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy
import torch
from packaging.version import parse as V

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.beam_search import Hypothesis
from espnet.nets.beam_search_partially_AR import PartiallyARBeamSearch
from espnet.nets.scorer_interface import ScorerInterface
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.build_tokenizer import build_tokenizer

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class PartiallyARInference(torch.nn.Module):
    """Mask-CTC-based partially autoregressive inference"""

    def __init__(
        self,
        ctc: CTC,
        decoder: AbsDecoder,
        threshold_probability: float,
        sos: int = 4999,
        eos: int = 4999,
        mask_token: int = 5000,
        token_list: List[int] = None,
        scorers: Dict[str, ScorerInterface] = None,
        weights: Dict[str, float] = None,
        beam_size: int = 10,
        max_seq_len: int = 5,
        max_mask_parallel: int = 5,
    ):
        """Initialize Mask-CTC inference"""
        super().__init__()
        self.ctc = ctc
        self.decoder = decoder
        self.mask_token = mask_token
        self.threshold_probability = threshold_probability
        token_list = token_list + ['<mask>']

        self.sos = sos
        self.eos = eos
        self.max_seq_len = max_seq_len

        logging.info(f'vocab_size: {len(token_list)}')
        ctc_weight = weights['ctc'] if 'ctc' in weights.keys() else 0.0
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
    
    def set_hyp_primer(self, l: List[int]):
        self.primer = l

    def forward(self, x: torch.Tensor, *args, **kwargs) -> List[Hypothesis]:
        """Perform Semi-AR inference"""
        # greedy ctc outputs
        enc_out = x.unsqueeze(0)
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
            # pad with mask tokens to ensure compatibility with sos/eos tokens
            yseq = torch.tensor(
                [self.mask_token] + y_in.tolist()[0] + [self.mask_token], device=y_in.device
            )
            return [Hypothesis(yseq=yseq)], {}

        # Get the corresponding encoder frames for each mask token.
        merged_mask_len = torch.cat((torch.LongTensor([0]), torch.cumsum(torch.LongTensor([len(list(x[1])) for x in groupby(y_in[0])]) - 1, dim=0)[:-1]))
        _t = torch.LongTensor([x[0] for x in groupby(ctc_ids[0])])
        y_nonzero_idx = torch.nonzero(_t).squeeze(1)
        y_token_ends = torch.LongTensor(
            [len(list(x[1])) for x in groupby(ctc_ids[0])]
        )
        y_token_ends = torch.cumsum(y_token_ends, dim=0)
        
        y_in = torch.stack([x[0] for x in groupby(y_in[0])]).unsqueeze(0)
        mask_num = torch.sum(y_in[0] == self.mask_token)

        y_hat_tokens = y_hat[y_idx]
        result = y_in[0].clone().tolist()
        
        # compute mask parallel decode.
        for i in range((mask_num // self.max_mask_parallel) + 1):
            bs_iter = i * self.max_mask_parallel
            max_iter = min(self.max_mask_parallel, mask_num - bs_iter)
            self.beam_search.init_masks()
            for m in range(bs_iter, bs_iter + max_iter):
                mask_idx = self._get_mask_idx(y_in, m)
                yhat_idx = mask_idx + merged_mask_len[mask_idx]

                start_enc, end_enc = self._get_enc_ids(mask_idx, mask_idx + 1, merged_mask_len, y_in[0], y_nonzero_idx, y_token_ends)
                if end_enc is None:
                    end_enc = enc_out.size(1) - 1
                
                res_mask = self._get_mask_idx(result, 0)
                hyp_primer, next_token = self.init_beam_search(y_hat_tokens, yhat_idx, mask_idx, y_in)
                self.beam_search.add_mask(self.primer + hyp_primer, next_token, start_enc, end_enc)
            
            # If you got Out of memory error here,
            # Please consider lowering the threshold value.
            hypos = self.beam_search(enc_out.squeeze(0), self.max_seq_len)

            for i_hypo, hypo in enumerate(hypos):
                res_mask = self._get_mask_idx(result, 0)
                hypo_list = [x[0] for x in groupby(hypo.yseq[len(self.beam_search.masks[i_hypo][0]):])][:-1] # remove eos
                result = result[:res_mask] + hypo_list + result[res_mask+1:]

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        yseq = torch.tensor([self.mask_token] + result + [self.mask_token])
        return [Hypothesis(yseq=yseq)]
    

    def _get_enc_ids(self, token_start_idx, token_end_idx, merged_mask_len, masked_tokens, y_nonzero_idx, y_token_ends):
        if token_start_idx == 0:
            mml_prev = 0
        else:
            mml_prev = merged_mask_len[token_start_idx - 1]

        mml_cur = merged_mask_len[token_start_idx]

        si_for_y_nonzero = token_start_idx + mml_prev
        ei_for_y_nonzero = token_end_idx + mml_cur

        if token_start_idx <= 0:
            si_for_enc_out = 0
        else:
            si_for_y_ends = y_nonzero_idx[si_for_y_nonzero - 1]
            si_for_enc_out = y_token_ends[si_for_y_ends] - 1
        
        if token_start_idx >= len(masked_tokens):
            ei_for_enc_out = None
        elif ei_for_y_nonzero == len(y_nonzero_idx):
            ei_for_enc_out = None
        else:
            ei_for_y_starts = y_nonzero_idx[ei_for_y_nonzero]
            ei_for_enc_out = y_token_ends[ei_for_y_starts]
        
        return si_for_enc_out, ei_for_enc_out

    def init_beam_search_result(self, mask_idx, y_in, result, res_mask):
        hyp_primer = [self.sos] + result[:res_mask] if res_mask > 0 else [self.sos]
        next_token = y_in[0, mask_idx + 1] if mask_idx < len(y_in[0]) -1 else self.eos
        return hyp_primer, next_token

    def init_beam_search(self, y_hat_tokens, yhat_idx, mask_idx, y_in):
        hyp_primer = [self.sos] + y_hat_tokens[:yhat_idx].tolist() if mask_idx > 0 else [self.sos]
        next_token = y_in[0, mask_idx + 1].tolist() if mask_idx < len(y_in[0]) -1 else [self.eos]
        return hyp_primer, next_token

    def _get_mask_idx(self, y_in, i: int, cs: torch.Tensor = None) -> List[int]:
        if cs is None:
            if type(y_in) != torch.Tensor: # then y_in is a list.
                y_in = torch.tensor(y_in, device='cpu').unsqueeze(0)
            cs = torch.cumsum(y_in[0] == self.mask_token, dim=0)
        
        return (cs == i + 1).nonzero()[0].item()
