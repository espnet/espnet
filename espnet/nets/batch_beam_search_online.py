"""Parallel beam search module for online simulation."""

import logging
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any

import yaml

import torch
import sys
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search import BatchHypothesis
from espnet.nets.beam_search import Hypothesis
from espnet.nets.e2e_asr_common import end_detect



class BatchBeamSearchOnline(BatchBeamSearch):
    """Online beam search implementation.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def __init__(self, *args, feature_width=0, text_width=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
        self.feature_width = feature_width
        self.text_width = text_width
        
    def reset(self):
        self.encbuffer = None
        self.running_hyps = None
        self.prev_hyps = []
        self.ended_hyps = []
        self.process_idx = 0
        self.prev_nbest_hyps = []

    def score_full(
        self, hyp: BatchHypothesis, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if len(hyp.yseq) > 0 and self.text_width > 0 and len(hyp.yseq[0]) > self.text_width:
                temp_yseq = hyp.yseq.narrow(1, -10, 10).clone()
                temp_yseq[:,0] = 3951
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        return scores, states
        
    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0,
        is_final: bool = True
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        if is_final == False and x.size(0) == 0:
            return self.prev_nbest_hyps
        
        self.conservative = True  # always true
        if not self.running_hyps:
            self.running_hyps = self.init_hyp(x)
        if self.encbuffer is None:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer,x], axis=0)
        
        # set length bounds
        if maxlenratio == 0:
            maxlen = self.encbuffer.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * self.encbuffer.size(0)))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("accumulated input length: " + str(self.encbuffer.shape[0]))
        logging.info("max output length: " + str(maxlen))

        # extend states for ctc
        self.extend(self.encbuffer, self.running_hyps)
        block_ended_hyps=[]
        prev_repeat = False
        while self.process_idx < maxlen:
            logging.debug("position " + str(self.process_idx) + " " + str(maxlen))
            logging.debug(self.running_hyps.yseq.shape)

            if self.text_width > 0 and self.encbuffer.shape[0] > self.feature_width:
                h = self.encbuffer.narrow(0, self.encbuffer.shape[0]-200, 200)
            else:
                h = self.encbuffer
            h =  self.encbuffer
            self.running_hyps.states['decoder'] = [ None for _ in self.running_hyps.states['decoder']]
            
            best = self.search(self.running_hyps, h)
            
            n_batch = best.yseq.shape[0]

            if not is_final:
                is_eos = (
                    best.yseq[torch.arange(n_batch), best.length - 1]
                    == self.eos
                )

                if is_eos.any():
                    break
                else:
                    self.running_hyps = best
            if is_final:
                # post process of one iteration
                self.running_hyps = self.post_process(self.process_idx,
                                                      maxlen,
                                                      maxlenratio,
                                                      best,
                                                      self.ended_hyps)
                if maxlenratio == 0.0 and end_detect([h.asdict() for h in self.ended_hyps], self.process_idx):
                    logging.info(f"end detected at {self.process_idx}")
                    break
                elif len(self.running_hyps) == 0:
                    logging.info("no hypothesis. Finish decoding.")
                    break
            logging.debug(f"remained hypothesis: {len(self.running_hyps)}")

            self.process_idx += 1

        ended_hyps = self.ended_hyps if is_final else self.unbatchfy(self.running_hyps)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        logging.info(str(len(nbest_hyps)))
        if len(nbest_hyps) == 0:
            return []

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        self.prev_nbest_hyps = nbest_hyps
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The exxtended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
