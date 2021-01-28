"""Parallel beam search module for online simulation."""

import argparse
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from pathlib import Path

import yaml

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search import BatchHypothesis
from espnet.nets.beam_search import Hypothesis
from espnet2.utils import config_argparse



class BatchBeamSearchOnlineSim(BatchBeamSearch):
    """online beam search implementation."""

    def set_streaming_config( self, asr_config: str ):
        train_config_file = Path(asr_config)
        with train_config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
            config = args["config"]
        config_file = Path(config)            
        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        if 'encoder_conf' in args.keys():
            enc_args = args['encoder_conf']
        if enc_args and 'block_size' in enc_args:
            self.block_size = enc_args['block_size']
        if enc_args and 'hop_size' in enc_args:
            self.hop_size = enc_args['hop_size']
        if enc_args and 'look_ahead' in enc_args:
            self.look_ahead = enc_args['look_ahead']
        

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
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

        self.conservative = True  # always true 

        if self.block_size and self.hop_size:
            cur_end_frame = int( self.block_size - self.hop_size/2 )
        else:
            cur_end_frame = x.shape[0]
        process_idx = 0
        if cur_end_frame < x.shape[0]:
            h = x.narrow( 0, 0, cur_end_frame )
        else:
            h = x
            
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(h)
        # running_hyps = self.init_hyp(x)
        prev_hyps = []
        ended_hyps = []
        prev_repeat = False

        continue_decode = True

        while continue_decode:
            move_to_next_block = False
            if cur_end_frame < x.shape[0]:
                h = x.narrow( 0, 0, cur_end_frame )
            else:
                h = x

            # extend states for ctc
            self.extend( h, running_hyps )

            while process_idx < maxlen:
                logging.debug("position " + str(process_idx))
                best = self.search(running_hyps, x)
                    
                if process_idx == maxlen-1:
                    # end decoding
                    running_hyps = self.post_process(process_idx, maxlen, maxlenratio, best, ended_hyps)
                n_batch = best.yseq.shape[0]
                local_ended_hyps = []
                is_local_eos = (
                    best.yseq[ torch.arange(n_batch), best.length-1] == self.eos
                )
                for i in range( is_local_eos.shape[0] ):
                    if is_local_eos[i]:
                        hyp = self._select( best, i )
                        local_ended_hyps.append( hyp )
                    elif best.yseq[i,-1] in best.yseq[i,:-1] and not prev_repeat and cur_end_frame < x.shape[0]:
                        move_to_next_block = True
                        prev_repeat = True
                if maxlenratio == 0.0 and end_detect([lh.asdict() for lh in local_ended_hyps], process_idx):
                    logging.info(f"end detected at {process_idx}")
                    continue_decode = False
                    break
                if len( local_ended_hyps ) > 0 and cur_end_frame < x.shape[0]:
                    move_to_next_block = True
                    
                if move_to_next_block:
                    if self.hop_size and cur_end_frame + int(self.hop_size) + int(self.hop_size/2) < x.shape[0]:
                        cur_end_frame += int(self.hop_size)
                    else:
                        cur_end_frame = x.shape[0]
                    logging.debug( 'Going to next block: %d', cur_end_frame )
                    if process_idx > 1 and len( prev_hyps ) > 0 and self.conservative:
                        running_hyps = prev_hyps
                        process_idx -= 1
                        prev_hyps = []
                    break

                prev_repeat = False
                prev_hyps = running_hyps
                running_hyps = self.post_process(process_idx, maxlen, maxlenratio, best, ended_hyps)
                
                if cur_end_frame >= x.shape[0]:
                    for hyp in local_ended_hyps:
                        ended_hyps.append(hyp)

                if len(running_hyps) == 0:
                    logging.info("no hypothesis. Finish decoding.")
                    continue_decode = False                    
                    break
                else:
                    logging.debug(f"remained hypotheses: {len(running_hyps)}")
                # increment number
                process_idx += 1

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

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
        return nbest_hyps

    def search(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        scores, states = self.score_full(running_hyps, x.expand(n_batch, *x.shape))
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)
    

    def extend( self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """ Extend probabilities and states with more encoded chunks 

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The exxtended hypothesis

        """

        for k, d in self.scorers.items():
            if hasattr( d, 'extend_prob' ):
                d.extend_prob( x )
            if hasattr( d, 'extend_state' ):
                hyps.states[k] = d.extend_state( hyps.states[k] )

        
