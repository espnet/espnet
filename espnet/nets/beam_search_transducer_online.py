"""Parallel beam search module for online simulation."""

import logging
from typing import Any  # noqa: H301
from typing import Dict  # noqa: H301
from typing import List  # noqa: H301
from typing import Tuple  # noqa: H301

import torch
import numpy as np

from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.transducer.utils import (
    create_lm_batch_states,
    init_lm_state,
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    select_lm_state,
    subtract,
)
from espnet.nets.transducer_decoder_interface import ExtendedHypothesis, Hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: H301
from espnet.nets.batch_beam_search import BatchHypothesis  # noqa: H301
from espnet.nets.beam_search import Hypothesis
from espnet.nets.e2e_asr_common import end_detect


class BeamSearchTransducerOnline(BeamSearchTransducer):
    """Online beam search implementation.
    Only mAES search algorithm is supported for now.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def __init__(
        self,
        *args,
        block_size=40,
        hop_size=16,
        look_ahead=16,
        disable_repetition_detection=False,
        encoded_feat_length_limit=0,
        decoder_text_length_limit=0,
        token_list: List[str] = None,
        **kwargs,
    ):
        """Initialize beam search."""
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.disable_repetition_detection = disable_repetition_detection
        self.encoded_feat_length_limit = encoded_feat_length_limit
        self.decoder_text_length_limit = decoder_text_length_limit
        self.beam = min(self.beam_size, self.vocab_size)

        self.reset()

    def reset(self):
        """Reset parameters."""
        self.encbuffer = None
        self.running_hyps = None
        self.beam_state = None
        self.process_idx = 0
        self.prev_output = None

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
            if (
                self.decoder_text_length_limit > 0
                and len(hyp.yseq) > 0
                and len(hyp.yseq[0]) > self.decoder_text_length_limit
            ):
                temp_yseq = hyp.yseq.narrow(
                    1, -self.decoder_text_length_limit, self.decoder_text_length_limit
                ).clone()
                temp_yseq[:, 0] = self.sos
                self.running_hyps.states["decoder"] = [
                    None for _ in self.running_hyps.states["decoder"]
                ]
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def init_hyp(self, enc_out: torch.Tensor) -> List[ExtendedHypothesis]:
        self.beam_state = self.decoder.init_state(self.beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(self.beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, self.beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            self.beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(self.beam_state, 0)

        if self.use_lm:
            beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                None, beam_lm_tokens, 1
            )
            lm_state = select_lm_state(
                beam_lm_states, 0, self.lm_layers, self.is_wordlm
            )
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        return kept_hyps

    def prefix_search(
        self, hyps: List[ExtendedHypothesis], enc_out_t: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """Prefix search for NSC and mAES strategies.

        Based on https://arxiv.org/pdf/1211.3711.pdf

        """
        for j, hyp_j in enumerate(hyps[:-1]):
            for hyp_i in hyps[(j + 1) :]:
                curr_id = len(hyp_j.yseq)
                pref_id = len(hyp_i.yseq)

                if (
                    is_prefix(hyp_j.yseq, hyp_i.yseq)
                    and (curr_id - pref_id) <= self.prefix_alpha
                ):
                    logp = torch.log_softmax(
                        self.joint_network(enc_out_t, hyp_i.dec_out[-1]),
                        dim=-1,
                    )

                    curr_score = hyp_i.score + float(logp[hyp_j.yseq[pref_id]])

                    for k in range(pref_id, (curr_id - 1)):
                        logp = torch.log_softmax(
                            self.joint_network(enc_out_t, hyp_j.dec_out[k]),
                            dim=-1,
                        )

                        curr_score += float(logp[hyp_j.yseq[k + 1]])

                    hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

        return hyps

    def __call__(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        is_final: bool = True,
    ) -> List[ExtendedHypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[ExtendedHypothesis]: N-best decoding results

        """
        if self.encbuffer is None:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

        x = self.encbuffer

        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))

        ret = None
        while True:
            if  x.shape[0] <= self.process_idx:
                break
            h = x[self.process_idx]

            logging.debug("Start processing idx: %d", self.process_idx)

            if self.running_hyps is None:
                self.running_hyps = self.init_hyp(h)
            ret = self.process_one_block(h)
            logging.debug("Finished processing idx: %d", self.process_idx)

            # increment number
            self.process_idx += 1

        if ret is None:
            if self.prev_output is None:
                return []
            else:
                return self.prev_output
        else:
            self.prev_output = ret
            # N-best results
            return ret

        #return ret

    def process_one_block(self, enc_out_t):
        """Recognize one block."""
        kept_hyps = self.running_hyps
        beam_state = self.beam_state
        cache = {}

        hyps = self.prefix_search(
            sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
            enc_out_t,
        )
        kept_hyps = []

        beam_enc_out = enc_out_t.unsqueeze(0)

        list_b = []
        duplication_check = [hyp.yseq for hyp in hyps]

        for n in range(self.nstep):
            beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])

            beam_logp, beam_idx = torch.log_softmax(
                self.joint_network(beam_enc_out, beam_dec_out)
                / self.softmax_temperature,
                dim=-1,
            ).topk(self.max_candidates, dim=-1)

            k_expansions = select_k_expansions(
                hyps,
                beam_idx,
                beam_logp,
                self.expansion_gamma,
            )

            list_exp = []
            for i, hyp in enumerate(hyps):
                for k, new_score in k_expansions[i]:
                    new_hyp = ExtendedHypothesis(
                        yseq=hyp.yseq[:],
                        score=new_score,
                        dec_out=hyp.dec_out[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        lm_scores=hyp.lm_scores,
                    )

                    if k == 0:
                        list_b.append(new_hyp)
                    else:
                        if new_hyp.yseq + [int(k)] not in duplication_check:
                            new_hyp.yseq.append(int(k))

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * float(
                                    hyp.lm_scores[k]
                                )

                            list_exp.append(new_hyp)

            if not list_exp:
                kept_hyps = sorted(list_b, key=lambda x: x.score, reverse=True)[
                    : self.beam
                ]

                break
            else:
                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [hyp.dec_state for hyp in list_exp],
                    [hyp.yseq for hyp in list_exp],
                )

                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    list_exp,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_states = create_lm_batch_states(
                        [hyp.lm_state for hyp in list_exp],
                        self.lm_layers,
                        self.is_wordlm,
                    )
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(list_exp)
                    )

                if n < (self.nstep - 1):
                    for i, hyp in enumerate(list_exp):
                        hyp.dec_out.append(beam_dec_out[i])
                        hyp.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            hyp.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            hyp.lm_scores = beam_lm_scores[i]

                    hyps = list_exp[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(beam_enc_out, beam_dec_out)
                        / self.softmax_temperature,
                        dim=-1,
                    )

                    for i, hyp in enumerate(list_exp):
                        hyp.score += float(beam_logp[i, 0])

                        hyp.dec_out.append(beam_dec_out[i])
                        hyp.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            hyp.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            hyp.lm_scores = beam_lm_scores[i]

                    kept_hyps = sorted(
                        list_b + list_exp, key=lambda x: x.score, reverse=True
                    )[: self.beam]

        self.running_hyps = kept_hyps
        self.beam_state = beam_state

        return self.sort_nbest(kept_hyps)

    def assemble_hyps(self, ended_hyps):
        """Assemble the hypotheses."""
        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return []

        # report the best result
        best = nbest_hyps[0]
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if token_list is not None:
            logging.info(
                "best hypo: " + "".join([token_list[x] for x in best.yseq[1:]]) + "\n"
            )
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
