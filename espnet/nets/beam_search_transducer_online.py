"""Parallel beam search module for online simulation."""

import logging
from typing import Any  # noqa: H301
from typing import Dict  # noqa: H301
from typing import List  # noqa: H301
from typing import Tuple  # noqa: H301

import torch
import numpy as np

from espnet2.asr.transducer.beam_search_transducer import (
    BeamSearchTransducer,
    ExtendedHypothesis,
    Hypothesis,
)
from espnet.nets.pytorch_backend.transducer.utils import (
    create_lm_batch_states,
    init_lm_state,
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    select_lm_state,
    subtract,
)

# from espnet.nets.transducer_decoder_interface import ExtendedHypothesis, Hypothesis
# from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: H301
# from espnet.nets.batch_beam_search import BatchHypothesis  # noqa: H301
# from espnet.nets.e2e_asr_common import end_detect


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
        self.decoder.set_device(x.device)

        if self.encbuffer is None:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

        x = self.encbuffer

        ret = None
        while True:
            if x.shape[0] <= self.process_idx:
                break
            h = x[self.process_idx]

            logging.debug("Start processing idx: %d", self.process_idx)
            if self.search_algorithm == self.align_length_sync_decoding:
                raise NotImplementedError
            ret = self.search_algorithm(h)
            logging.debug("Finished processing idx: %d", self.process_idx)

            # increment number
            self.process_idx += 1

        if is_final:
            return ret

        if ret is None:
            if self.prev_output is None:
                return []
            else:
                return self.prev_output
        else:
            self.prev_output = ret
            # N-best results
            return ret

    def default_beam_search(self, enc_out_t: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (1, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.running_hyps is None:
            # Init hyps
            dec_state = self.decoder.init_state(1)

            kept_hyps = [
                Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
            ]
        else:
            kept_hyps = self.running_hyps

        beam_k = min(self.beam, (self.vocab_size - 1))
        cache = {}
        cache_lm = {}

        hyps = kept_hyps
        kept_hyps = []

        if self.token_list is not None:
            logging.debug(
                "\n"
                + "\n".join(
                    [
                        "hypo: "
                        + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                        + f", score: {round(float(hyp.score), 2)}"
                        for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                    ]
                )
            )

        while True:
            max_hyp = max(hyps, key=lambda x: x.score)
            hyps.remove(max_hyp)

            dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

            logp = torch.log_softmax(
                self.joint_network(enc_out_t, dec_out),
                dim=-1,
            )
            top_k = logp[1:].topk(beam_k, dim=-1)

            kept_hyps.append(
                Hypothesis(
                    score=(max_hyp.score + float(logp[0:1])),
                    yseq=max_hyp.yseq[:],
                    dec_state=max_hyp.dec_state,
                    lm_state=max_hyp.lm_state,
                )
            )

            if self.use_lm:
                if tuple(max_hyp.yseq) not in cache_lm:
                    lm_scores, lm_state = self.lm.score(
                        torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:],
                            device=self.decoder.device,
                        ),
                        max_hyp.lm_state,
                        None,
                    )
                    cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                else:
                    lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
            else:
                lm_state = max_hyp.lm_state

            for logp, k in zip(*top_k):
                score = max_hyp.score + float(logp)

                if self.use_lm:
                    score += self.lm_weight * lm_scores[k + 1]

                hyps.append(
                    Hypothesis(
                        score=score,
                        yseq=max_hyp.yseq[:] + [int(k + 1)],
                        dec_state=state,
                        lm_state=lm_state,
                    )
                )

            hyps_max = float(max(hyps, key=lambda x: x.score).score)
            kept_most_prob = sorted(
                [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                key=lambda x: x.score,
            )
            if len(kept_most_prob) >= self.beam:
                kept_hyps = kept_most_prob
                break

        self.running_hyps = kept_hyps
        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, enc_out_t: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out_t: Encoder output sequence. (1, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.running_hyps is None:
            # Init hyps
            beam_state = self.decoder.init_state(self.beam)

            B = [
                Hypothesis(
                    yseq=[self.blank_id],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                )
            ]
            cache = {}
            self.beam_state = beam_state

            if self.use_lm:
                B[0].lm_state = self.lm.zero_state()
        else:
            beam_state = self.beam_state
            B = self.running_hyps
            cache = {}

        A = []
        C = B

        enc_out_t = enc_out_t.unsqueeze(0)

        for v in range(self.max_sym_exp):
            D = []

            beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                C,
                beam_state,
                cache,
                self.use_lm,
            )

            beam_logp = torch.log_softmax(
                self.joint_network(enc_out_t, beam_dec_out),
                dim=-1,
            )
            beam_topk = beam_logp[:, 1:].topk(self.beam, dim=-1)

            seq_A = [h.yseq for h in A]

            for i, hyp in enumerate(C):
                if hyp.yseq not in seq_A:
                    A.append(
                        Hypothesis(
                            score=(hyp.score + float(beam_logp[i, 0])),
                            yseq=hyp.yseq[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                        )
                    )
                else:
                    dict_pos = seq_A.index(hyp.yseq)

                    A[dict_pos].score = np.logaddexp(
                        A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                    )

            if v < (self.max_sym_exp - 1):
                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens, [c.lm_state for c in C], None
                    )

                for i, hyp in enumerate(C):
                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        D.append(new_hyp)

            C = sorted(D, key=lambda x: x.score, reverse=True)[: self.beam]

        B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam]
        self.beam_state = beam_state
        self.running_hyps = B

        return self.sort_nbest(B)

    def nsc_beam_search(self, enc_out_t: torch.Tensor) -> List[ExtendedHypothesis]:
        """N-step constrained beam search implementation.

        Based on/Modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Args:
            enc_out_t: Encoder output sequence. (1, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        if self.running_hyps is None:
            # Init hyps
            beam_state = self.decoder.init_state(self.beam)

            init_tokens = [
                ExtendedHypothesis(
                    yseq=[self.blank_id],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                )
            ]

            cache = {}

            beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                init_tokens,
                beam_state,
                cache,
                self.use_lm,
            )

            state = self.decoder.select_state(beam_state, 0)

            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    beam_lm_tokens,
                    [i.lm_state for i in init_tokens],
                    None,
                )
                lm_state = beam_lm_states[0]
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
        else:
            beam_state = None
            kept_hyps = self.running_hyps
            cache = {}

        beam_k = min(self.beam, (self.vocab_size - 1))

        hyps = self.prefix_search(
            sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
            enc_out_t,
        )
        kept_hyps = []

        beam_enc_out = enc_out_t.unsqueeze(0)

        S = []
        V = []
        for n in range(self.nstep):
            beam_dec_out = torch.stack([hyp.dec_out[-1] for hyp in hyps])

            beam_logp = torch.log_softmax(
                self.joint_network(beam_enc_out, beam_dec_out),
                dim=-1,
            )
            beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

            for i, hyp in enumerate(hyps):
                S.append(
                    ExtendedHypothesis(
                        yseq=hyp.yseq[:],
                        score=hyp.score + float(beam_logp[i, 0:1]),
                        dec_out=hyp.dec_out[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        lm_scores=hyp.lm_scores,
                    )
                )

                for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                    score = hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * float(hyp.lm_scores[k])

                    V.append(
                        ExtendedHypothesis(
                            yseq=hyp.yseq[:] + [int(k)],
                            score=score,
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

            V.sort(key=lambda x: x.score, reverse=True)
            V = subtract(V, hyps)[: self.beam]

            beam_state = self.decoder.create_batch_states(
                beam_state,
                [v.dec_state for v in V],
                [v.yseq for v in V],
            )
            beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                V,
                beam_state,
                cache,
                self.use_lm,
            )

            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    beam_lm_tokens, [v.lm_state for v in V], None
                )

            if n < (self.nstep - 1):
                for i, v in enumerate(V):
                    v.dec_out.append(beam_dec_out[i])

                    v.dec_state = self.decoder.select_state(beam_state, i)

                    if self.use_lm:
                        v.lm_state = beam_lm_states[i]
                        v.lm_scores = beam_lm_scores[i]

                hyps = V[:]
            else:
                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )

                for i, v in enumerate(V):
                    if self.nstep != 1:
                        v.score += float(beam_logp[i, 0])

                    v.dec_out.append(beam_dec_out[i])

                    v.dec_state = self.decoder.select_state(beam_state, i)

                    if self.use_lm:
                        v.lm_state = beam_lm_states[i]
                        v.lm_scores = beam_lm_scores[i]

        kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[: self.beam]

        self.running_hyps = kept_hyps
        return self.sort_nbest(kept_hyps)

    def modified_adaptive_expansion_search(self, enc_out_t):
        """Recognize one block."""
        if self.running_hyps is None:
            # Init hyps
            beam_state = self.decoder.init_state(self.beam)

            init_tokens = [
                ExtendedHypothesis(
                    yseq=[self.blank_id],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                )
            ]

            cache = {}

            beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                init_tokens,
                beam_state,
                cache,
                self.use_lm,
            )

            state = self.decoder.select_state(beam_state, 0)

            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    beam_lm_tokens, [i.lm_state for i in init_tokens], None
                )

                lm_state = beam_lm_states[0]
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
        else:
            beam_state = None
            kept_hyps = self.running_hyps
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
                self.joint_network(beam_enc_out, beam_dec_out),
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
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens, [k.lm_state for k in list_exp], None
                    )

                if n < (self.nstep - 1):
                    for i, hyp in enumerate(list_exp):
                        hyp.dec_out.append(beam_dec_out[i])
                        hyp.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            hyp.lm_state = beam_lm_states[i]
                            hyp.lm_scores = beam_lm_scores[i]

                    hyps = list_exp[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(beam_enc_out, beam_dec_out),
                        dim=-1,
                    )

                    for i, hyp in enumerate(list_exp):
                        hyp.score += float(beam_logp[i, 0])

                        hyp.dec_out.append(beam_dec_out[i])
                        hyp.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            hyp.lm_states = beam_lm_states[i]
                            hyp.lm_scores = beam_lm_scores[i]

                    kept_hyps = sorted(
                        list_b + list_exp, key=lambda x: x.score, reverse=True
                    )[: self.beam]

        self.running_hyps = kept_hyps

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
