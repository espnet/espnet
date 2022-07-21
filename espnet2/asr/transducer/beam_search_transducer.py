"""Search algorithms for Transducer models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.joint_network import JointNetwork
from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.pytorch_backend.transducer.utils import (
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    subtract,
)


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for Transducer."""

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        expansion_gamma: int = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = True,
        nbest: int = 1,
        token_list: List[str] = None,
    ):
        """Initialize Transducer search module.

        Args:
            decoder: Decoder module.
            joint_network: Joint network module.
            beam_size: Beam size.
            lm: LM class.
            lm_weight: LM weight for soft fusion.
            search_type: Search algorithm to use during inference.
            max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
            u_max: Maximum output sequence length. (ALSD)
            nstep: Number of maximum expansion steps at each time step. (NSC/mAES)
            prefix_alpha: Maximum prefix length in prefix search. (NSC/mAES)
            expansion_beta:
              Number of additional candidates for expanded hypotheses selection. (mAES)
            expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
            score_norm: Normalize final scores by length. ("default")
            nbest: Number of final hypothesis.

        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim

        self.sos = self.vocab_size - 1
        self.token_list = token_list

        self.blank_id = decoder.blank_id

        if self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError
            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError
            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError
            self.nstep = nstep
            self.prefix_alpha = prefix_alpha

            self.search_algorithm = self.nsc_beam_search
        elif search_type == "maes":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError
            self.nstep = nstep if nstep > 1 else 2
            self.prefix_alpha = prefix_alpha
            self.expansion_gamma = expansion_gamma
            self.expansion_beta = expansion_beta

            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError

        self.use_lm = lm is not None
        self.lm = lm
        self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.nbest = nbest

    def __call__(
        self, enc_out: torch.Tensor
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        nbest_hyps = self.search_algorithm(enc_out)

        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[ExtendedHypothesis]]
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: Hypothesis.

        Return:
            hyps: Sorted hypothesis.

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

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
                        self.joint_network(enc_out_t, hyp_i.dec_out[-1]), dim=-1,
                    )

                    curr_score = hyp_i.score + float(logp[hyp_j.yseq[pref_id]])

                    for k in range(pref_id, (curr_id - 1)):
                        logp = torch.log_softmax(
                            self.joint_network(enc_out_t, hyp_j.dec_out[k]), dim=-1,
                        )

                        curr_score += float(logp[hyp_j.yseq[k + 1]])

                    hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

        return hyps

    def greedy_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)

        for enc_out_t in enc_out:
            logp = torch.log_softmax(self.joint_network(enc_out_t, dec_out), dim=-1,)
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state

                dec_out, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)]
        cache = {}
        cache_lm = {}

        for enc_out_t in enc_out:
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
                    self.joint_network(enc_out_t, dec_out), dim=-1,
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
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C, beam_state, cache, self.use_lm,
                )

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out), dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

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

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoder output sequences. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)

        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        final = []
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for i in range(t_max + u_max):
            A = []

            B_ = []
            B_enc_out = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u

                if t > (t_max - 1):
                    continue

                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_, beam_state, cache, self.use_lm,
                )

                beam_enc_out = torch.stack([x[1] for x in B_enc_out])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out), dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens, [b.lm_state for b in B_], None,
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == (t_max - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def nsc_beam_search(self, enc_out: torch.Tensor) -> List[ExtendedHypothesis]:
        """N-step constrained beam search implementation.

        Based on/Modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens, beam_state, cache, self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_scores, beam_lm_states = self.lm.batch_score(
                beam_lm_tokens, [i.lm_state for i in init_tokens], None,
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

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True), enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            S = []
            V = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([hyp.dec_out[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out), dim=-1,
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
                V = subtract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state, [v.dec_state for v in V], [v.yseq for v in V],
                )
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V, beam_state, cache, self.use_lm,
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
                        self.joint_network(beam_enc_out, beam_dec_out), dim=-1,
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.dec_out.append(beam_dec_out[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = beam_lm_states[i]
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)

    def modified_adaptive_expansion_search(
        self, enc_out: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """It's the modified Adaptive Expansion Search (mAES) implementation.

        Based on/modified from https://ieeexplore.ieee.org/document/9250505 and NSC.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens, beam_state, cache, self.use_lm,
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

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True), enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            list_b = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out), dim=-1,
                )
                k_expansions = select_k_expansions(
                    hyps, beam_logp, beam, self.expansion_gamma, self.expansion_beta
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
                            new_hyp.yseq.append(int(k))

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * float(
                                    hyp.lm_scores[k]
                                )

                            list_exp.append(new_hyp)

                if not list_exp:
                    kept_hyps = sorted(list_b, key=lambda x: x.score, reverse=True)[
                        :beam
                    ]

                    break
                else:
                    beam_state = self.decoder.create_batch_states(
                        beam_state,
                        [hyp.dec_state for hyp in list_exp],
                        [hyp.yseq for hyp in list_exp],
                    )

                    beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                        list_exp, beam_state, cache, self.use_lm,
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
                            self.joint_network(beam_enc_out, beam_dec_out), dim=-1,
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
                        )[:beam]

        return self.sort_nbest(kept_hyps)
