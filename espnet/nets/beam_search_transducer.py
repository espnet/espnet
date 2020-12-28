"""Search algorithms for transducer models."""

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract
from espnet2.asr.decoder.abs_decoder import AbsDecoder


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[List[List[torch.Tensor]], List[torch.Tensor]]
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for transducer."""

    def __init__(
        self,
        decoder: Union[AbsDecoder, torch.nn.Module],
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        score_norm: bool = True,
    ):
        """Initialize transducer beam search.

        Args:
            decoder: Decoder class to use
            beam_size: Number of hypotheses kept during search
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            search_type: type of algorithm to use for search
            max_sym_exp: number of maximum symbol expansions at each time step ("tsd")
            u_max: maximum output sequence length ("alsd")
            nstep: number of maximum expansion steps at each time step ("nsc")
            prefix_alpha: maximum prefix length in prefix search ("nsc")
            score_norm: normalize final scores by length ("default")
        """
        assert check_argument_types()

        self.decoder = decoder
        self.beam_size = beam_size

        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim
        self.blank = decoder.blank

        if self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            self.search_algorithm = self.nsc_beam_search
        else:
            raise NotImplementedError

        self.lm = lm
        self.lm_weight = lm_weight

        self.max_sym_exp = max_sym_exp
        self.u_max = u_max
        self.nstep = nstep
        self.prefix_alpha = prefix_alpha
        self.score_norm = score_norm

    def __call__(self, h: torch.Tensor) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        if hasattr(self.decoder, "att_list"):
            self.decoder.att_list[0].reset()

        if hasattr(self.decoder, "att"):
            self.decoder.att[0].reset()

        nbest_hyps = self.search_algorithm(h)

        return nbest_hyps

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses

        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def greedy_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            hyp: 1-best decoding results

        """
        init_tensor = h.unsqueeze(0)
        dec_state = self.decoder.init_state(init_tensor)

        hyp = Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)

        cache = {}

        y, state, _ = self.decoder.score(hyp, cache, init_tensor)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.decoder.joint_network(hi, y[0]), dim=-1)
            logp, pred = torch.max(ytu, dim=-1)

            if pred != self.blank:
                hyp.yseq.append(int(pred))
                hyp.score += float(logp)

                hyp.dec_state = state

                y, state, _ = self.decoder.score(hyp, cache, init_tensor)

        return [hyp]

    def default_beam_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        init_tensor = h.unsqueeze(0)
        blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

        dec_state = self.decoder.init_state(init_tensor)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)]

        cache = {}

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache, init_tensor)

                ytu = torch.log_softmax(self.decoder.joint_network(hi, y[0]), dim=-1)

                top_k = ytu[1:].topk(beam_k, dim=-1)

                ytu = (
                    torch.cat((top_k[0], ytu[0:1])),
                    torch.cat((top_k[1] + 1, blank_tensor)),
                )

                if self.lm:
                    lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)

                for logp, k in zip(*ytu):
                    new_hyp = Hypothesis(
                        score=(max_hyp.score + float(logp)),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )

                    if k == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        new_hyp.dec_state = state

                        new_hyp.yseq.append(int(k))

                        if self.lm:
                            new_hyp.lm_state = lm_state
                            new_hyp.score += self.lm_weight * lm_scores[0][k]

                        hyps.append(new_hyp)

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        init_tensor = h.unsqueeze(0)
        beam_state = self.decoder.init_state(torch.zeros((beam, self.hidden_size)))

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        if self.lm:
            if hasattr(self.lm.predictor, "wordlm"):
                lm_model = self.lm.predictor.wordlm
                lm_type = "wordlm"
            else:
                lm_model = self.lm.predictor
                lm_type = "lm"

                B[0].lm_state = init_lm_state(lm_model)

            lm_layers = len(lm_model.rnn)

        cache = {}

        for hi in h:
            A = []
            C = B

            h_enc = hi.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C, beam_state, cache, init_tensor
                )

                beam_logp = torch.log_softmax(
                    self.decoder.joint_network(h_enc, beam_y), dim=-1
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

                if v < self.max_sym_exp:
                    if self.lm:
                        beam_lm_states = create_lm_batch_state(
                            [c.lm_state for c in C], lm_type, lm_layers
                        )

                        beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                            beam_lm_states, beam_lm_tokens, len(C)
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                                new_hyp.lm_state = select_lm_state(
                                    beam_lm_states, i, lm_type, lm_layers
                                )

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        h_length = int(h.size(0))
        u_max = min(self.u_max, (h_length - 1))

        init_tensor = h.unsqueeze(0)
        beam_state = self.decoder.init_state(torch.zeros((beam, self.hidden_size)))

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        final = []

        if self.lm:
            if hasattr(self.lm.predictor, "wordlm"):
                lm_model = self.lm.predictor.wordlm
                lm_type = "wordlm"
            else:
                lm_model = self.lm.predictor
                lm_type = "lm"

                B[0].lm_state = init_lm_state(lm_model)

            lm_layers = len(lm_model.rnn)

        cache = {}

        for i in range(h_length + u_max):
            A = []

            B_ = []
            h_states = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u + 1

                if t > (h_length - 1):
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

            if B_:
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_, beam_state, cache, init_tensor
                )

                h_enc = torch.stack([h[1] for h in h_states])

                beam_logp = torch.log_softmax(
                    self.decoder.joint_network(h_enc, beam_y), dim=-1
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                if self.lm:
                    beam_lm_states = create_lm_batch_state(
                        [b.lm_state for b in B_], lm_type, lm_layers
                    )

                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(B_)
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if h_states[i][0] == (h_length - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, lm_type, lm_layers
                            )

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def nsc_beam_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Note: the algorithm is not in his "complete" form but works almost as
        intended.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        init_tensor = h.unsqueeze(0)
        blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

        beam_state = self.decoder.init_state(torch.zeros((beam, self.hidden_size)))

        init_tokens = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens, beam_state, cache, init_tensor
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.lm:
            beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                None, beam_lm_tokens, 1
            )

            if hasattr(self.lm.predictor, "wordlm"):
                lm_model = self.lm.predictor.wordlm
                lm_type = "wordlm"
            else:
                lm_model = self.lm.predictor
                lm_type = "lm"

            lm_layers = len(lm_model.rnn)

            lm_state = select_lm_state(beam_lm_states, 0, lm_type, lm_layers)
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=state,
                y=[beam_y[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
            kept_hyps = []

            h_enc = hi.unsqueeze(0)

            for j in range(len(hyps) - 1):
                for i in range((j + 1), len(hyps)):
                    if (
                        is_prefix(hyps[j].yseq, hyps[i].yseq)
                        and (len(hyps[j].yseq) - len(hyps[i].yseq)) <= self.prefix_alpha
                    ):
                        next_id = len(hyps[i].yseq)

                        ytu = torch.log_softmax(
                            self.decoder.joint_network(hi, hyps[i].y[-1]), dim=0
                        )

                        curr_score = hyps[i].score + float(ytu[hyps[j].yseq[next_id]])

                        for k in range(next_id, (len(hyps[j].yseq) - 1)):
                            ytu = torch.log_softmax(
                                self.decoder.joint_network(hi, hyps[j].y[k]), dim=0
                            )

                            curr_score += float(ytu[hyps[j].yseq[k + 1]])

                        hyps[j].score = np.logaddexp(hyps[j].score, curr_score)

            S = []
            V = []
            for n in range(self.nstep):
                beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(
                    self.decoder.joint_network(h_enc, beam_y), dim=-1
                )
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                if self.lm:
                    beam_lm_scores = torch.stack([hyp.lm_scores for hyp in hyps])

                for i, hyp in enumerate(hyps):
                    i_topk = (
                        torch.cat((beam_topk[0][i], beam_logp[i, 0:1])),
                        torch.cat((beam_topk[1][i] + 1, blank_tensor)),
                    )

                    for logp, k in zip(*i_topk):
                        new_hyp = Hypothesis(
                            yseq=hyp.yseq[:],
                            score=(hyp.score + float(logp)),
                            y=hyp.y[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )

                        if k == self.blank:
                            S.append(new_hyp)
                        else:
                            new_hyp.yseq.append(int(k))

                            if self.lm:
                                new_hyp.score += self.lm_weight * float(
                                    beam_lm_scores[i, k]
                                )

                        V.append(new_hyp)

                V = sorted(V, key=lambda x: x.score, reverse=True)
                V = substract(V, hyps)[:beam]

                l_state = [v.dec_state for v in V]
                l_tokens = [v.yseq for v in V]

                beam_state = self.decoder.create_batch_states(
                    beam_state, l_state, l_tokens
                )
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V, beam_state, cache, init_tensor
                )

                if self.lm:
                    beam_lm_states = create_lm_batch_state(
                        [v.lm_state for v in V], lm_type, lm_layers
                    )
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(V)
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, lm_type, lm_layers
                            )
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.decoder.joint_network(h_enc, beam_y), dim=-1
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, lm_type, lm_layers
                            )
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)
