"""Search algorithms for transducer models."""

from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn.functional as F


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[List[List[torch.Tensor]], List[torch.Tensor]]
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
        )._asdict()


class BeamSearchTransducerESPnet2:
    """Beam search implementation for transducer."""

    def __init__(
        self,
        decoder,
        beam_size: int,
        lm,
        lm_weight: float,
        search_type: str = "default",
        nstep: int = 1,
        prefix_alpha: int = 1,
        max_sym_exp: int = 2,
        score_norm: bool = True,
    ):
        """Initialize transducer beam search.

        Args:
            decoder: Decoder class to use
            beam_size: Number of hypotheses kept during search
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            search_type: type of algorithm to use for search
            nstep: number of maximum expansion steps at each time step ("nsc")
            prefix_alpha: maximum prefix length in prefix search ("nsc")
            max_sym_exp: number of maximum symbol expansions at each time step ("tsd)
            score_norm: normalize final scores by length ("default")

        """
        self.decoder = decoder
        self.blank = decoder.blank
        self.beam_size = beam_size

        if self.beam_size <= 1:
            search_algorithm = self.greedy_search
        elif search_type == "default":
            search_algorithm = self.default_beam_search
        else:
            raise NotImplementedError

        self.search_algorithm = search_algorithm

        self.lm = lm
        self.lm_weight = lm_weight

        self.score_norm = score_norm

    def __call__(self, x: torch.Tensor) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x: Encoded speech feature (T, D)

        Returns:
            nbest_hyps: N-best decoding results

        """
        nbest_hyps = self.search_algorithm(x)

        return nbest_hyps

    def greedy_search(self, x: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.

        Args:
            x: encoder hidden state sequences (maxlen_in, Henc)

        Returns:
            hyp: 1-best decoding results

        """
        init_tensor = x.unsqueeze(0)
        dec_state = self.decoder.init_state(init_tensor)

        hyp = Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)

        cache = {}

        y, state, _ = self.decoder.score(hyp, cache, init_tensor)

        for i, xi in enumerate(x):
            ytu = torch.log_softmax(self.decoder.joint(xi, y[0]), dim=-1)
            logp, pred = torch.max(ytu, dim=-1)

            if pred != self.blank:
                hyp.yseq.append(int(pred))
                hyp.score += float(logp)

                hyp.dec_state = state

                y, state, _ = self.decoder.score(hyp, cache, init_tensor)

        return hyp

    def default_beam_search(self, x: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: encoder hidden state sequences (Tmax, Henc)

        Returns:
            nbest_hyps: n-best decoding results

        """
        beam = min(self.beam_size, self.decoder.vocab_size)
        beam_k = min(beam, (self.decoder.vocab_size - 1))

        init_tensor = x.unsqueeze(0)
        blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

        dec_state = self.decoder.init_state(init_tensor)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)]

        cache = {}

        for xi in x:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache, init_tensor)

                ytu = F.log_softmax(self.decoder.joint(xi, y[0]), dim=-1)

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

        if self.score_norm:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x.score / len(x.yseq), reverse=True
            )
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)

        return nbest_hyps
