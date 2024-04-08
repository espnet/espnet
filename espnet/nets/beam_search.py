"""Beam search module."""

import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface, ScorerInterface

logger = logging.getLogger(__name__)


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()
    # dec hidden state corresponding to yseq, used for searchable hidden ints
    hs: List[torch.Tensor] = []

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        return_hs: bool = False,
        hyp_primer: List[int] = None,
        normalize_length: bool = False,
    ):
        """Initialize beam search.

        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules
                e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search
                will be `int(pre_beam_ratio * beam_size)`
            return_hs (bool): Whether to return hidden intermediates
            normalize_length (bool): If true, select the best ended hypotheses
                based on length-normalized scores rather than the accumulated scores

        """
        super().__init__()
        # set scorers
        self.weights = weights
        self.scorers = dict()
        self.full_scorers = dict()
        self.part_scorers = dict()
        # this module dict is required for recursive cast
        # `self.to(device, dtype)` in `recog.py`
        self.nn_dict = torch.nn.ModuleDict()
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(
                v, ScorerInterface
            ), f"{k} ({type(v)}) does not implement ScorerInterface"
            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos

        # added for OpenAI Whisper decoding
        self.hyp_primer = hyp_primer

        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        if (
            pre_beam_score_key is not None
            and pre_beam_score_key != "full"
            and pre_beam_score_key not in self.full_scorers
        ):
            raise KeyError(f"{pre_beam_score_key} is not found in {self.full_scorers}")
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = (
            self.pre_beam_score_key is not None
            and self.pre_beam_size < self.n_vocab
            and len(self.part_scorers) > 0
        )
        self.return_hs = return_hs
        self.normalize_length = normalize_length

    def set_hyp_primer(self, hyp_primer: List[int] = None) -> None:
        """Set the primer sequence for decoding.

        Used for OpenAI Whisper models.
        """
        self.hyp_primer = hyp_primer

    def init_hyp(self, x: torch.Tensor) -> List[Hypothesis]:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0

        # NOTE (Shih-Lun): added for OpenAI Whisper ASR
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                hs=[],
                yseq=torch.tensor(primer, device=x.device),
            )
        ]

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append

        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score_full(
        self, hyp: Hypothesis, x: torch.Tensor, pre_x: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

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
            if "decoder" in k and self.return_hs:
                scores[k], hs, states[k] = d.score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif pre_x is not None:
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self, hyp: Hypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 1D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.part_scorers`
                and tensor score values of shape: `(len(ids),)`,
                and state dict that has string keys
                and state values of `self.part_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
            Its shape is `(self.n_vocab,)`.
            ids (torch.Tensor): The partial token ids to compute topk

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`

        """
        # no pre beam performed
        if weighted_scores.size(0) == ids.size(0):
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids

        # mask pruned in pre-beam not to select in topk
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
        next_part_scores: Dict[str, torch.Tensor],
        part_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, torch.Tensor]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
            if self.return_hs:
                hs, scores, states = self.score_full(hyp, x, pre_x=pre_x)
            else:
                scores, states = self.score_full(hyp, x, pre_x=pre_x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                if self.return_hs:
                    new_hs = hyp.hs + [hs.squeeze(0)]
                else:
                    new_hs = []
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                        hs=new_hs,
                    )
                )

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.
                If minlenratio<0.0, its absolute value is interpreted
                as a constant min output length.
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if pre_x is not None:
            inp = pre_x
        else:
            inp = x
        if maxlenratio == 0:
            maxlen = inp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * inp.size(0))
        logger.info("decoder input length: " + str(inp.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x if pre_x is None else pre_x)
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x, pre_x=pre_x)
            # post process of one iteration
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

        if self.normalize_length:
            # Note (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)

        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning(
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
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"total log probability: {best.score:.2f}")
        logger.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logger.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logger.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                "best hypo length: {} == max output length: {}".format(
                    best.yseq[1:-1].shape[0], maxlen
                )
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                + "please consider to increase the maxlenratio."
            )
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            List[Hypothesis]: The new running hypotheses.

        """
        logger.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: "
                + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                if i >= minlen:
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps


def beam_search(
    x: torch.Tensor,
    sos: int,
    eos: int,
    beam_size: int,
    vocab_size: int,
    scorers: Dict[str, ScorerInterface],
    weights: Dict[str, float],
    token_list: List[str] = None,
    maxlenratio: float = 0.0,
    minlenratio: float = 0.0,
    pre_beam_ratio: float = 1.5,
    pre_beam_score_key: str = "full",
) -> list:
    """Perform beam search with scorers.

    Args:
        x (torch.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        scorers (dict[str, ScorerInterface]): Dict of decoder modules
            e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.
        pre_beam_score_key (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search
            will be `int(pre_beam_ratio * beam_size)`

    Returns:
        list: N-best decoding results

    """
    ret = BeamSearch(
        scorers,
        weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=pre_beam_score_key,
        sos=sos,
        eos=eos,
        token_list=token_list,
    ).forward(x=x, maxlenratio=maxlenratio, minlenratio=minlenratio)
    return [h.asdict() for h in ret]
