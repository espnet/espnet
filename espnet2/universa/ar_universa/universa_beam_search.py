"""Beam search module."""

import copy
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from typeguard import typechecked

logger = logging.getLogger(__name__)


def deep_copy_without_element(lst, element_to_remove):
    # Create a deep copy of the original list
    new_list = copy.deepcopy(lst)

    # Remove the element if it exists in the top level
    if element_to_remove in new_list:
        new_list.remove(element_to_remove)

    return new_list


class Hypothesis(NamedTuple):
    """Hypothesis data type.

    Attributes:
        yseq: Sequence of token IDs.
        score: Score of the hypothesis.
        scores: Dictionary of scores for each token in the sequence.
        states: Dictionary of additional states.
    """

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()
    unused_meta_label_ids: List[int] = None


class ARUniVERSABeamSearch:
    """Beam search module for ARUniVERSA models."""

    @typechecked
    def __init__(
        self,
        scorers: Dict[
            str, Any
        ],  # NOTE(jiatong): need a better type (beyond ScorerInterface)
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        meta_label_for_search: List[int],
        token_list: Optional[List[str]] = None,
        skip_meta_label_score: bool = False,
        beam_masking: Optional[Dict[int, Tuple[int, int]]] = None,
        use_fixed_order: bool = False,
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
            meta_label_for_search (list[int]): List of meta label ids for search
            token_list (list[str]): List of tokens for debug log
            skip_meta_label_score (bool): Whether to extend without scorer.
                If True, the beam search will be performed without scoring.

        """
        super().__init__()
        # set scorers
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if len(set(meta_label_for_search)) != len(meta_label_for_search):
            raise ValueError("Requested metrics must be unique")
        self.use_fixed_order = use_fixed_order
        self.weights = weights
        self.meta_label_for_search = meta_label_for_search
        self.skip_meta_label_score = skip_meta_label_score
        self.beam_masking = beam_masking
        self.scorers = dict()

        # this module dict is required for recursive cast
        # `self.to(device, dtype)` in `recog.py`
        self.nn_dict = torch.nn.ModuleDict()
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            self.scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos  # NOTE(jiatong): this is not used in the current implementation

        self.token_list = token_list
        self.beam_size = beam_size
        self.n_vocab = vocab_size

    def init_hyp(self, x: torch.Tensor) -> Hypothesis:
        """Initialize a hypothesis.

        Args:
            x (torch.Tensor): Encoder output tensor.

        Returns:
            Hypothesis: Initialized hypothesis.
        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0

        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=torch.tensor([self.sos], device=x.device),
                unused_meta_label_ids=copy.deepcopy(self.meta_label_for_search),
            )
        ]

    @typechecked
    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append

        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        new_x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, new_x))

    @typechecked
    def score(
        self,
        hyp: Hypothesis,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.scorers.items():
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)

        return scores, states

    @typechecked
    def beam(
        self,
        weighted_scores: torch.Tensor,
        ids: torch.Tensor,
        use_id_size: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
            Its shape is `(self.n_vocab,)`.
            ids (torch.Tensor): The partial token ids to compute topk
            use_id_size (bool): Whether to use the size of ids for topk.
                If True, the topk will be computed based on the size of ids.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`

        """
        size = ids.numel() if use_id_size else min(self.beam_size, ids.numel())
        local_ids = weighted_scores[ids].topk(size).indices
        return ids[local_ids], local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_scores: Dict[str, torch.Tensor],
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            idx (int): The next token id for `next_full_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_scores.items():
            new_scores[k] = prev_scores[k] + v[idx]
        return new_scores

    def extend(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
    ) -> List[Hypothesis]:
        """Extend the hypotheses with the list of meta label ids.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
        """
        extended_hyps = []
        for hyp in running_hyps:
            labels = hyp.unused_meta_label_ids
            if self.use_fixed_order:
                labels = labels[:1]
            part_ids = torch.tensor(labels, device=x.device)
            if self.skip_meta_label_score:
                weighted_scores = torch.zeros(
                    self.n_vocab, dtype=x.dtype, device=x.device
                )
                scores, states = self.score(hyp, x)
                for k in self.scorers:
                    scores[k] = torch.zeros_like(scores[k])
                weighted_scores += hyp.score
                use_id_size = True
            else:
                # scoring
                weighted_scores = torch.zeros(
                    self.n_vocab, dtype=x.dtype, device=x.device
                )
                scores, states = self.score(hyp, x)
                for k in self.scorers:
                    weighted_scores += self.weights[k] * scores[k]

                # add previous hyp score
                weighted_scores += hyp.score
                use_id_size = False

            # update hyps
            for j, _ in zip(*self.beam(weighted_scores, part_ids, use_id_size)):
                j = int(j)
                # will be (2 x beam at most)
                extended_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(hyp.scores, scores, j),
                        states=states,
                        unused_meta_label_ids=deep_copy_without_element(
                            hyp.unused_meta_label_ids, j
                        ),
                    )
                )
        return extended_hyps

    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        extended_running_hyps = []

        extended_running_hyps.extend(self.extend(running_hyps, x))
        for hyp in extended_running_hyps:
            # scoring
            weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
            scores, states = self.score(hyp, x)
            for k in self.scorers:
                weighted_scores += self.weights[k] * scores[k]

            # add previous hyp score
            weighted_scores += hyp.score

            # NOTE(jiatong): conduct pre-beam with value tokens based on metric meta
            # label ids
            if self.beam_masking is None:
                part_ids = torch.arange(self.n_vocab, device=x.device)
            else:
                token_range = self.beam_masking.get(int(hyp.yseq[-1]), None)
                if token_range is None:
                    part_ids = torch.arange(self.n_vocab, device=x.device)
                else:
                    start_idx, end_idx = token_range
                    part_ids = torch.arange(start_idx, end_idx, device=x.device)

            # update hyps
            for j, _ in zip(*self.beam(weighted_scores, part_ids)):
                j = int(j)
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(hyp.scores, scores, j),
                        states=states,
                        unused_meta_label_ids=deep_copy_without_element(
                            hyp.unused_meta_label_ids, j
                        ),
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
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        inp = x
        logger.info("decoder input length: " + str(inp.shape[0]))
        logger.info(
            "expected output length: " + str(len(self.meta_label_for_search) * 2)
        )

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        for i in range(len(self.meta_label_for_search)):
            logger.debug("position " + str(i) + " and position " + str(i + 1))
            # extend hypotheses
            running_hyps = self.search(running_hyps, x)

        nbest_hyps = sorted(running_hyps, key=lambda x: x.score, reverse=True)

        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning("there is no N-best results, likely due to a bug")
            return []

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            v = float(v) if isinstance(v, torch.Tensor) else v
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )

        score = (
            float(best.score) if isinstance(best.score, torch.Tensor) else best.score
        )
        logger.info(f"total log probability: {score:.2f}")
        logger.info(f"normalized log probability: {score / len(best.yseq):.2f}")
        if self.token_list is not None:
            logger.info(
                "best hypo: "
                + " ".join([self.token_list[x] for x in best.yseq[1:]])
                + "\n"
            )
        return nbest_hyps
