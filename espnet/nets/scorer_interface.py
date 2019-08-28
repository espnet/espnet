"""Scorer interface module."""

from typing import Any
from typing import Tuple

import torch


class ScorerInterface:
    """Scorer interface for beam search.

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`

    """

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.

    The partial scorer performs scoring when non-partial scorer finished scoring,
    and recieves pre-pruned next tokens to score because it is too heavy to score
    all the tokens.

    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`

    """

    def select_state(self, state: Any, i: int) -> Any:
        """Select state with relative ids in the main beam search (required).

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search

        Returns:
            state: pruned state

        """
        raise NotImplementedError

    def score_partial(self, y: torch.Tensor, next_tokens: torch.Tensor, state: Any, x: torch.Tensor) \
            -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]: Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        raise NotImplementedError
