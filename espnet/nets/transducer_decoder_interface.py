"""Transducer decoder interface module."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch


class TransducerDecoderInterface:
    """Decoder interface for transducer models."""

    def init_state(self, init_tensor: torch.Tensor = None) -> Any:
        """Initialize decoder and attention states.

        Args:
            init_tensor: input features

        Returns:
            state: initial state

        """
        raise NotImplementedError("zero_state method is not implemented")

    def score(
        self,
        hyp: Dict[str, Union[float, List, torch.Tensor, None]],
        cache: Dict[str, Any],
        init_tensor: torch.Tensor = None,
    ) -> Union[Tuple[Any], torch.Tensor]:
        """Forward one step.

        Args:
            hyp: hypothese
            cache: pairs of (y, state) for each token sequence (key)
            init_tensor: initial tensor for attention decoder

        Returns:
            y: decoder outputs
            (tuple): decoder and attention states
            lm_tokens: token id for LM

        """
        raise NotImplementedError("forward_one_step method is not implemented")

    def batch_score(
        self,
        hyps: List[Dict[str, Union[float, List, torch.Tensor, None]]],
        batch_states: Tuple[Any],
        cache: Dict[str, Any],
    ) -> Union[Tuple[Any], torch.Tensor]:
        """Forward batch one step.

        Args:
            hyps: batch of hypothesis
            batch_states (tuple): batch of decoder and attention states
            cache: pairs of (y, state) for each token sequence (key)
            init_tensor: initial tensor for attention decoder

        Returns:
            batch_y : decoder outputs
            batch_states (tuple): batch of decoder and attention states
            lm_tokens: batch of token ids for LM

        """
        raise NotImplementedError("forward_batch_one_step method is not implemented")

    def select_state(self, batch_states: Tuple[Any], idx: int) -> Tuple[Any]:
        """Get decoder state from batch for given id.

        Args:
            batch_states: batch of decoder and attention states
            idx: index to extract state from batch of states

        Returns:
            (tuple): decoder and attention state for given id

        """
        raise NotImplementedError("get_idx_dec_state method is not implemented")

    def create_batch_states(
        self,
        batch_states: Tuple[Any],
        l_states: List[Tuple[Any]],
        l_tokens: List[List[int]] = None,
    ) -> Tuple[Any]:
        """Create batch of decoder states.

        Args:
            batch_states: batch of decoder and attentions states
            l_states: list of decoder and attention states
            l_tokens: list of token sequences for batch

        Returns:
            batch_states: batch of decoder and attention states

        """
        raise NotImplementedError("create_batch_states method is not implemented")
