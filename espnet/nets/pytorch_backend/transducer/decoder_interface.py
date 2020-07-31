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
        init_tensor: torch.Tensor = None,
    ) -> Union[Tuple[Any], torch.Tensor]:
        """Forward one step.

        Args:
            hyp: hypothese
            init_tensor: initial tensor for attention decoder

        Returns:
            tgt: decoder outputs
            (tuple): decoder and attention states
            lm_tokens: input token id for LM

        """
        raise NotImplementedError("forward_one_step method is not implemented")

    def batch_score(
        self,
        hyps: List[Dict[str, Union[float, List, torch.Tensor, None]]],
        state: Tuple[Any],
    ) -> Union[Tuple[Any], torch.Tensor]:
        """Forward batch one step.

        Args:
            hyps: batch of hypothesis
            state (tuple): batch of decoder and attention states

        Returns:
            tgt: decoder outputs
            (tuple): batch of decoder and attention states
            lm_tokens: input token ids for LM

        """
        raise NotImplementedError("forward_batch_one_step method is not implemented")

    def select_state(self, state: Tuple[Any], idx: int) -> Tuple[Any]:
        """Get decoder state from batch for given id.

        Args:
            state: batch of decoder states
            idx: index to extract state from beam state

        Returns:
            state: decoder and attention state for given id

        """
        raise NotImplementedError("get_idx_dec_state method is not implemented")

    def create_batch_state(
        self,
        state: Tuple[Any],
        hyps: List[Dict[str, Union[float, List, torch.Tensor, None]]],
    ) -> Tuple[Any]:
        """Create batch of decoder states.

        Args:
            state: list of decoder states
            hyps: batch of hypothesis

        Returns:
            state: batch of decoder and attention states

        """
        raise NotImplementedError("get_batch_states method is not implemented")
