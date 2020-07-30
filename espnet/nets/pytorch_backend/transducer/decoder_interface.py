"""Transducer decoder interface module."""

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch


class TransducerDecoderInterface:
    """Decoder interface for transducer models."""

    def zero_state(
        self, init_tensor: torch.Tensor = None
    ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
        """Initialize decoder states.

        Args:
            init_tensor: input features

        Returns:
            state: initial state

        """
        raise NotImplementedError("zero_state method is not implemented")

    def forward_one_step(
        self,
        hyp: Dict[str, Union[float, List, torch.Tensor, None]],
        init_tensor: torch.Tensor = None,
    ) -> Union[
        List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]], torch.Tensor
    ]:
        """Forward one step.

        Args:
            hyp: hypothese
            init_tensor: initial tensor for attention computation

        Returns:
            tgt: decoder outputs
            new_state: new decoder state
            att_w: attention weights
            lm_tokens: input token id for LM

        """
        raise NotImplementedError("forward_one_step method is not implemented")

    def forward_batch_one_step(
        self,
        hyps: List[Dict[str, Union[float, List, torch.Tensor, None]]],
        state: Union[List, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        att_w: torch.Tensor = None,
        att_params: List[Union[int, torch.Tensor]] = None,
    ) -> Union[List, Tuple[List[torch.Tensor], List[torch.Tensor]], torch.Tensor]:
        """Forward batch one step.

        Args:
            hyps: batch of hypothesis
            state: batch of decoder states
            att_w : batch of attention weights
            att_params : attention parameters

        Returns:
            tgt: decoder outputs
            new_state: new batch of decoder states
            att_w: new batch of attention weights
            lm_tokens: input token ids for LM

        """
        raise NotImplementedError("forward_batch_one_step method is not implemented")

    def get_idx_dec_state(
        self,
        state: Union[List, Tuple[List[torch.Tensor]]],
        idx: int,
        att_state: torch.Tensor = None,
    ) -> Union[List, Tuple[List[torch.Tensor], List[torch.Tensor]], int]:
        """Get decoder state from batch for given id.

        Args:
            state: batch of decoder states
            idx: index to extract state from beam state

        Returns:
            state: decoder state for id
            att_w: attention weights for given id

        """
        raise NotImplementedError("get_idx_dec_state method is not implemented")

    def get_batch_dec_states(
        self,
        state: List[Union[List, Tuple[List[torch.Tensor], List[torch.Tensor]]]],
        hyps: List[Dict[str, Union[float, List, torch.Tensor, None]]],
    ) -> Union[List, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Create batch of decoder states.

        Args:
            state: list of decoder states
            hyps: batch of hypothesis

        Returns:
            state: batch of decoder states
            att_w : batch of attention weights

        """
        raise NotImplementedError("get_batch_states method is not implemented")
