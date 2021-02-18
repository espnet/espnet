"""Transducer decoder interface module."""

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch


@dataclass
class Hypothesis:
    """Default hypothesis definition for beam search."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[torch.Tensor], torch.Tensor
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class NSCHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search."""

    y: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class TransducerDecoderInterface:
    """Decoder interface for transducer models."""

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size for initial state
            device: Device for initial state

        Returns:
            state: Initialized state

        """
        raise NotImplementedError("init_state method is not implemented")

    def score(
        self,
        hyp: Union[Hypothesis, NSCHypothesis],
        cache: Dict[str, Any],
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        torch.Tensor,
        List[Optional[torch.Tensor]],
    ]:
        """Forward one hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (y, state) for each token sequence (key)

        Returns:
            y: Decoder outputs
            new_state: New decoder state
            lm_tokens: Token id for LM

        """
        raise NotImplementedError("score method is not implemented")

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[NSCHypothesis]],
        batch_states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        cache: Dict[str, Any],
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        torch.Tensor,
        List[Optional[torch.Tensor]],
    ]:
        """Forward batch of hypotheses.

        Args:
            hyps: Batch of hypotheses
            batch_states: Batch of decoder states
            cache: pairs of (y, state) for each token sequence (key)

        Returns:
            batch_y: Decoder outputs
            batch_states: Batch of decoder states
            lm_tokens: Batch of token ids for LM

        """
        raise NotImplementedError("batch_score method is not implemented")

    def select_state(
        self,
        batch_states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        idx: int,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Get decoder state from batch for given id.

        Args:
            batch_states: Batch of decoder states
            idx: Index to extract state from batch

        Returns:
            state_idx: Decoder state for given id

        """
        raise NotImplementedError("select_state method is not implemented")

    def create_batch_states(
        self,
        batch_states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        l_states: List[
            Union[
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                List[Optional[torch.Tensor]],
            ]
        ],
        l_tokens: List[List[int]],
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Create batch of decoder states.

        Args:
            batch_states: Batch of decoder states
            l_states: List of decoder states
            l_tokens: List of token sequences for input batch

        Returns:
            batch_states: Batch of decoder states

        """
        raise NotImplementedError("create_batch_states method is not implemented")
