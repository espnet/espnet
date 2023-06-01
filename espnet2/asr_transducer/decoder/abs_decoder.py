"""Abstract decoder definition for Transducer models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """Abstract decoder module."""

    @abstractmethod
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences.

        Returns:
            : Decoder output sequences.

        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        label_sequence: List[int],
        states: Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ) -> Tuple[
        torch.Tensor,
        Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ]:
        """One-step forward hypothesis.

        Args:
            label_sequence: Current label sequence.
            state: Decoder hidden states.

        Returns:
            out: Decoder output sequence.
            state: Decoder hidden states.

        """
        raise NotImplementedError

    @abstractmethod
    def batch_score(
        self,
        hyps: List[Any],
    ) -> Tuple[
        torch.Tensor,
        Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
    ]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            out: Decoder output sequences.
            states: Decoder hidden states.

        """
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: torch.Tensor) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        raise NotImplementedError

    @abstractmethod
    def init_state(
        self, batch_size: int
    ) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.tensor]],
    ]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Decoder hidden states.

        """
        raise NotImplementedError

    @abstractmethod
    def select_state(
        self,
        states: Union[
            List[Dict[str, torch.Tensor]],
            List[torch.Tensor],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
        idx: int = 0,
    ) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor]],
    ]:
        """Get specified ID state from batch of states, if provided.

        Args:
            states: Decoder hidden states.
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID.

        """
        raise NotImplementedError

    @abstractmethod
    def create_batch_states(
        self,
        new_states: List[
            Union[
                List[Dict[str, Optional[torch.Tensor]]],
                List[List[torch.Tensor]],
                Tuple[torch.Tensor, Optional[torch.Tensor]],
            ],
        ],
    ) -> Union[
        List[Dict[str, torch.Tensor]],
        List[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor]],
    ]:
        """Create batch of decoder hidden states given a list of new states.

        Args:
            new_states: Decoder hidden states.

        Returns:
            : Decoder hidden states.

        """
        raise NotImplementedError
