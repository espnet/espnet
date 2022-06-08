"""Transducer decoder interface module."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class TransducerDecoderInterface:
    """Decoder interface for Transducer models."""

    def init_state(
        self,
        batch_size: int,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            state: Initial decoder hidden states.

        """
        raise NotImplementedError("init_state(...) is not implemented")

    def score(
        self,
        hyp: Hypothesis,
        cache: Dict[str, Any],
    ) -> Tuple[
        torch.Tensor,
        Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        torch.Tensor,
    ]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, dec_state) for each token sequence. (key)

        Returns:
            dec_out: Decoder output sequence.
            new_state: Decoder hidden states.
            lm_tokens: Label ID for LM.

        """
        raise NotImplementedError("score(...) is not implemented")

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[
        torch.Tensor,
        Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        torch.Tensor,
    ]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            dec_states: Decoder hidden states.
            cache: Pairs of (dec_out, dec_states) for each label sequence. (key)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences.
            dec_states: Decoder hidden states.
            lm_labels: Label ID sequences for LM.

        """
        raise NotImplementedError("batch_score(...) is not implemented")

    def select_state(
        self,
        batch_states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[torch.Tensor]
        ],
        idx: int,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Get specified ID state from decoder hidden states.

        Args:
            batch_states: Decoder hidden states.
            idx: State ID to extract.

        Returns:
            state_idx: Decoder hidden state for given ID.

        """
        raise NotImplementedError("select_state(...) is not implemented")

    def create_batch_states(
        self,
        states: Union[
            Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
        ],
        new_states: List[
            Union[
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                List[Optional[torch.Tensor]],
            ]
        ],
        l_tokens: List[List[int]],
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        """Create decoder hidden states.

        Args:
            batch_states: Batch of decoder states
            l_states: List of decoder states
            l_tokens: List of token sequences for input batch

        Returns:
            batch_states: Batch of decoder states

        """
        raise NotImplementedError("create_batch_states(...) is not implemented")
