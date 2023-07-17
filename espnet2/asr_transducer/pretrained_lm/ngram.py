"""Pretrained token-level N-gram model implementation."""

from typing import List, Tuple

try:
    import kenlm
except ImportError:
    logging.error(
        "KenLM was not installed. LM scoring with a n-gram can't be performed."
    )
import numpy as np
import torch


class PretrainedTokenNgram:
    """Pretrained token-level N-gram model.

    Args:
        model_path: Model path.
        token_list: List of known tokens.
        score_weight: Weight for the outputted log-probabilities.
        device: Device to pin the parameters on.

    """

    def __init__(
        self,
        model_path: str,
        token_list: List[str],
        score_weight: float,
        device: str,
    ) -> None:
        super().__init__()

        self.lm = kenlm.Model(model_path)
        self.lm_state = kenlm.State()
        self.log10_to_ln = 1 / np.log10(np.e)

        self.score_weight = score_weight

        self.device = device

        self.token_list = [x if x != "<eos>" else "</s>" for x in token_list]
        self.num_tokens = len(self.token_list)

        self.sos_id = self.num_tokens - 1

    def zero_state(self) -> List["kenlm.State"]:
        """Initialize KenLM state with null context.

        Args:
            None

        Returns:
            : KenLM state with null context.

        """
        state = kenlm.State()

        return self.lm.NullContextWrite(state)

    def score(
        self,
        label: torch.Tensor,
        state: "kenlm.State",
    ) -> Tuple[torch.Tensor, "kenlm.State"]:
        """Perform LM full scoring given an input label.

        Args:
            label: Previous label. (1, 1)
            state: Previous KenLM state.

        Returns:
            lm_logp: Weighted Log-probabilities. (vocab_size)
            new_state: kenLM state.

        """
        label = self.token_list[label]
        new_state = kenlm.State()

        lm_logp = torch.zeros((self.num_tokens), device=self.device)

        self.lm.BaseScore(state, label, new_state)

        for i in range(1, self.num_tokens):
            lm_logp[i] = (
                self.lm.BaseScore(
                    new_state,
                    self.token_list[i],
                    self.lm_state,
                )
                * self.log10_to_ln
            )

        return (self.score_weight * lm_logp), new_state

    def batch_score(
        self, labels: List[int], states: List["kenlm.State"]
    ) -> Tuple[torch.Tensor, List["kenlm.State"]]:
        """Perform LM full scoring given a batch of labels.

        Args:
            labels: Previous labels. [B]
            states: Previous KenLM states [B]

        Returns:
            lm_logp: KenLM log-probabilities. (B, vocab_size)
            states: KenLM states. [B]

        """
        batch = len(labels)
        new_states = [kenlm.State() for i in range(batch)]

        lm_logp = torch.zeros((batch, self.num_tokens), device=self.device)

        for i in range(batch):
            self.lm.BaseScore(states[i], self.token_list[int(labels[i])], new_states[i])

            for j in range(1, self.num_tokens):
                lm_logp[i, j] = (
                    self.lm.BaseScore(new_states[i], self.token_list[j], self.lm_state)
                    * self.log10_to_ln
                )

        return (self.score_weight * lm_logp), new_states

    def select_state(self, state: List["kenlm.State"], idx: int) -> "kenlm.State":
        """Get specified ID state from KenLM states.

        Args:
            state: KenLM state. [B]
            idx: State ID to extract.

        Returns:
            : KenLM state for given ID.

        """
        return state[idx]
