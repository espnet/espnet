"""Pretrained token-level language model implementation."""

from typing import Dict, List, Tuple

import torch


class PretrainedTokenLSTM(torch.nn.Module):
    """Pretrained token-level LSTM LM.

    Args:
        vocab_size: Vocabulary size.
        embed_size: Embedding size.
        hidden_size: LSTM hidden size.
        num_layers: Number of layers.
        state_dict: Pre-trained token LSTM LM state dict.
        score_weight: Weight for the outputted log-probabilities.
        device: Device to pin the parameters on.
        padding_id: Padding symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        state_dict: Dict,
        score_weight: float,
        device: torch.device,
        padding_id: int = 0,
    ) -> None:
        super().__init__()

        self.encoder = torch.nn.Embedding(
            vocab_size, embed_size, padding_idx=padding_id
        )

        self.rnn = torch.nn.LSTM(
            embed_size, hidden_size, num_layers, dropout=0.0, batch_first=True
        )

        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rescore_weight = rescore_weight

        self.device = device

        self.sos_id = vocab_size - 1

        self.load_state_dict(model_state_dict)

    def zero_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create LSTM hidden state filled with zero values.

        Args:
            batch_size: Batch size.

        Returns:
            state: LSTM state hidden with zero values.
                       ((L, B, D_hidden), (L, B, D_hidden))

        """
        h = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=torch.float,
            device=self.device,
        )
        c = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=torch.float,
            device=self.device,
        )

        return (h, c)

    def forward(
        self,
        labels: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """LM computation.

        Args:
            labels: Label ID sequences. (B, L)
            states: LSTM states. ((L, B, D_hidden), (L, B, D_hidden))

        Returns:
           : Weighted LM log-probabilities. (B, vocab_size)
           states: LSTM hidden states. ((L, B, D_hidden), (L, B, D_hidden))

        """
        embed = self.encoder(labels)
        lm_out, state = self.rnn(embed, state)

        lm_out = self.decoder(
            lm_out.contiguous().view(lm_out.size(0) * lm_out.size(1), lm_out.size(2))
        )

        return lm_out, state

    def score(
        self,
        label: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute LM scores given an input label.

        Args:
            label: Previous label. (1, 1)
            state: Previous LM hidden states. ((L, 1, D_hidden), (L, 1, D_hidden))

        Returns:
           lm_logp: Weighted LM log-probability given the input label. (D_vocab)
           state: LM hidden state. ((L, 1, D_hidden), (L, 1, D_hidden))

        """
        label = torch.full(
            (1, 1),
            self.sos_id if label == 0 else label,
            dtype=torch.long,
            device=state[0].device,
        )

        lm_out, state = self(label, state)

        lm_logp = torch.log_softmax(lm_out, dim=-1)
        lm_logp[..., 0] = 0

        return (self.score_weight * lm_logp.squeeze(0)), state

    def batch_score(
        self,
        labels: List[int],
        states: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute LM scores given a batch of labels.

        Args:
            labels: Previous labels. [B]
            states: Previous LM hidden states.
                        [B x ((L, 1, D_hidden), (L, 1, D_hidden))]

        Returns:
            lm_logp: Weighted LM log-probabilities. (B, vocab_size)
            states: LM hidden state. ((L, B, D_hidden), (L, B, D_hidden))

        """
        labels = torch.tensor(
            [[self.sos_id] if i == 0 else [i] for i in labels],
            dtype=torch.long,
            device=self.device,
        )

        states = (
            torch.cat([s[0] for s in states], dim=1),
            torch.cat([s[1] for s in states], dim=1),
        )

        lm_out, states = self(labels, states)

        lm_logp = torch.log_softmax(lm_out, dim=-1)
        lm_logp[..., 0] = 0

        return (self.score_weight * lm_logp), states

    def select_state(
        self, state: Tuple[torch.Tensor, torch.Tensor], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get specified ID state from LSTM hidden states.

        Args:
            state: LSTM hidden state. ((L, B, D_hidden), (L, B, D_hidden))
            idx: State ID to extract.

        Returns:
            : LSTM hidden state for given ID. ((L, 1, D_hidden), (L, 1, D_hidden))

        """
        return (
            state[0][:, idx : idx + 1, :],
            state[1][:, idx : idx + 1, :],
        )
