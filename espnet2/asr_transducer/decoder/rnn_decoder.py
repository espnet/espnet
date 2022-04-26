"""RNN decoder definition for Transducer models."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet2.asr_transducer.beam_search_transducer import ExtendedHypothesis
from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder


class RNNDecoder(AbsDecoder):
    """RNN decoder module.

    Args:
        dim_vocab: Output dimension.
        dim_embedding: Embedding dimension.
        dim_hidden: Hidden dimension.
        rnn_type: Decoder layers type.
        num_layers: Number of decoder layers.
        dropout: Dropout rate for decoder layers.
        dropout_embed: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    def __init__(
        self,
        dim_vocab: int,
        dim_embedding: int = 256,
        dim_hidden: int = 256,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
        dropout_embed: float = 0.0,
        embed_pad: int = 0,
    ):
        assert check_argument_types()

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()

        self.embed = torch.nn.Embedding(dim_vocab, dim_embedding, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        rnn_class = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU

        self.rnn = torch.nn.ModuleList(
            [rnn_class(dim_embedding, dim_hidden, 1, batch_first=True)]
        )

        for _ in range(1, num_layers):
            self.rnn += [rnn_class(dim_hidden, dim_hidden, 1, batch_first=True)]

        self.dropout_rnn = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for _ in range(num_layers)]
        )

        self.dlayers = num_layers
        self.dtype = rnn_type

        self.dim_output = dim_hidden
        self.dim_vocab = dim_vocab

        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def forward(
        self,
        labels: torch.Tensor,
        states: Tuple[torch.Tensor, Optional[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, U, D_dec)
            states: Decoder states. ((N, 1, D_dec), (N, 1, D_dec))

        """
        if states is None:
            states = self.init_state(labels.size(0))

        dec_embed = self.dropout_embed(self.embed(labels))
        dec_out, states = self.rnn_forward(dec_embed, states)

        return dec_out

    def rnn_forward(
        self,
        sequence: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Encode source label sequences.

        Args:
            sequence: RNN input sequences. (B, D_emb)
            state: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        Returns:
            sequence: RNN output sequences. (B, D_dec)
            (h_next, c_next): Decoder hidden states. (N, B, D_dec), (N, B, D_dec))

        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(sequence.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                sequence, (
                    h_next[layer : layer + 1],
                    c_next[layer : layer + 1],
                ) = self.rnn[layer](
                    sequence, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                sequence, h_next[layer : layer + 1] = self.rnn[layer](
                    sequence, hx=h_prev[layer : layer + 1]
                )

            sequence = self.dropout_rnn[layer](sequence)

        return sequence, (h_next, c_next)

    def score(
        self,
        label: torch.Tensor,
        label_sequence: List[int],
        dec_state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            label: Previous label. (1, 1)
            label_sequence: Current label sequence.
            dec_state: Previous decoder hidden states.
                       ((N, 1, D_dec), (N, 1, D_dec)) or None
            cache: Pairs of (dec_out, state) for each label sequence (key).

        Returns:
            dec_out: Decoder output sequence. (1, D_dec) or None
            dec_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))

        """
        str_labels = "_".join(list(map(str, label_sequence)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_embed = self.embed(label)
            dec_out, dec_state = self.rnn_forward(dec_embed, dec_state)

            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0], dec_state

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        labels = torch.LongTensor([[h.yseq[-1]] for h in hyps], device=self.device)
        dec_embed = self.embed(labels)

        states = self.create_batch_states(
            self.init_state(labels.size(0)), [h.dec_state for h in hyps]
        )
        dec_out, states = self.rnn_forward(dec_embed, states)

        return dec_out.squeeze(1), states

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dim_output,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dim_output,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def select_state(
        self, states: Tuple[torch.Tensor, Optional[torch.Tensor]], idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID.
              ((N, 1, D_dec), (N, 1, D_dec))

        """
        return (
            states[0][:, idx : idx + 1, :],
            states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

    def create_batch_states(
        self,
        states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        new_states: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create decoder hidden states.

        Args:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            new_states: Decoder hidden states. [N x ((1, D_dec), (1, D_dec))]

        Returns:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        return (
            torch.cat([s[0] for s in new_states], dim=1),
            torch.cat([s[1] for s in new_states], dim=1)
            if self.dtype == "lstm"
            else None,
        )
