"""RNN decoder definition for Transducer model."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from espnet.nets.transducer_decoder_interface import ExtendedHypothesis
from espnet.nets.transducer_decoder_interface import Hypothesis
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class RNNDecoder(TransducerDecoderInterface, torch.nn.Module):
    """RNN decoder module for Transducer model.

    Args:
        odim: Output dimension.
        dtype: Decoder units type.
        dlayers: Number of decoder layers.
        dunits: Number of decoder units per layer..
        embed_dim: Embedding layer dimension.
        dropout_rate: Dropout rate for decoder layers.
        dropout_rate_embed: Dropout rate for embedding layer.
        blank_id: Blank symbol ID.

    """

    def __init__(
        self,
        odim: int,
        dtype: str,
        dlayers: int,
        dunits: int,
        embed_dim: int,
        dropout_rate: float = 0.0,
        dropout_rate_embed: float = 0.0,
        blank_id: int = 0,
    ):
        """Transducer initializer."""
        super().__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank_id)
        self.dropout_embed = torch.nn.Dropout(p=dropout_rate_embed)

        dec_net = torch.nn.LSTM if dtype == "lstm" else torch.nn.GRU

        self.decoder = torch.nn.ModuleList(
            [dec_net(embed_dim, dunits, 1, batch_first=True)]
        )
        self.dropout_dec = torch.nn.Dropout(p=dropout_rate)

        for _ in range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits, 1, batch_first=True)]

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype

        self.odim = odim

        self.ignore_id = -1
        self.blank_id = blank_id

        self.multi_gpus = torch.cuda.device_count() > 1

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
            self.dunits,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

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
                ) = self.decoder[layer](
                    sequence, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                sequence, h_next[layer : layer + 1] = self.decoder[layer](
                    sequence, hx=h_prev[layer : layer + 1]
                )

            sequence = self.dropout_dec(sequence)

        return sequence, (h_next, c_next)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, T, U, D_dec)

        """
        init_state = self.init_state(labels.size(0))
        dec_embed = self.dropout_embed(self.embed(labels))

        dec_out, _ = self.rnn_forward(dec_embed, init_state)

        return dec_out

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, state) for each label sequence. (key)

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
            label: Label ID for LM. (1,)

        """
        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_emb = self.embed(label)

            dec_out, dec_state = self.rnn_forward(dec_emb, hyp.dec_state)
            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0][0], dec_state, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            cache: Pairs of (dec_out, dec_states) for each label sequences. (keys)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            dec_states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            lm_labels: Label ID sequences for LM. (B,)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_labels = "_".join(list(map(str, hyp.yseq)))

            if str_labels in cache:
                done[i] = cache[str_labels]
            else:
                process.append((str_labels, hyp.yseq[-1], hyp.dec_state))

        if process:
            labels = torch.LongTensor([[p[1]] for p in process], device=self.device)
            p_dec_states = self.create_batch_states(
                self.init_state(labels.size(0)), [p[2] for p in process]
            )

            dec_emb = self.embed(labels)
            dec_out, new_states = self.rnn_forward(dec_emb, p_dec_states)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                state = self.select_state(new_states, j)

                done[i] = (dec_out[j], state)
                cache[process[j][0]] = (dec_out[j], state)

                j += 1

        dec_out = torch.cat([d[0] for d in done], dim=0)
        dec_states = self.create_batch_states(dec_states, [d[1] for d in done])

        if use_lm:
            lm_labels = torch.LongTensor([h.yseq[-1] for h in hyps], device=self.device)

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None

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
        check_list: Optional[List] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
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
