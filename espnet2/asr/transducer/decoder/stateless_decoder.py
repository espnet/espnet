"""Stateless decoder definition for Transducer models."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet2.asr.transducer.decoder.abs_decoder import AbsDecoder


class StatelessDecoder(AbsDecoder):
    """Stateless Transducer decoder module.

    Args:
        vocab_size: Output dimension.
        hidden_size: Number of embedding units.
        dropout_embed: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout_embed: float = 0.0,
        embed_pad: int = 0,
    ):
        assert check_argument_types()

        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        self.dunits = hidden_size
        self.odim = vocab_size

        self.ignore_id = -1
        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: int) -> None:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. None

        """
        return None

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, U, D_dec)

        """
        dec_embed = self.dropout_embed(self.embed(labels))

        return dec_embed

    def score(
        self, hyp: Hypothesis, cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Decoder output for each seen label sequence.

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            dec_state: Decoder hidden state. None
            label: Label ID for LM. (1,)

        """
        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_label = str(hyp.yseq[-1])

        if str_label in cache:
            dec_out = cache[str_label]
        else:
            dec_out = self.embed(label)

            cache[str_label] = dec_out

        return dec_out[0][0], None, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: None,
        cache: Dict[str, torch.Tensor],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            dec_states: Decoder hidden states. None
            cache: Decoder output for each seen label sequences.
            use_lm: Whether to compute label IDs for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            : Decoder hidden states. None
            lm_labels: Label IDs for LM. (B,)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_label = str(hyp.yseq[-1])

            if str_label in cache:
                done[i] = cache[str_label]
            else:
                process.append((str_label, hyp.yseq[-1]))

        if process:
            labels = torch.LongTensor([[p[1]] for p in process], device=self.device)
            dec_out = self.embed(labels)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                done[i] = dec_out[j]
                cache[process[j][0]] = dec_out[j]

                j += 1

        dec_out = torch.cat([d[0] for d in done], dim=0)

        if use_lm:
            lm_labels = torch.LongTensor(
                [h.yseq[-1] for h in hyps], device=self.device
            ).view(final_batch, 1)

            return dec_out, None, lm_labels

        return dec_out, None, None

    def select_state(self, states: None) -> None:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. None
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID. None

        """
        return None

    def create_batch_states(self, states: None, new_states: List[None]) -> None:
        """Create decoder hidden states.

        Args:
            states: Decoder hidden states. None
            new_states: Decoder hidden states. [N x None]

        Returns:
            states: Decoder hidden states. None

        """
        return None
