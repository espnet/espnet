"""Custom decoder definition for Transducer model."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.utils import (check_batch_states,
                                                          check_state,
                                                          pad_sequence)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.transducer_decoder_interface import (
    ExtendedHypothesis, Hypothesis, TransducerDecoderInterface)


class CustomDecoder(TransducerDecoderInterface, torch.nn.Module):
    """Custom decoder module for Transducer model.

    Args:
        odim: Output dimension.
        dec_arch: Decoder block architecture (type and parameters).
        input_layer: Input layer type.
        repeat_block: Number of times dec_arch is repeated.
        joint_activation_type: Type of activation for joint network.
        positional_encoding_type: Positional encoding type.
        positionwise_layer_type: Positionwise layer type.
        positionwise_activation_type: Positionwise activation type.
        input_layer_dropout_rate: Dropout rate for input layer.
        blank_id: Blank symbol ID.

    """

    def __init__(
        self,
        odim: int,
        dec_arch: List,
        input_layer: str = "embed",
        repeat_block: int = 0,
        joint_activation_type: str = "tanh",
        positional_encoding_type: str = "abs_pos",
        positionwise_layer_type: str = "linear",
        positionwise_activation_type: str = "relu",
        input_layer_dropout_rate: float = 0.0,
        blank_id: int = 0,
    ):
        """Construct a CustomDecoder object."""
        torch.nn.Module.__init__(self)

        self.embed, self.decoders, ddim, _ = build_blocks(
            "decoder",
            odim,
            input_layer,
            dec_arch,
            repeat_block=repeat_block,
            positional_encoding_type=positional_encoding_type,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_activation_type=positionwise_activation_type,
            input_layer_dropout_rate=input_layer_dropout_rate,
            padding_idx=blank_id,
        )

        self.after_norm = LayerNorm(ddim)

        self.dlayers = len(self.decoders)
        self.dunits = ddim
        self.odim = odim

        self.blank_id = blank_id

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(
        self,
        batch_size: Optional[int] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            state: Initial decoder hidden states. [N x None]

        """
        state = [None] * self.dlayers

        return state

    def forward(
        self, dec_input: torch.Tensor, dec_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode label ID sequences.

        Args:
            dec_input: Label ID sequences. (B, U)
            dec_mask: Label mask sequences.  (B, U)

        Return:
            dec_output: Decoder output sequences. (B, U, D_dec)
            dec_output_mask: Mask of decoder output sequences. (B, U)

        """
        dec_input = self.embed(dec_input)

        dec_output, dec_mask = self.decoders(dec_input, dec_mask)
        dec_output = self.after_norm(dec_output)

        return dec_output, dec_mask

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, dec_state) for each label sequence. (key)

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            dec_state: Decoder hidden states. [N x (1, U, D_dec)]
            lm_label: Label ID for LM. (1,)

        """
        labels = torch.tensor([hyp.yseq], device=self.device)
        lm_label = labels[:, -1]

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_out_mask = subsequent_mask(len(hyp.yseq)).unsqueeze_(0)

            new_state = check_state(hyp.dec_state, (labels.size(1) - 1), self.blank_id)

            dec_out = self.embed(labels)

            dec_state = []
            for s, decoder in zip(new_state, self.decoders):
                dec_out, dec_out_mask = decoder(dec_out, dec_out_mask, cache=s)
                dec_state.append(dec_out)

            dec_out = self.after_norm(dec_out[:, -1])

            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0], dec_state, lm_label

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: List[Optional[torch.Tensor]],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            dec_states: Decoder hidden states. [N x (B, U, D_dec)]
            cache: Pairs of (h_dec, dec_states) for each label sequences. (keys)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            dec_states: Decoder hidden states. [N x (B, U, D_dec)]
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
                process.append((str_labels, hyp.yseq, hyp.dec_state))

        if process:
            labels = pad_sequence([p[1] for p in process], self.blank_id)
            labels = torch.LongTensor(labels, device=self.device)

            p_dec_states = self.create_batch_states(
                self.init_state(),
                [p[2] for p in process],
                labels,
            )

            dec_out = self.embed(labels)

            dec_out_mask = (
                subsequent_mask(labels.size(-1))
                .unsqueeze_(0)
                .expand(len(process), -1, -1)
            )

            new_states = []
            for s, decoder in zip(p_dec_states, self.decoders):
                dec_out, dec_out_mask = decoder(dec_out, dec_out_mask, cache=s)
                new_states.append(dec_out)

            dec_out = self.after_norm(dec_out[:, -1])

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                state = self.select_state(new_states, j)

                done[i] = (dec_out[j], state)
                cache[process[j][0]] = (dec_out[j], state)

                j += 1

        dec_out = torch.stack([d[0] for d in done])
        dec_states = self.create_batch_states(
            dec_states, [d[1] for d in done], [[0] + h.yseq for h in hyps]
        )

        if use_lm:
            lm_labels = torch.LongTensor(
                [hyp.yseq[-1] for hyp in hyps], device=self.device
            )

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None

    def select_state(
        self, states: List[Optional[torch.Tensor]], idx: int
    ) -> List[Optional[torch.Tensor]]:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. [N x (B, U, D_dec)]
            idx: State ID to extract.

        Returns:
            state_idx: Decoder hidden state for given ID. [N x (1, U, D_dec)]

        """
        if states[0] is None:
            return states

        state_idx = [states[layer][idx] for layer in range(self.dlayers)]

        return state_idx

    def create_batch_states(
        self,
        states: List[Optional[torch.Tensor]],
        new_states: List[Optional[torch.Tensor]],
        check_list: List[List[int]],
    ) -> List[Optional[torch.Tensor]]:
        """Create decoder hidden states sequences.

        Args:
            states: Decoder hidden states. [N x (B, U, D_dec)]
            new_states: Decoder hidden states. [B x [N x (1, U, D_dec)]]
            check_list: Label ID sequences.

        Returns:
            states: New decoder hidden states. [N x (B, U, D_dec)]

        """
        if new_states[0][0] is None:
            return states

        max_len = max(len(elem) for elem in check_list) - 1

        for layer in range(self.dlayers):
            states[layer] = check_batch_states(
                [s[layer] for s in new_states], max_len, self.blank_id
            )

        return states
