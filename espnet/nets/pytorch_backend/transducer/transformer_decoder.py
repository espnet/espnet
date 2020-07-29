"""Decoder definition for transformer-transducer models."""

import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    DecoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.utils import pad_sequence
from espnet.nets.pytorch_backend.transducer.utils import pad_state

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class DecoderTT(torch.nn.Module):
    """Decoder module for transformer-transducer models.

    Args:
        odim (int): dimension of outputs
        edim (int): dimension of encoder outputs
        jdim (int): dimension of joint-space
        dec_arch (List[dict]): list of layer definitions
        input_layer (str): input layer type
        repeat_block (int): if N > 1, repeat block N times
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        positionwise_layer_type (str): linear of conv1d
        positionwise_conv_kernel_size (int) : kernel size of positionwise conv1d layer
        dropout_rate_embed (float): dropout rate for embedding layer
        dropout_rate (float): dropout rate
        attention_dropout_rate (float): dropout rate in attention
        positional_dropout_rate (float): dropout rate after adding positional encoding
        use_glu (bool): wheter to use GLU in joint network
        normalize_before (bool): whether to use layer_norm before the first block
        blank (int): blank symbol ID

    """

    def __init__(
        self,
        odim,
        edim,
        jdim,
        dec_arch,
        input_layer="embed",
        repeat_block=0,
        pos_enc_class=PositionalEncoding,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        dropout_rate_embed=0.0,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        normalize_before=True,
        blank=0,
    ):
        """Construct a Decoder object for transformer-transducer models."""
        torch.nn.Module.__init__(self)

        self.embed, self.decoders, ddim = build_blocks(
            odim,
            input_layer,
            dec_arch,
            DecoderLayer,
            repeat_block=repeat_block,
            pos_enc_class=pos_enc_class,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            dropout_rate_embed=dropout_rate_embed,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            att_dropout_rate=attention_dropout_rate,
            padding_idx=blank,
        )

        self.normalize_before = normalize_before

        if self.normalize_before:
            self.after_norm = LayerNorm(ddim)

        self.lin_enc = torch.nn.Linear(edim, jdim)
        self.lin_dec = torch.nn.Linear(ddim, jdim, bias=False)

        self.lin_out = torch.nn.Linear(jdim, odim)

        self.dunits = ddim
        self.odim = odim

        self.blank = blank

    def forward(self, tgt, tgt_mask, memory):
        """Forward transformer-transducer decoder.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor
                                (batch, maxlen_out, #mels) in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, maxlen_out)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory (torch.Tensor): encoded memory, float32  (batch, maxlen_in, feat)

        Return:
            z (torch.Tensor): joint output (batch, maxlen_in, maxlen_out, odim)
            tgt_mask (torch.Tensor): score mask before softmax (batch, maxlen_out)

        """
        tgt = self.embed(tgt)

        tgt, tgt_mask = self.decoders(tgt, tgt_mask)
        tgt = self.after_norm(tgt)

        h_enc = memory.unsqueeze(2)
        h_dec = tgt.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z, tgt_mask

    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor):
                batch of expanded hidden state (batch, maxlen_in, 1, enc_dim)
            h_dec (torch.Tensor):
                batch of expanded hidden state (batch, 1, maxlen_out, dec_dim)

        Returns:
            z (torch.Tensor): output (batch, maxlen_in, maxlen_out, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def zero_state(self, init_tensor=None):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): input features (B, dec_dim)

        Returns:
            state (torch.Tensor): list of L decoder states (None)

        """
        state = [None] * len(self.decoders)

        return state

    def forward_one_step(self, hyp, init_tensor=None):
        """Forward one step.

        Args:
            hyp (dict): hypothese

        Returns:
            tgt (torch.Tensor): decoder outputs (1, dec_dim)
            new_state (list): list of L decoder states (1, max_length, dec_dim)
            lm_tokens (torch.Tensor): input token id for LM (1)

        """
        tgt = to_device(self, torch.tensor(hyp["yseq"]).unsqueeze(0))
        lm_tokens = tgt[:, -1]
        tgt_mask = to_device(self, subsequent_mask(len(hyp["yseq"])).unsqueeze(0))

        state = hyp["dec_state"]

        tgt = self.embed(tgt)

        new_state = []
        for s, decoder in zip(state, self.decoders):
            tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
            new_state.append(tgt)

        if self.normalize_before:
            tgt = self.after_norm(tgt[:, -1])
        else:
            tgt = tgt[:, -1]

        return tgt, new_state, None, lm_tokens

    def forward_batch_one_step(self, hyps, state, att_w=None, att_params=None):
        """Forward batch one step.

        Args:
            hyps (list of dict): batch of hypothesis
            state (list): list of L decoder states (B, max_len, dec_dim)

        Returns:
            tgt (torch.Tensor): decoder output (B, dec_dim)
            new_state (list): list of L decoder states (B, max_len, dec_dim)
            lm_tokens (torch.Tensor): input token ids for LM (B)

        """
        tokens = [h["yseq"] for h in hyps]
        tokens = pad_sequence(tokens, self.blank)
        batch = len(tokens)

        b_tokens = to_device(self, torch.LongTensor(tokens).view(batch, -1))
        tgt_mask = to_device(
            self, subsequent_mask(b_tokens.size(-1)).unsqueeze(0).expand(batch, -1, -1)
        )

        tgt = self.embed(b_tokens)

        new_state = []
        for s, decoder in zip(state, self.decoders):
            tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
            new_state.append(tgt)

        if self.normalize_before:
            tgt = self.after_norm(tgt[:, -1])
        else:
            tgt = tgt[:, -1]

        return tgt, new_state, None, b_tokens[:, -1]

    def get_idx_dec_state(self, state, idx, att_state=None):
        """Get decoder state from batch for given id.

        Args:
            state (list): list of L decoder states (B, max_len, dec_dim)
            idx (int): index to extract state from beam state

        Returns:
            state (list): list of L decoder states (max_len, dec_dim)
            idx_state (dict): dict of lm state for given id

        """
        state_idx = [state[layer][idx] for layer in range(len(self.decoders))]

        return state_idx, None

    def get_batch_dec_states(self, state, hyps):
        """Create batch of decoder states.

        Args:
            state (list): input list of decoder states (B, max_len, dec_dim)
            hyps (list): batch of hypothesis

        Returns:
            state (list): output list of L decoder states (B, max_len, dec_dim)

        """
        max_length = max([len(h["yseq"]) for h in hyps])

        for layer in range(len(self.decoders)):
            state[layer] = pad_state(
                [h["dec_state"][layer] for h in hyps], max_length, self.blank
            )

        return state, None
