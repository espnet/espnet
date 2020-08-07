"""Decoder definition for transformer-transducer models."""

import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    DecoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.utils import check_state
from espnet.nets.pytorch_backend.transducer.utils import pad_batch_state
from espnet.nets.pytorch_backend.transducer.utils import pad_sequence

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class DecoderTT(TransducerDecoderInterface, torch.nn.Module):
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

    def init_state(self, init_tensor=None):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features (B, dec_dim)

        Returns:
            state (list): batch of decoder decoder states [L x None]

        """
        state = [None] * len(self.decoders)

        return state

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
            h_enc (torch.Tensor): batch of expanded hidden state (B, T, 1, enc_dim)
            h_dec (torch.Tensor): batch of expanded hidden state (B, 1, U, dec_dim)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def score(self, hyp, cache, init_tensor=None):
        """Forward one step.

        Args:
            hyp (dict): hypothese
            cache (dict): states cache

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            (tuple): decoder and attention states
                ([L x (1, max_len, dec_dim)], None)
            lm_tokens (torch.Tensor): token id for LM (1)

        """
        tgt = to_device(self, torch.tensor(hyp["yseq"]).unsqueeze(0))
        lm_tokens = tgt[:, -1]

        str_yseq = "".join([str(x) for x in hyp["yseq"]])

        if str_yseq in cache:
            y, new_state = cache[str_yseq]
        else:
            tgt_mask = to_device(self, subsequent_mask(len(hyp["yseq"])).unsqueeze(0))

            state = hyp["dec_state"]
            state = check_state(state, (tgt.size(1) - 1), self.blank)

            tgt = self.embed(tgt)

            new_state = []
            for s, decoder in zip(state, self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                new_state.append(tgt)

            if self.normalize_before:
                y = self.after_norm(tgt[:, -1])
            else:
                y = tgt[:, -1]

            cache[str_yseq] = (y, new_state)

        return y, (new_state, None), lm_tokens

    def batch_score(self, hyps, batch_states, cache, init_tensor=None):
        """Forward batch one step.

        Args:
            hyps (list of dict): batch of hypothesis
            batch_states (tuple): decoder and attention states
                ([L x (B, max_len, dec_dim)], None)
            cache (dict): states cache

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): decoder and attention states
                ([L x (B, max_len, dec_dim)], None)
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        final_batch = len(hyps)

        tokens = []
        process = []
        _y = [None for _ in range(final_batch)]
        _states = [None for _ in range(final_batch)]
        _tokens = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp["yseq"]])

            if str_yseq in cache:
                _y[i], _states[i] = cache[str_yseq]
                _tokens[i] = hyp["yseq"]
            else:
                tokens.append(hyp["yseq"])
                process.append((str_yseq, hyp["dec_state"], hyp["yseq"]))

        if process:
            batch = len(tokens)

            tokens = pad_sequence(tokens, self.blank)
            b_tokens = to_device(self, torch.LongTensor(tokens).view(batch, -1))

            tgt_mask = to_device(
                self,
                subsequent_mask(b_tokens.size(-1)).unsqueeze(0).expand(batch, -1, -1),
            )

            dec_state = self.init_state()

            dec_state = self.create_batch_states(
                (dec_state, None), [(p[1], None) for p in process], tokens,
            )

            tgt = self.embed(b_tokens)

            next_state = []
            for s, decoder in zip(dec_state[0], self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                next_state.append(tgt)

            if self.normalize_before:
                tgt = self.after_norm(tgt[:, -1])
            else:
                tgt = tgt[:, -1]

        j = 0
        for i in range(final_batch):
            if _y[i] is None:
                _y[i] = tgt[j]

                new_state = self.select_state((next_state, None), j)
                _states[i] = new_state[0]

                _tokens[i] = process[j][2]
                cache[process[j][0]] = (_y[i], new_state[0])

                j += 1

        batch_states = self.create_batch_states(
            batch_states, [(s, None) for s in _states], _tokens
        )
        batch_y = torch.stack(_y)

        lm_tokens = pad_sequence([h["yseq"] for h in hyps], self.blank)

        return batch_y, batch_states, lm_tokens

    def select_state(self, batch_states, idx):
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (tuple): batch of decoder and attention states
                ([L x (B, max_len, dec_dim)], None)
            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder and attention states
                ([L x (1, max_len, dec_dim)], None)

        """
        if batch_states[0][0] is not None:
            state_idx = [
                batch_states[0][layer][idx] for layer in range(len(self.decoders))
            ]
        else:
            state_idx = batch_states[0]

        return (state_idx, None)

    def create_batch_states(self, batch_states, l_states, l_tokens):
        """Create batch of decoder states.

        Args:
            batch_states (tuple): batch of decoder and attention states
                ([L x (B, max_len, dec_dim)], None)
            l_states (list): list of decoder and attention states
                [B x ([L x (1, max_len, dec_dim)], None)]
            l_tokens (list): list of token sequences for batch

        Returns:
            batch_states (tuple): batch of decoder and attention states
                ([L x (B, max_len, dec_dim)], None)

        """
        if batch_states[0][0] is not None:
            max_len = max([len(t) for t in l_tokens])

            for layer in range(len(self.decoders)):
                batch_states[0][layer] = pad_batch_state(
                    [s[0][layer] for s in l_states], max_len, self.blank
                )

        return batch_states
