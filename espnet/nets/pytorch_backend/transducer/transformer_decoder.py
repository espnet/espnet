"""Decoder definition for transformer-transducer models."""

import six
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import DecoderLayer

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class Decoder(torch.nn.Module):
    """Decoder module for transformer-transducer models.

    Args:
        odim (int): dimension of outputs
        jdim (int): dimension of joint-space
        attention_dim (int): dimension of attention
        attention_heads (int): number of heads in multi-head attention
        linear_units (int): number of units in position-wise feed forward
        num_blocks (int): number of decoder blocks
        dropout_rate (float): dropout rate for decoder
        positional_dropout_rate (float): dropout rate for positional encoding
        attention_dropout_rate (float): dropout rate for attention
        input_layer (str or torch.nn.Module): input layer type
        padding_idx (int): padding value for embedding
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        blank (int): blank symbol ID

    """

    def __init__(self, odim, jdim,
                 attention_dim=512,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 input_layer="embed",
                 pos_enc_class=PositionalEncoding,
                 blank=0):
        """Construct a Decoder object for transformer-transducer models."""
        torch.nn.Module.__init__(self)

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate))
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate))
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")

        self.decoders = repeat(
            num_blocks,
            lambda: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate))

        self.after_norm = LayerNorm(attention_dim)

        self.lin_enc = torch.nn.Linear(attention_dim, jdim)
        self.lin_dec = torch.nn.Linear(attention_dim, jdim, bias=False)
        self.lin_out = torch.nn.Linear(jdim, odim)

        self.attention_dim = attention_dim
        self.odim = odim

        self.blank = blank

    def forward(self, tgt, tgt_mask, memory):
        """Forward transformer-transducer decoder.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels) in the other cases
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
            h_enc (torch.Tensor): batch of expanded hidden state (batch, maxlen_in, 1, Henc)
            h_dec (torch.Tensor): batch of expanded hidden state (batch, 1, maxlen_out, Hdec)

        Returns:
            z (torch.Tensor): output (batch, maxlen_in, maxlen_out, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward_one_step(self, tgt, tgt_mask, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels) in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, Tmax)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)

        """
        tgt = self.embed(tgt)

        if cache is None:
            cache = self.init_state()
        new_cache = []

        for c, decoder in zip(cache, self.decoders):
            tgt, tgt_mask = decoder(tgt, tgt_mask, c)
            new_cache.append(tgt)

        tgt = self.after_norm(tgt[:, -1])

        return tgt, new_cache

    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.decoders))]

    def recognize(self, h, recog_args):
        """Greedy search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        hyp = {'score': 0.0, 'yseq': [self.blank]}

        ys = to_device(self, torch.tensor(hyp['yseq'], dtype=torch.long)).unsqueeze(0)
        ys_mask = to_device(self, subsequent_mask(1).unsqueeze(0))
        y, c = self.forward_one_step(ys, ys_mask, None)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp['yseq'].append(int(pred))
                hyp['score'] += float(logp)

                ys = to_device(self, torch.tensor(hyp['yseq']).unsqueeze(0))
                ys_mask = to_device(self, subsequent_mask(len(hyp['yseq'])).unsqueeze(0))

                y, c = self.forward_one_step(ys, ys_mask, c)

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
        """Beam search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)
        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        if rnnlm:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'cache': None, 'lm_state': None}]
        else:
            kept_hyps = [{'score': 0.0, 'yseq': [self.blank], 'cache': None}]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x['score'])
                hyps.remove(new_hyp)

                ys = to_device(self, torch.tensor(new_hyp['yseq']).unsqueeze(0))
                ys_mask = to_device(self, subsequent_mask(len(new_hyp['yseq'])).unsqueeze(0))
                y, c = self.forward_one_step(ys, ys_mask, new_hyp['cache'])

                ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(new_hyp['lm_state'], ys[:, -1])

                for k in six.moves.range(self.odim):
                    beam_hyp = {'score': new_hyp['score'] + float(ytu[k]),
                                'yseq': new_hyp['yseq'][:],
                                'cache': new_hyp['cache']}

                    if rnnlm:
                        beam_hyp['lm_state'] = new_hyp['lm_state']

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp['yseq'].append(int(k))
                        beam_hyp['cache'] = c

                        if rnnlm:
                            beam_hyp['lm_state'] = rnnlm_state
                            beam_hyp['score'] += recog_args.lm_weight * rnnlm_scores[0][k]

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'] / len(x['yseq']), reverse=True)[:nbest]
        else:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x['score'], reverse=True)[:nbest]

        return nbest_hyps
