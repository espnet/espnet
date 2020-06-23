"""Decoder definition for transformer-transducer models."""

import numpy as np
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.utils import is_prefix

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class Decoder(torch.nn.Module):
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
                batch of expanded hidden state (batch, maxlen_in, 1, Henc)
            h_dec (torch.Tensor):
                batch of expanded hidden state (batch, 1, maxlen_out, Hdec)

        Returns:
            z (torch.Tensor): output (batch, maxlen_in, maxlen_out, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward_one_step(self, tgt, tgt_mask, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels)
                                in the other cases
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
        hyp = {"score": 0.0, "yseq": [self.blank]}

        ys = to_device(self, torch.tensor(hyp["yseq"], dtype=torch.long)).unsqueeze(0)
        ys_mask = to_device(self, subsequent_mask(1).unsqueeze(0))
        y, c = self.forward_one_step(ys, ys_mask, None)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                ys = to_device(self, torch.tensor(hyp["yseq"]).unsqueeze(0))
                ys_mask = to_device(
                    self, subsequent_mask(len(hyp["yseq"])).unsqueeze(0)
                )

                y, c = self.forward_one_step(ys, ys_mask, c)

        return [hyp]

    def recognize_beam_default(self, h, recog_args, rnnlm=None):
        """Beam search implementation.

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

        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "cache": None,
                "y": [],
                "lm_state": None,
            }
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x["yseq"]), reverse=True)
            kept_hyps = []

            for j in range(len(hyps) - 1):
                for i in range((j + 1), len(hyps)):
                    if is_prefix(hyps[j]["yseq"], hyps[i]["yseq"]):
                        next_id = len(hyps[i]["yseq"])

                        ys = to_device(self, torch.tensor(hyps[i]["yseq"]).unsqueeze(0))

                        ys_mask = to_device(
                            self, subsequent_mask(len(hyps[i]["yseq"])).unsqueeze(0)
                        )

                        y, _ = self.forward_one_step(ys, ys_mask, hyps[i]["cache"])

                        ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)

                        curr_score = float(hyps[i]["score"]) + float(
                            ytu[hyps[j]["yseq"][next_id]]
                        )

                        for k in range(next_id, (len(hyps[j]["yseq"]) - 1)):
                            ytu = torch.log_softmax(
                                self.joint(hi, hyps[j]["y"][k]), dim=0
                            )

                            curr_score += float(ytu[hyps[j]["yseq"][k + 1]])

                        hyps[j]["score"] = np.logaddexp(
                            float(hyps[j]["score"]), curr_score
                        )

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                ys = to_device(self, torch.tensor(new_hyp["yseq"]).unsqueeze(0))
                ys_mask = to_device(
                    self, subsequent_mask(len(new_hyp["yseq"])).unsqueeze(0)
                )

                y, c = self.forward_one_step(ys, ys_mask, new_hyp["cache"])

                ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], ys[:, -1]
                    )

                for k in range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "cache": new_hyp["cache"],
                        "y": new_hyp["y"],
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["yseq"].append(int(k))
                        beam_hyp["cache"] = c

                        beam_hyp["y"].append(y[0])

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )

                        hyps.append(beam_hyp)

                hyps_max = float(max(hyps, key=lambda x: x["score"])["score"])
                kept_most_prob = len(
                    sorted(kept_hyps, key=lambda x: float(x["score"]) > hyps_max)
                )
                if kept_most_prob >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps

    def recognize_beam_nsc(self, h, recog_args, rnnlm=None):
        """Beam search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        raise NotImplementedError
