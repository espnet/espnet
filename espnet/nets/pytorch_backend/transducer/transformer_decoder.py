"""Decoder definition for transformer-transducer models."""

import numpy as np
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    DecoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.utils import get_beam_lm_states
from espnet.nets.pytorch_backend.transducer.utils import get_idx_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import substract

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
            cache = [None] * len(self.decoders)
        new_cache = []

        for c, decoder in zip(cache, self.decoders):
            tgt, tgt_mask = decoder(tgt, tgt_mask, cache=c)
            new_cache.append(tgt)

        if self.normalize_before:
            tgt = self.after_norm(tgt[:, -1])
        else:
            tgt = tgt[:, -1]

        return tgt, new_cache

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
            {"score": 0.0, "yseq": [self.blank], "cache": None, "lm_state": None}
        ]

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

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
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["yseq"].append(int(k))
                        beam_hyp["cache"] = c

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
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """

        def pad_sequence(seqlist):
            maxlen = max(len(x) for x in seqlist)

            final = [([self.blank] * (maxlen - len(x))) + x for x in seqlist]

            return final

        def pad_cache(cache, pred_length):
            batch = len(cache)
            maxlen = max([c.size(0) for c in cache])
            ddim = cache[0].size(1)

            final_dims = (batch, maxlen, ddim)
            final = cache[0].data.new(*final_dims).fill_(0)

            for i, c in enumerate(cache):
                final[i, (maxlen - c.size(0)) : maxlen, :] = c

            trim_val = final[0].size(0) - (pred_length - 1)

            return final[:, trim_val:, :]

        beam = recog_args.beam_size
        w_range = min(beam, self.odim)

        nstep = recog_args.nstep
        prefix_alpha = recog_args.prefix_alpha

        nbest = recog_args.nbest

        w_tokens = [self.blank for _ in range(w_range)]
        w_tokens = torch.LongTensor(w_tokens).view(w_range, -1)

        w_tokens_mask = (
            subsequent_mask(w_tokens.size(-1)).unsqueeze(0).expand(w_range, -1, -1)
        )

        w_y, w_c = self.forward_one_step(w_tokens, w_tokens_mask, None)

        cache = []
        for layer in range(len(self.decoders)):
            cache.append(w_c[layer][0])

        if rnnlm:
            w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(
                None, w_tokens[:, -1], w_range
            )

            if hasattr(rnnlm.predictor, "wordlm"):
                lm_type = "wordlm"
                lm_layers = len(w_rnnlm_states[0])
            else:
                lm_type = "lm"
                lm_layers = len(w_rnnlm_states["c"])

            rnnlm_states = get_idx_lm_state(w_rnnlm_states, 0, lm_type, lm_layers)
            rnnlm_scores = w_rnnlm_scores[0]
        else:
            rnnlm_states = None
            rnnlm_scores = None

        kept_hyps = [
            {
                "score": 0.0,
                "yseq": [self.blank],
                "cache": cache,
                "y": [w_y[0]],
                "lm_states": rnnlm_states,
                "lm_scores": rnnlm_scores,
            }
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x["yseq"]), reverse=True)
            kept_hyps = []

            for j in range(len(hyps) - 1):
                for i in range((j + 1), len(hyps)):
                    if (
                        is_prefix(hyps[j]["yseq"], hyps[i]["yseq"])
                        and (len(hyps[j]["yseq"]) - len(hyps[i]["yseq"]))
                        <= prefix_alpha
                    ):
                        next_id = len(hyps[i]["yseq"])

                        ytu = torch.log_softmax(self.joint(hi, hyps[i]["y"][-1]), dim=0)

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

            S = []
            V = []
            for n in range(nstep):
                h_enc = hi.unsqueeze(0).expand(w_range, -1)

                w_y = torch.stack([hyp["y"][-1] for hyp in hyps])

                if len(hyps) == 1:
                    w_y = w_y.expand(w_range, -1)

                w_logprobs = torch.log_softmax(self.joint(h_enc, w_y), dim=-1).view(-1)

                if rnnlm:
                    w_rnnlm_scores = torch.stack([hyp["lm_scores"] for hyp in hyps])

                    if len(hyps) == 1:
                        w_rnnlm_scores = w_rnnlm_scores.expand(w_range, -1)

                    w_rnnlm_scores = w_rnnlm_scores.contiguous().view(-1)

                for i, hyp in enumerate(hyps):
                    pos_k = i * self.odim
                    k_i = w_logprobs.narrow(0, pos_k, self.odim)

                    if rnnlm:
                        lm_k_i = w_rnnlm_scores.narrow(0, pos_k, self.odim)

                    for k in range(self.odim):
                        curr_score = float(k_i[k])

                        w_hyp = {
                            "yseq": hyp["yseq"][:],
                            "score": hyp["score"] + curr_score,
                            "cache": hyp["cache"],
                            "y": hyp["y"][:],
                            "lm_states": hyp["lm_states"],
                            "lm_scores": hyp["lm_scores"],
                        }

                        if k == self.blank:
                            S.append(w_hyp)
                        else:
                            w_hyp["yseq"].append(int(k))

                            if rnnlm:
                                w_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                            V.append(w_hyp)

                V = sorted(V, key=lambda x: x["score"], reverse=True)
                V = substract(V, hyps)[:w_range]

                w_tokens = pad_sequence([v["yseq"] for v in V])
                w_tokens = torch.LongTensor(w_tokens).view(w_range, -1)

                w_tokens_mask = (
                    subsequent_mask(w_tokens.size(-1))
                    .unsqueeze(0)
                    .expand(w_range, -1, -1)
                )

                for layer in range(len(self.decoders)):
                    w_c[layer] = pad_cache(
                        [v["cache"][layer] for v in V], w_tokens.size(1)
                    )

                w_y, w_c = self.forward_one_step(w_tokens, w_tokens_mask, w_c)

                if rnnlm:
                    w_rnnlm_states = get_beam_lm_states(
                        [v["lm_states"] for v in V], lm_type, lm_layers
                    )

                    w_rnnlm_states, w_rnnlm_scores = rnnlm.buff_predict(
                        w_rnnlm_states, w_tokens[:, -1], w_range
                    )

                if n < (nstep - 1):
                    for i, v in enumerate(V):
                        v["cache"] = [
                            w_c[layer][i] for layer in range(len(self.decoders))
                        ]
                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

                    hyps = V[:]
                else:
                    w_logprobs = torch.log_softmax(self.joint(h_enc, w_y), dim=-1).view(
                        -1
                    )
                    blank_score = w_logprobs[0 :: self.odim]

                    for i, v in enumerate(V):
                        if nstep != 1:
                            v["score"] += float(blank_score[i])

                        v["cache"] = [
                            w_c[layer][i] for layer in range(len(self.decoders))
                        ]
                        v["y"].append(w_y[i])

                        if rnnlm:
                            v["lm_states"] = get_idx_lm_state(
                                w_rnnlm_states, i, lm_type, lm_layers
                            )
                            v["lm_scores"] = w_rnnlm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x["score"], reverse=True)[
                :w_range
            ]

        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
        )[:nbest]

        return nbest_hyps
