import random
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import initial_att
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet2.utils.get_default_kwargs import get_default_kwargs


def build_attention_list(
    eprojs: int,
    dunits: int,
    atype: str = "location",
    num_att: int = 1,
    num_encs: int = 1,
    aheads: int = 4,
    adim: int = 320,
    awin: int = 5,
    aconv_chans: int = 10,
    aconv_filts: int = 100,
    han_mode: bool = False,
    han_type=None,
    han_heads: int = 4,
    han_dim: int = 320,
    han_conv_chans: int = -1,
    han_conv_filts: int = 100,
    han_win: int = 5,
):

    att_list = torch.nn.ModuleList()
    if num_encs == 1:
        for i in range(num_att):
            att = initial_att(
                atype,
                eprojs,
                dunits,
                aheads,
                adim,
                awin,
                aconv_chans,
                aconv_filts,
            )
            att_list.append(att)
    elif num_encs > 1:  # no multi-speaker mode
        if han_mode:
            att = initial_att(
                han_type,
                eprojs,
                dunits,
                han_heads,
                han_dim,
                han_win,
                han_conv_chans,
                han_conv_filts,
                han_mode=True,
            )
            return att
        else:
            att_list = torch.nn.ModuleList()
            for idx in range(num_encs):
                att = initial_att(
                    atype[idx],
                    eprojs,
                    dunits,
                    aheads[idx],
                    adim[idx],
                    awin[idx],
                    aconv_chans[idx],
                    aconv_filts[idx],
                )
                att_list.append(att)
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
    return att_list


class RNNDecoder(AbsDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        sampling_probability: float = 0.0,
        dropout: float = 0.0,
        context_residual: bool = False,
        replace_sos: bool = False,
        num_encs: int = 1,
        att_conf: dict = get_default_kwargs(build_attention_list),
        embed_pad: Optional[int] = None,
        use_attention: bool = True,
        use_output_layer: bool = True,
    ):
        # FIXME(kamo): The parts of num_spk should be refactored more more more
        assert check_argument_types()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()
        eprojs = encoder_output_size
        self.dtype = rnn_type
        self.dunits = hidden_size
        self.dlayers = num_layers
        self.context_residual = context_residual
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.blank = embed_pad
        self.odim = vocab_size
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs
        self.use_attention = use_attention
        self.use_output_layer = use_output_layer

        # for multilingual translation
        self.replace_sos = replace_sos
        self.embed = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=embed_pad)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()

        if use_attention:
            input_size = hidden_size + eprojs
        else:
            input_size = hidden_size

        self.decoder += [
            torch.nn.LSTMCell(input_size, hidden_size)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(input_size, hidden_size)
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(hidden_size, hidden_size)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(hidden_size, hidden_size)
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf

        if context_residual:
            self.output = torch.nn.Linear(hidden_size + eprojs, vocab_size)
        else:
            self.output = torch.nn.Linear(hidden_size, vocab_size)

        self.att_list = build_attention_list(
            eprojs=eprojs, dunits=hidden_size, **att_conf
        )

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for i in range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]),
                    (z_prev[i], c_prev[i]),
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for i in range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens, strm_idx=0):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            hs_pad = [hs_pad]
            hlens = [hlens]

        # attention index for the attention module
        # in SPA (speaker parallel attention),
        # att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att_list) - 1)

        # hlens should be list of list of integer
        hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

        # get dim, length info
        olength = ys_in_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad[0])]
        z_list = [self.zero_state(hs_pad[0])]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad[0]))
            z_list.append(self.zero_state(hs_pad[0]))
        z_all = []
        if self.num_encs == 1:
            att_w = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            att_w_list = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in range(olength):
            if self.use_attention:
                if self.num_encs == 1:
                    att_c, att_w = self.att_list[att_idx](
                        hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w
                    )
                else:
                    for idx in range(self.num_encs):
                        att_c_list[idx], att_w_list[idx] = self.att_list[idx](
                            hs_pad[idx],
                            hlens[idx],
                            self.dropout_dec[0](z_list[0]),
                            att_w_list[idx],
                        )
                    hs_pad_han = torch.stack(att_c_list, dim=1)
                    hlens_han = [self.num_encs] * len(ys_in_pad)
                    att_c, att_w_list[self.num_encs] = self.att_list[self.num_encs](
                        hs_pad_han,
                        hlens_han,
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[self.num_encs],
                    )
                if i > 0 and random.random() < self.sampling_probability:
                    z_out = self.output(z_all[-1])
                    z_out = np.argmax(z_out.detach().cpu(), axis=1)
                    z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                    ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
                else:
                    # utt x (zdim + hdim)
                    ey = torch.cat((eys[:, i, :], att_c), dim=1)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
                if self.context_residual:
                    z_all.append(
                        torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                    )  # utt x (zdim + hdim)
                else:
                    z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)
            else:
                z_list, c_list = self.rnn_forward(
                    eys[:, i, :], z_list, c_list, z_list, c_list
                )
                z_all.append(self.dropout_dec[-1](z_list[-1]))

        z_all = torch.stack(z_all, dim=1)
        if self.use_output_layer:
            z_all = self.output(z_all)
        z_all.masked_fill_(
            make_pad_mask(ys_in_lens, z_all, 1),
            0,
        )
        return z_all, ys_in_lens

    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att_list) - 1)
        if self.num_encs == 1:
            a = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()

        if self.use_output_layer:
            return dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=a,
                workspace=(att_idx, z_list, c_list),
            )
        else:
            if self.use_attention:
                return ((z_list, c_list), None)
            else:
                return (z_list, c_list)

    def init_batch_states(self, x: torch.Tensor) -> torch.Tensor:
        z_list = [self.zero_state(x)]
        c_list = [self.zero_state(x)]

        for _ in range(1, self.dlayers):
            z_list.append(self.zero_state(x))
            c_list.append(self.zero_state(x))

        if self.use_attention:
            return ((z_list, c_list), None)
        else:
            return (z_list, c_list)

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att_list[att_idx](
                x[0].unsqueeze(0),
                [x[0].size(0)],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"],
            )
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att_list[idx](
                    x[idx].unsqueeze(0),
                    [x[idx].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"][idx],
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att_list[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"][self.num_encs],
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(
            ey, z_list, c_list, state["z_prev"], state["c_prev"]
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )

    def step_transducer(
        self, hyp: Hypothesis, cache: dict, init_tensor: torch.Tensor = None
    ) -> Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]]:
        """Forward one step.

        Args:
            hyp: Hypothesis
            cache: States cache
            init_tensor: initial tensor for att. (1, max_len)

        Returns:
            y: Decoder outputs (1, D_dec)
            state: Decoder states
                ([L x (1, D_dec)], [L x (1, D_dec)]) or
                (([L x (1, D_dec)], [L x (1, D_dec)]), (1, max_len))
            vy[0]: Token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp.yseq[-1], dtype=torch.long))

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, state = cache[str_yseq]
        else:
            ey = self.embed(vy)

            if self.use_attention:
                att_c, att_w = self.att_list[0](
                    init_tensor,
                    [init_tensor.size(1)],
                    hyp.dec_state[0][0][0],
                    hyp.dec_state[1],
                )

                ey = torch.cat((ey[0], att_c), dim=1)

                dec_state = self.rnn_forward(
                    ey,
                    hyp.dec_state[0][0][:],
                    hyp.dec_state[0][1][:],
                    *hyp.dec_state[0],
                )
                state = (dec_state, att_w)
            else:
                dec_state = self.rnn_forward(
                    ey[0],
                    hyp.dec_state[0][:],
                    hyp.dec_state[1][:],
                    *hyp.dec_state,
                )
                state = dec_state
            y = self.dropout_dec[-1](dec_state[0][-1])

            cache[str_yseq] = (y, state)

        return y, state, vy[0]

    def batch_step_transducer(
        self,
        hyps: List,
        batch_states: Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]],
        cache: dict,
        init_tensor: torch.Tensor = None,
    ) -> Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]]:
        """Forward batch one step.

        Args:
            hyps: Batch of hypotheses
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)]) or
                (([L x (B, D_dec)], [L x (B, D_dec)]), (B, max_len))
            cache: States cache

        Returns:
            batch_y: Decoder outputs (B, D_dec)
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)]) or
                (([L x (B, D_dec)], [L x (B, D_dec)]), (B, max_len))
            lm_tokens: Batch of token ids for LM (B, 1)

        """
        final_batch = len(hyps)

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp.yseq])

            if str_yseq in cache:
                done[i] = cache[str_yseq]
            else:
                tokens.append(hyp.yseq[-1])
                process.append((str_yseq, hyp.dec_state))

        if process:
            batch = len(process)

            tokens = to_device(self, torch.LongTensor(tokens).view(batch))

            if self.use_attention:
                state = self.init_batch_states(init_tensor)
                dec_state = self._create_batch_states(state, [p[1] for p in process])

                ey = self.embed(tokens)

                enc_hs = init_tensor.expand(batch, -1, -1)
                enc_len = [init_tensor.squeeze(0).size(0)] * batch

                att_c, att_w = self.att_list[0](
                    enc_hs, enc_len, state[0][0][0], state[1]
                )

                ey = torch.cat((ey, att_c), dim=1)

                dec_state = self.rnn_forward(
                    ey, state[0][0][:], state[0][1][:], *state[0]
                )
            else:
                dec_state = self.init_batch_states(torch.zeros((batch, self.dunits)))
                dec_state = self._create_batch_states(
                    dec_state, [p[1] for p in process]
                )

                ey = self.embed(tokens)

                dec_state = self.rnn_forward(
                    ey, dec_state[0][:], dec_state[1][:], *dec_state
                )
            y = self.dropout_dec[-1](dec_state[0][-1])

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                if self.use_attention:
                    new_state = self._select_state((dec_state, att_w), j)
                else:
                    new_state = self._select_state(dec_state, j)

                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        batch_states = self._create_batch_states(batch_states, [d[1] for d in done])
        batch_y = torch.stack([d[0] for d in done])

        lm_tokens = to_device(
            self, torch.LongTensor([h.yseq[-1] for h in hyps]).view(final_batch, 1)
        )

        return batch_y, batch_states, lm_tokens

    def _select_state(
        self,
        batch_states: Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]],
        idx: int,
    ) -> Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)]) or
                (([L x (B, D_dec)], [L x (B, D_dec)]), (B, max_len))
            idx: Index to extract state from batch of states

        Returns:
            (): Decoder states for given id
                ([L x (1, D_dec)], [L x (1, D_dec)]) or
                (([L x (1, D_dec)], [L x (1, D_dec)]), (1, max_len))
        """
        if self.use_attention:
            z_list = [batch_states[0][0][layer][idx] for layer in range(self.dlayers)]
            c_list = [batch_states[0][1][layer][idx] for layer in range(self.dlayers)]

            att_state = (
                batch_states[1][idx] if batch_states[1] is not None else batch_states[1]
            )
            return ((z_list, c_list), att_state)
        else:
            z_list = [batch_states[0][layer][idx] for layer in range(self.dlayers)]
            c_list = [batch_states[1][layer][idx] for layer in range(self.dlayers)]

            return (z_list, c_list)

    def _create_batch_states(
        self,
        batch_states: Union[Tuple[List, List], Tuple[List, List, torch.Tensor]],
        l_states: Union[List[Tuple[List, List]], List[Tuple[List, List, torch.Tensor]]],
        l_tokens: List[int] = None,
    ) -> Union[Tuple[List, List], Tuple[Tuple[List, List], torch.Tensor]]:
        """Create batch of decoder states.

        Args:
            batch_states: Batch of decoder states
               ([L x (B, D_dec)], [L x (B, D_dec)]) or
               (([L x (B, D_dec)], [L x (B, D_dec)]), (B, max_len))
            l_states: List of decoder states
                [B x ([L x (1, D_dec)], [L x (1, D_dec)])] or
                [B x (([L x (1, D_dec)], [L x (1, D_dec)]), (1, max_len))]

        Returns:
            batch_states: Batch of decoder states
                ([L x (B, D_dec)], [L x (B, D_dec)]) or
                (([L x (B, D_dec)], [L x (B, D_dec)]), (B, max_len))

        """
        if self.use_attention:
            for layer in range(self.dlayers):
                batch_states[0][0][layer] = torch.stack(
                    [s[0][0][layer] for s in l_states]
                )
                batch_states[0][1][layer] = torch.stack(
                    [s[0][1][layer] for s in l_states]
                )

            att_states = (
                torch.stack([s[1] for s in l_states])
                if l_states[0][1] is not None
                else None
            )

            return (batch_states[0], att_states)
        else:
            for layer in range(self.dlayers):
                batch_states[0][layer] = torch.stack([s[0][layer] for s in l_states])
                batch_states[1][layer] = torch.stack([s[1][layer] for s in l_states])

            return batch_states
