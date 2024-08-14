import random

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, to_device
from espnet.nets.pytorch_backend.rnn.attentions import initial_att


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
    """
    Build a list of attention modules based on the specified parameters.

    This function creates and returns a list of attention modules for use in
    speech recognition models. It supports both single and multi-encoder
    configurations, as well as hierarchical attention networks (HAN).

    Args:
        eprojs (int): Number of encoder projection units.
        dunits (int): Number of decoder units.
        atype (str, optional): Attention type. Defaults to "location".
        num_att (int, optional): Number of attention modules. Defaults to 1.
        num_encs (int, optional): Number of encoders. Defaults to 1.
        aheads (int, optional): Number of attention heads. Defaults to 4.
        adim (int, optional): Attention dimension. Defaults to 320.
        awin (int, optional): Attention window size. Defaults to 5.
        aconv_chans (int, optional): Number of attention convolution channels. Defaults to 10.
        aconv_filts (int, optional): Number of attention convolution filters. Defaults to 100.
        han_mode (bool, optional): Whether to use hierarchical attention network. Defaults to False.
        han_type (str, optional): Type of hierarchical attention. Defaults to None.
        han_heads (int, optional): Number of HAN attention heads. Defaults to 4.
        han_dim (int, optional): HAN attention dimension. Defaults to 320.
        han_conv_chans (int, optional): Number of HAN convolution channels. Defaults to -1.
        han_conv_filts (int, optional): Number of HAN convolution filters. Defaults to 100.
        han_win (int, optional): HAN window size. Defaults to 5.

    Returns:
        torch.nn.ModuleList: A list of attention modules.

    Raises:
        ValueError: If the number of encoders is less than one.

    Note:
        The function behavior changes based on the number of encoders and whether
        hierarchical attention network mode is enabled.

    Examples:
        >>> att_list = build_attention_list(256, 320, num_encs=2, han_mode=True)
        >>> print(len(att_list))
        1
        >>> att_list = build_attention_list(256, 320, num_encs=2, han_mode=False)
        >>> print(len(att_list))
        2
    """
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
    """
    RNN-based decoder for sequence-to-sequence models.

    This class implements a recurrent neural network (RNN) decoder, which can be
    used in various sequence-to-sequence tasks such as speech recognition or
    machine translation. It supports both LSTM and GRU cell types, multiple
    layers, and various attention mechanisms.

    The decoder uses an embedding layer, followed by multiple RNN layers with
    dropout, and an output layer. It also incorporates attention mechanisms to
    focus on different parts of the input sequence during decoding.

    Attributes:
        embed (torch.nn.Embedding): Embedding layer for input tokens.
        decoder (torch.nn.ModuleList): List of RNN cells (LSTM or GRU).
        dropout_dec (torch.nn.ModuleList): List of dropout layers for RNN outputs.
        output (torch.nn.Linear): Output layer for vocabulary distribution.
        att_list (torch.nn.ModuleList): List of attention modules.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        rnn_type (str, optional): Type of RNN cell to use ('lstm' or 'gru'). Defaults to "lstm".
        num_layers (int, optional): Number of RNN layers. Defaults to 1.
        hidden_size (int, optional): Hidden size of the RNN. Defaults to 320.
        sampling_probability (float, optional): Probability of sampling from previous output. Defaults to 0.0.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        context_residual (bool, optional): Whether to use context residual connection. Defaults to False.
        replace_sos (bool, optional): Whether to replace <sos> token. Defaults to False.
        num_encs (int, optional): Number of encoders. Defaults to 1.
        att_conf (dict, optional): Configuration for attention modules. Defaults to get_default_kwargs(build_attention_list).

    Raises:
        ValueError: If an unsupported RNN type is specified.

    Note:
        This decoder supports both single and multi-encoder configurations, as well as
        speaker parallel attention (SPA) for multi-speaker scenarios.

    Example:
        >>> decoder = RNNDecoder(vocab_size=1000, encoder_output_size=256, num_layers=2, hidden_size=512)
        >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, encoder_output_size)
        >>> hlens = torch.full((32,), 100)  # (batch_size,)
        >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # (batch_size, max_output_length)
        >>> ys_in_lens = torch.full((32,), 20)  # (batch_size,)
        >>> decoder_output, output_lengths = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
    """

    @typechecked
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
    ):
        # FIXME(kamo): The parts of num_spk should be refactored more more more
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
        self.odim = vocab_size
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual translation
        self.replace_sos = replace_sos

        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            (
                torch.nn.LSTMCell(hidden_size + eprojs, hidden_size)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(hidden_size + eprojs, hidden_size)
            )
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                (
                    torch.nn.LSTMCell(hidden_size, hidden_size)
                    if self.dtype == "lstm"
                    else torch.nn.GRUCell(hidden_size, hidden_size)
                )
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
        """
            Initialize a zero state for the decoder.

        This method creates a tensor of zeros with the same batch size as the input
        and the decoder's hidden size. It's typically used to initialize the hidden
        state of the RNN at the start of the decoding process.

        Args:
            hs_pad (torch.Tensor): The padded hidden state tensor from the encoder.
                It is used to determine the batch size for the zero state.

        Returns:
            torch.Tensor: A tensor of zeros with shape (batch_size, hidden_size),
                where batch_size is inferred from hs_pad and hidden_size is the
                decoder's hidden size (self.dunits).

        Example:
            >>> decoder = RNNDecoder(...)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, encoder_output_size)
            >>> initial_state = decoder.zero_state(hs_pad)
            >>> print(initial_state.shape)
            torch.Size([32, 320])  # Assuming self.dunits = 320
        """
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        """
            Perform a forward pass through the RNN layers.

        This method processes the input through all layers of the RNN (LSTM or GRU),
        applying dropout between vertical connections.

        Args:
            ey (torch.Tensor): Input tensor, typically the concatenation of the
                embedded previous output and the context vector.
            z_list (list): List to store output hidden states for each layer.
            c_list (list): List to store cell states for each layer (used for LSTM only).
            z_prev (list): List of previous hidden states for each layer.
            c_prev (list): List of previous cell states for each layer (used for LSTM only).

        Returns:
            tuple: A tuple containing:
                - z_list (list): Updated list of output hidden states for each layer.
                - c_list (list): Updated list of cell states for each layer (for LSTM only).

        Note:
            - For LSTM, both hidden state (z) and cell state (c) are updated.
            - For GRU, only the hidden state (z) is updated.
            - Dropout is applied to the output of each layer except the last one.

        Example:
            >>> decoder = RNNDecoder(...)
            >>> ey = torch.randn(32, 512)  # (batch_size, embed_size + context_size)
            >>> z_list = [torch.zeros(32, 320) for _ in range(decoder.dlayers)]
            >>> c_list = [torch.zeros(32, 320) for _ in range(decoder.dlayers)]
            >>> z_prev = [torch.zeros(32, 320) for _ in range(decoder.dlayers)]
            >>> c_prev = [torch.zeros(32, 320) for _ in range(decoder.dlayers)]
            >>> z_list, c_list = decoder.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
        """
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
        """
            Perform a forward pass of the RNN decoder.

        This method processes the encoder outputs and generates decoder outputs
        for the entire sequence.

        Args:
            hs_pad (torch.Tensor or List[torch.Tensor]): Padded hidden states from encoder(s).
                For single encoder: tensor of shape (batch, time, hidden_size).
                For multiple encoders: list of such tensors.
            hlens (torch.Tensor or List[torch.Tensor]): Lengths of encoder hidden states.
                For single encoder: tensor of shape (batch,).
                For multiple encoders: list of such tensors.
            ys_in_pad (torch.Tensor): Padded input label sequences.
                Shape: (batch, sequence_length).
            ys_in_lens (torch.Tensor): Lengths of input label sequences.
                Shape: (batch,).
            strm_idx (int, optional): Stream index for multi-speaker attention.
                Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - z_all (torch.Tensor): Output sequence scores.
                    Shape: (batch, sequence_length, vocab_size).
                - ys_in_lens (torch.Tensor): Lengths of input sequences.

        Note:
            - This method supports both single and multiple encoder scenarios.
            - It implements teacher forcing with optional scheduled sampling.
            - For multiple encoders, hierarchical attention is used.

        Example:
            >>> decoder = RNNDecoder(...)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch, time, hidden_size)
            >>> hlens = torch.full((32,), 100)
            >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # (batch, sequence_length)
            >>> ys_in_lens = torch.full((32,), 20)
            >>> z_all, out_lens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
            >>> print(z_all.shape)
            torch.Size([32, 20, 1000])  # (batch, sequence_length, vocab_size)
        """
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

        z_all = torch.stack(z_all, dim=1)
        z_all = self.output(z_all)
        z_all.masked_fill_(
            make_pad_mask(ys_in_lens, z_all, 1),
            0,
        )
        return z_all, ys_in_lens

    def init_state(self, x):
        """
            Initialize the decoder state for inference.

        This method sets up the initial state of the decoder, including hidden states,
        cell states (for LSTM), and attention weights. It's typically used at the start
        of the decoding process during inference.

        Args:
            x (torch.Tensor or List[torch.Tensor]): The encoder output(s).
                For single encoder: tensor of shape (batch, time, hidden_size).
                For multiple encoders: list of such tensors.

        Returns:
            dict: A dictionary containing the initial decoder state with keys:
                - c_prev (list): Initial cell states for each layer (for LSTM).
                - z_prev (list): Initial hidden states for each layer.
                - a_prev (None or list): Initial attention weights.
                - workspace (tuple): Additional workspace information including:
                    - att_idx (int): Index of the current attention module.
                    - z_list (list): List of initial hidden states.
                    - c_list (list): List of initial cell states (for LSTM).

        Note:
            - The method handles both single and multiple encoder scenarios.
            - For multiple encoders, it initializes states for all encoders and the
              hierarchical attention network (HAN).
            - The attention modules are reset to clear any pre-computed values.

        Example:
            >>> decoder = RNNDecoder(...)
            >>> x = torch.randn(1, 100, 256)  # (batch, time, hidden_size)
            >>> initial_state = decoder.init_state(x)
            >>> print(initial_state.keys())
            dict_keys(['c_prev', 'z_prev', 'a_prev', 'workspace'])
        """
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
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        """
            Calculate the log probability score for the next token.

        This method computes the score for the next token in the sequence given the
        current state and encoder outputs. It's typically used in beam search decoding.

        Args:
            yseq (torch.Tensor): Current output sequence.
                Shape: (sequence_length,).
            state (dict): Current decoder state containing:
                - c_prev (list): Previous cell states for each layer (for LSTM).
                - z_prev (list): Previous hidden states for each layer.
                - a_prev (None or list): Previous attention weights.
                - workspace (tuple): Additional workspace information.
            x (torch.Tensor or List[torch.Tensor]): The encoder output(s).
                For single encoder: tensor of shape (batch, time, hidden_size).
                For multiple encoders: list of such tensors.

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Log probability scores for the next token.
                    Shape: (vocab_size,).
                - new_state (dict): Updated decoder state.

        Note:
            - This method supports both single and multiple encoder scenarios.
            - For multiple encoders, it uses hierarchical attention.
            - The context vector is concatenated with the embedded input for scoring.

        Example:
            >>> decoder = RNNDecoder(...)
            >>> yseq = torch.tensor([1, 2, 3])  # Current sequence
            >>> x = torch.randn(1, 100, 256)  # (batch, time, hidden_size)
            >>> state = decoder.init_state(x)
            >>> logp, new_state = decoder.score(yseq, state, x)
            >>> print(logp.shape)
            torch.Size([1000])  # Assuming vocab_size = 1000
        """
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
