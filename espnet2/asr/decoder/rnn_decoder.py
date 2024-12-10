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
    Builds a list of attention mechanisms for a neural network decoder.

    This function creates a list of attention modules based on the specified
    parameters, including types and configurations for both single and
    multi-encoder setups. It initializes the attention mechanisms according to
    the input projection size and decoder units.

    Args:
        eprojs (int): The number of input projection dimensions.
        dunits (int): The number of decoder units.
        atype (str, optional): Type of attention mechanism to use. Defaults to
            "location".
        num_att (int, optional): Number of attention mechanisms to create.
            Defaults to 1.
        num_encs (int, optional): Number of encoders. Defaults to 1.
        aheads (int, optional): Number of attention heads. Defaults to 4.
        adim (int, optional): Dimension of the attention layer. Defaults to 320.
        awin (int, optional): Size of the attention window. Defaults to 5.
        aconv_chans (int, optional): Number of channels in the attention
            convolution. Defaults to 10.
        aconv_filts (int, optional): Size of filters in the attention
            convolution. Defaults to 100.
        han_mode (bool, optional): Flag to indicate if hierarchical attention
            mode is enabled. Defaults to False.
        han_type (optional): Type of hierarchical attention if `han_mode` is
            True. Defaults to None.
        han_heads (int, optional): Number of heads in hierarchical attention.
            Defaults to 4.
        han_dim (int, optional): Dimension of the hierarchical attention layer.
            Defaults to 320.
        han_conv_chans (int, optional): Number of channels in the hierarchical
            attention convolution. Defaults to -1.
        han_conv_filts (int, optional): Size of filters in the hierarchical
            attention convolution. Defaults to 100.
        han_win (int, optional): Size of the hierarchical attention window.
            Defaults to 5.

    Returns:
        torch.nn.ModuleList: A list of initialized attention modules.

    Raises:
        ValueError: If `num_encs` is less than or equal to 0.

    Examples:
        >>> att_list = build_attention_list(256, 128)
        >>> len(att_list)
        1
        >>> att_list = build_attention_list(256, 128, num_att=2)
        >>> len(att_list)
        2
        >>> att_list = build_attention_list(256, 128, num_encs=2)
        >>> len(att_list)
        2

    Note:
        This function is typically used in the context of sequence-to-sequence
        models where attention mechanisms are critical for aligning inputs and
        outputs.
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
    RNNDecoder is a recurrent neural network (RNN) based decoder for automatic 
    speech recognition (ASR). It is designed to convert encoded representations 
    from an encoder into a sequence of output tokens using attention mechanisms 
    and recurrent layers.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        encoder_output_size (int): The size of the encoder output.
        rnn_type (str): The type of RNN to use, either 'lstm' or 'gru'.
        num_layers (int): The number of recurrent layers in the decoder.
        hidden_size (int): The size of the hidden layers in the RNN.
        sampling_probability (float): The probability of using sampling during 
            decoding.
        dropout (float): The dropout rate for regularization.
        context_residual (bool): Whether to use context residual connections.
        replace_sos (bool): Whether to replace the start of sequence token.
        num_encs (int): The number of encoders to support.
        att_list (ModuleList): A list of attention modules for decoding.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Size of the encoder output.
        rnn_type (str, optional): Type of RNN ('lstm' or 'gru'). Defaults to 'lstm'.
        num_layers (int, optional): Number of layers in the RNN. Defaults to 1.
        hidden_size (int, optional): Size of hidden units. Defaults to 320.
        sampling_probability (float, optional): Probability for sampling. 
            Defaults to 0.0.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        context_residual (bool, optional): Use context residual connections. 
            Defaults to False.
        replace_sos (bool, optional): Replace the start of sequence token. 
            Defaults to False.
        num_encs (int, optional): Number of encoders. Defaults to 1.
        att_conf (dict, optional): Configuration for attention. Defaults to 
            built-in configuration.

    Returns:
        None

    Raises:
        ValueError: If rnn_type is not 'lstm' or 'gru'.

    Examples:
        # Initialize the decoder
        decoder = RNNDecoder(
            vocab_size=5000,
            encoder_output_size=256,
            rnn_type='lstm',
            num_layers=2,
            hidden_size=512,
            sampling_probability=0.1,
            dropout=0.2
        )

        # Forward pass through the decoder
        output, lengths = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        This class supports multiple encoders and can be used for multilingual 
        translation tasks.

    Todo:
        - Refactor handling of multiple speakers in future updates.
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
        Initialize the hidden state of the RNN decoder.

        This method creates a zero-filled tensor to be used as the initial hidden
        state for the RNN cells in the decoder. The size of the tensor matches
        the batch size of the input tensor, with the second dimension being equal
        to the number of hidden units in the RNN.

        Args:
            hs_pad (torch.Tensor): A tensor of shape (batch_size, hidden_size) 
                from which the batch size is inferred.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) filled with 
            zeros, which represents the initial hidden state.

        Examples:
            >>> decoder = RNNDecoder(vocab_size=100, encoder_output_size=256)
            >>> hs_pad = torch.randn(32, 256)  # Example input
            >>> initial_state = decoder.zero_state(hs_pad)
            >>> print(initial_state.shape)
            torch.Size([32, 320])  # Assuming hidden_size is 320

        Note:
            This method is primarily used to initialize the hidden state for 
            the first step of the decoding process in RNNs.
        """
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        """
        Performs a forward pass through the RNN layers.

        This method processes the input embedding `ey` and updates the hidden
        state and cell state of the RNN layers based on the specified RNN type
        (LSTM or GRU). It handles multiple layers of RNNs, applying dropout to
        the outputs of the previous layer.

        Args:
            ey (torch.Tensor): The input tensor of shape (batch_size, input_size),
                where `input_size` is the combined size of the embedding and
                attention context.
            z_list (list): A list of tensors containing the hidden states of each
                RNN layer, each of shape (batch_size, hidden_size).
            c_list (list): A list of tensors containing the cell states of each
                LSTM layer, each of shape (batch_size, hidden_size). This argument
                is ignored for GRU.
            z_prev (list): A list of tensors representing the previous hidden
                states for each RNN layer.
            c_prev (list): A list of tensors representing the previous cell states
                for each LSTM layer, this argument is ignored for GRU.

        Returns:
            tuple: A tuple containing:
                - z_list (list): The updated hidden states for each RNN layer.
                - c_list (list): The updated cell states for each LSTM layer.

        Examples:
            >>> rnn_decoder = RNNDecoder(vocab_size=100, encoder_output_size=256)
            >>> ey = torch.randn(32, 256)  # Example input tensor
            >>> z_list = [torch.zeros(32, 320) for _ in range(2)]  # Hidden states
            >>> c_list = [torch.zeros(32, 320) for _ in range(2)]  # Cell states
            >>> z_prev = [torch.zeros(32, 320) for _ in range(2)]  # Previous hidden
            >>> c_prev = [torch.zeros(32, 320) for _ in range(2)]  # Previous cell
            >>> z_list, c_list = rnn_decoder.rnn_forward(ey, z_list, c_list, z_prev, c_prev)

        Note:
            - The method handles both LSTM and GRU architectures seamlessly.
            - Dropout is applied to the outputs of the previous layer, controlled
            by the `dropout` parameter during initialization.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected
            shape.
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
        Performs the forward pass of the RNN decoder.

        This method takes the padded hidden states from the encoder and the
        previous output sequences, processes them through the RNN layers, and
        computes the logits for the next output sequence. It supports both
        single and multiple encoder modes.

        Args:
            hs_pad (torch.Tensor or List[torch.Tensor]): The padded hidden states
                from the encoder. If in multiple encoder mode, this should be a
                list of tensors, one for each encoder.
            hlens (torch.Tensor or List[torch.Tensor]): The lengths of the hidden
                states corresponding to `hs_pad`. Should match the format of
                `hs_pad`.
            ys_in_pad (torch.Tensor): The input sequences (previous outputs),
                padded to the maximum length.
            ys_in_lens (torch.Tensor): The lengths of the input sequences.
            strm_idx (int, optional): The index of the stream (encoder) to use
                for attention. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Logits for the output sequence of shape (batch_size, max_length,
                vocab_size).
                - The lengths of the input sequences after processing.

        Raises:
            ValueError: If the number of encoders is less than one.

        Examples:
            >>> decoder = RNNDecoder(vocab_size=5000, encoder_output_size=256)
            >>> hs_pad = torch.randn(10, 20, 256)  # Batch of 10, max length 20
            >>> hlens = torch.randint(1, 21, (10,))  # Random lengths
            >>> ys_in_pad = torch.randint(0, 5000, (10, 15))  # Previous outputs
            >>> ys_in_lens = torch.randint(1, 16, (10,))  # Random lengths
            >>> logits, new_lengths = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)
        
        Note:
            This function handles both single and multi-encoder scenarios. In the
            case of multiple encoders, the attention mechanism will select the
            appropriate encoder based on the provided `strm_idx`.

        Todo:
            - Optimize the performance of the forward pass for large sequences.
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
        Initializes the hidden and cell states for the RNN decoder.

        This method prepares the initial states required for the recurrent neural
        network (RNN) decoder. It supports multiple encoder configurations and 
        ensures that the initial states are correctly shaped for processing.

        Attributes:
            c_prev (list): A list containing the initial cell states for each RNN layer.
            z_prev (list): A list containing the initial hidden states for each RNN layer.
            a_prev (list or None): A list containing the initial attention states for each 
                encoder, or None if there is only one encoder.
            workspace (tuple): A tuple containing the attention index and the lists of 
                hidden and cell states.

        Args:
            x (torch.Tensor): The input tensor from which to derive the initial states. 
                Its shape should be (batch_size, encoder_output_size).

        Returns:
            dict: A dictionary containing the initialized states and workspace.

        Examples:
            >>> decoder = RNNDecoder(vocab_size=1000, encoder_output_size=512)
            >>> x = torch.randn(32, 512)  # Example input for a batch of size 32
            >>> states = decoder.init_state(x)
            >>> print(states['c_prev'])  # Should print the initialized cell states
            >>> print(states['z_prev'])  # Should print the initialized hidden states

        Note:
            This method is primarily used during the decoding process, where it sets
            up the initial states before generating outputs from the decoder.

        Todo:
            - Support stream index for `asr_mix` configurations in the future.
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
        Calculate the log probabilities of the next token in a sequence.

        This method computes the log probabilities of the next token given
        the previous tokens, the current state, and the encoder outputs. It 
        uses the recurrent neural network (RNN) to process the input and 
        apply attention mechanisms if multiple encoders are utilized.

        Args:
            yseq (torch.Tensor): A tensor containing the sequence of tokens 
                (shape: (T,)) where T is the sequence length.
            state (dict): A dictionary containing the current state, which 
                includes the previous hidden states and the attention context.
            x (torch.Tensor): A tensor containing the encoder outputs, where 
                the shape is (B, E), B is the batch size and E is the 
                encoder output size.

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Log probabilities of the next token 
                (shape: (vocab_size,)).
                - state (dict): A dictionary containing the updated state 
                with previous hidden states and attention weights.

        Examples:
            >>> decoder = RNNDecoder(vocab_size=100, encoder_output_size=256)
            >>> yseq = torch.tensor([1, 2, 3])  # Example sequence of tokens
            >>> state = decoder.init_state(x)
            >>> x = torch.randn(1, 256)  # Example encoder output
            >>> logp, new_state = decoder.score(yseq, state, x)
            >>> print(logp.shape)  # Should output: torch.Size([100])

        Note:
            This method supports both single and multiple encoder modes. In 
            single encoder mode, the encoder output is directly used. In 
            multiple encoder mode, attention weights are computed for each 
            encoder output.

        Raises:
            ValueError: If the number of encoders is less than one.
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
