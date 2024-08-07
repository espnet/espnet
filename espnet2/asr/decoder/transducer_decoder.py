"""(RNN-)Transducer decoder definition."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis, Hypothesis


class TransducerDecoder(AbsDecoder):
    """
        (RNN-)Transducer decoder module.

    This class implements a decoder for (RNN-)Transducer models, which can be used
    in automatic speech recognition (ASR) systems. It supports both LSTM and GRU
    as the underlying recurrent neural network types.

    Attributes:
        embed: Embedding layer for input tokens.
        dropout_embed: Dropout layer applied to the embedding output.
        decoder: List of RNN layers (LSTM or GRU).
        dropout_dec: List of dropout layers applied after each RNN layer.
        dlayers: Number of decoder layers.
        dunits: Number of hidden units in each decoder layer.
        dtype: Type of RNN used ('lstm' or 'gru').
        odim: Output dimension (vocabulary size).
        ignore_id: ID to ignore in the input (default: -1).
        blank_id: ID representing the blank/pad token.
        device: Device on which the model is allocated.

    Args:
        vocab_size: Size of the vocabulary (output dimension).
        rnn_type: Type of RNN to use ('lstm' or 'gru'). Defaults to 'lstm'.
        num_layers: Number of decoder layers. Defaults to 1.
        hidden_size: Number of hidden units in each decoder layer. Defaults to 320.
        dropout: Dropout rate for decoder layers. Defaults to 0.0.
        dropout_embed: Dropout rate for the embedding layer. Defaults to 0.0.
        embed_pad: ID of the padding token in the embedding layer. Defaults to 0.

    Raises:
        ValueError: If an unsupported RNN type is specified.

    Example:
        >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
        >>> labels = torch.randint(0, 1000, (32, 10))  # Batch of 32, sequence length 10
        >>> output = decoder(labels)
        >>> print(output.shape)
        torch.Size([32, 10, 320])  # (batch_size, sequence_length, hidden_size)

    Note:
        This implementation follows the Google Python Style Guide and PEP 8 standards.
        The decoder can be used as part of a larger (RNN-)Transducer model for ASR tasks.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        dropout: float = 0.0,
        dropout_embed: float = 0.0,
        embed_pad: int = 0,
    ):

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        dec_net = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU

        self.decoder = torch.nn.ModuleList(
            [
                dec_net(hidden_size, hidden_size, 1, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.dropout_dec = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for _ in range(num_layers)]
        )

        self.dlayers = num_layers
        self.dunits = hidden_size
        self.dtype = rnn_type
        self.odim = vocab_size

        self.ignore_id = -1
        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def set_device(self, device: torch.device):
        """
                Set GPU device to use.

        This method sets the device (GPU) on which the decoder will operate.

        Args:
            device (torch.device): The PyTorch device object representing the GPU
                to be used for computations.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> device = torch.device("cuda:0")
            >>> decoder.set_device(device)

        Note:
            This method should be called before performing any computations if you want
            to explicitly set the GPU device. If not called, the decoder will use the
            device of its parameters by default.
        """
        self.device = device

    def init_state(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """
                Initialize decoder states.

        This method initializes the hidden states of the decoder for a given batch size.
        It creates zero tensors for both LSTM (hidden state and cell state) and GRU (hidden state only).

        Args:
            batch_size (int): The batch size for which to initialize the states.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - h_n (torch.Tensor): The initial hidden state tensor of shape (N, B, D_dec),
                  where N is the number of layers, B is the batch size, and D_dec is the
                  decoder's hidden size.
                - c_n (Optional[torch.Tensor]): The initial cell state tensor for LSTM,
                  with the same shape as h_n. For GRU, this will be None.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
            >>> batch_size = 32
            >>> h_n, c_n = decoder.init_state(batch_size)
            >>> print(h_n.shape, c_n.shape)
            torch.Size([2, 32, 320]) torch.Size([2, 32, 320])

        Note:
            The initialized states are created on the same device as the decoder.
        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dunits,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def rnn_forward(
        self,
        sequence: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
                Encode source label sequences through the RNN layers.

        This method passes the input sequence through all RNN layers of the decoder,
        applying dropout after each layer.

        Args:
            sequence (torch.Tensor): RNN input sequences of shape (B, D_emb),
                where B is the batch size and D_emb is the embedding dimension.
            state (Tuple[torch.Tensor, Optional[torch.Tensor]]): Decoder hidden states.
                For LSTM, it's a tuple of (hidden_state, cell_state), each of shape (N, B, D_dec).
                For GRU, it's a tuple of (hidden_state, None).
                N is the number of layers, B is the batch size, and D_dec is the decoder's hidden size.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                - sequence (torch.Tensor): RNN output sequences of shape (B, D_dec).
                - (h_next, c_next) (Tuple[torch.Tensor, Optional[torch.Tensor]]):
                    Updated decoder hidden states. For LSTM, both h_next and c_next
                    have shape (N, B, D_dec). For GRU, c_next is None.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
            >>> batch_size, seq_len, hidden_size = 32, 10, 320
            >>> input_seq = torch.randn(batch_size, hidden_size)
            >>> initial_state = decoder.init_state(batch_size)
            >>> output_seq, (h_next, c_next) = decoder.rnn_forward(input_seq, initial_state)
            >>> print(output_seq.shape, h_next.shape, c_next.shape)
            torch.Size([32, 320]) torch.Size([2, 32, 320]) torch.Size([2, 32, 320])

        Note:
            This method handles both LSTM and GRU architectures based on the decoder's configuration.
        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(sequence.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                (
                    sequence,
                    (
                        h_next[layer : layer + 1],
                        c_next[layer : layer + 1],
                    ),
                ) = self.decoder[layer](
                    sequence, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                sequence, h_next[layer : layer + 1] = self.decoder[layer](
                    sequence, hx=h_prev[layer : layer + 1]
                )

            sequence = self.dropout_dec[layer](sequence)

        return sequence, (h_next, c_next)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
                Encode source label sequences.

        This method processes input label sequences through the decoder's embedding layer
        and RNN layers to produce decoder output sequences.

        Args:
            labels (torch.Tensor): Label ID sequences of shape (B, L), where B is the
                batch size and L is the sequence length.

        Returns:
            torch.Tensor: Decoder output sequences of shape (B, T, U, D_dec), where
                T is the number of time steps, U is the number of label tokens, and
                D_dec is the decoder's hidden size.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
            >>> batch_size, seq_length = 32, 10
            >>> labels = torch.randint(0, 1000, (batch_size, seq_length))
            >>> output = decoder(labels)
            >>> print(output.shape)
            torch.Size([32, 10, 320])

        Note:
            This method applies dropout to the embedded input before passing it through
            the RNN layers. The output shape might differ from the example if the decoder
            is configured differently.
        """
        init_state = self.init_state(labels.size(0))
        dec_embed = self.dropout_embed(self.embed(labels))

        dec_out, _ = self.rnn_forward(dec_embed, init_state)

        return dec_out

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
                Perform one-step forward computation for a single hypothesis.

        This method computes the decoder output for the next step given a current hypothesis.
        It uses a cache to store and retrieve previously computed results for efficiency.

        Args:
            hyp (Hypothesis): The current hypothesis containing the label sequence and decoder state.
            cache (Dict[str, Any]): A dictionary storing pairs of (dec_out, state) for each label sequence.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
                - dec_out (torch.Tensor): Decoder output sequence of shape (1, D_dec).
                - new_state (Tuple[torch.Tensor, Optional[torch.Tensor]]): Updated decoder hidden states.
                  For LSTM, it's (hidden_state, cell_state), each of shape (N, 1, D_dec).
                  For GRU, it's (hidden_state, None).
                - label (torch.Tensor): Label ID for language model, shape (1,).

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> hyp = Hypothesis(...)  # Create a hypothesis
            >>> cache = {}
            >>> dec_out, new_state, label = decoder.score(hyp, cache)
            >>> print(dec_out.shape, label.shape)
            torch.Size([1, 320]) torch.Size([1])

        Note:
            This method is typically used in beam search decoding for Transducer models.
            The cache helps avoid redundant computations for previously seen label sequences.
        """
        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_emb = self.embed(label)

            dec_out, dec_state = self.rnn_forward(dec_emb, hyp.dec_state)
            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0][0], dec_state, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
                Perform one-step forward computation for a batch of hypotheses.

        This method computes the decoder output for the next step given a batch of current hypotheses.
        It uses a cache to store and retrieve previously computed results for efficiency.

        Args:
            hyps (Union[List[Hypothesis], List[ExtendedHypothesis]]): A list of current hypotheses.
            dec_states (Tuple[torch.Tensor, Optional[torch.Tensor]]): Decoder hidden states.
                For LSTM, it's (hidden_state, cell_state), each of shape (N, B, D_dec).
                For GRU, it's (hidden_state, None).
            cache (Dict[str, Any]): A dictionary storing pairs of (dec_out, dec_states) for each label sequence.
            use_lm (bool): Whether to compute label ID sequences for language model integration.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - dec_out (torch.Tensor): Decoder output sequences of shape (B, D_dec).
                - dec_states (Tuple[torch.Tensor, torch.Tensor]): Updated decoder hidden states.
                  For LSTM, both elements have shape (N, B, D_dec). For GRU, the second element is None.
                - lm_labels (torch.Tensor): Label ID sequences for language model, shape (B,).
                  Returns None if use_lm is False.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> hyps = [Hypothesis(...) for _ in range(5)]  # Batch of 5 hypotheses
            >>> dec_states = decoder.init_state(5)
            >>> cache = {}
            >>> dec_out, new_states, lm_labels = decoder.batch_score(hyps, dec_states, cache, use_lm=True)
            >>> print(dec_out.shape, lm_labels.shape)
            torch.Size([5, 320]) torch.Size([5, 1])

        Note:
            This method is typically used in batch beam search decoding for Transducer models.
            It's more efficient than scoring hypotheses individually, especially for larger batch sizes.
        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_labels = "_".join(list(map(str, hyp.yseq)))

            if str_labels in cache:
                done[i] = cache[str_labels]
            else:
                process.append((str_labels, hyp.yseq[-1], hyp.dec_state))

        if process:
            labels = torch.LongTensor([[p[1]] for p in process], device=self.device)
            p_dec_states = self.create_batch_states(
                self.init_state(labels.size(0)), [p[2] for p in process]
            )

            dec_emb = self.embed(labels)
            dec_out, new_states = self.rnn_forward(dec_emb, p_dec_states)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                state = self.select_state(new_states, j)

                done[i] = (dec_out[j], state)
                cache[process[j][0]] = (dec_out[j], state)

                j += 1

        dec_out = torch.cat([d[0] for d in done], dim=0)
        dec_states = self.create_batch_states(dec_states, [d[1] for d in done])

        if use_lm:
            lm_labels = torch.LongTensor(
                [h.yseq[-1] for h in hyps], device=self.device
            ).view(final_batch, 1)

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None

    def select_state(
        self, states: Tuple[torch.Tensor, Optional[torch.Tensor]], idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
                Extract a specific state from the decoder hidden states.

        This method selects and returns the hidden state for a given index from the batch
        of decoder hidden states.

        Args:
            states (Tuple[torch.Tensor, Optional[torch.Tensor]]): Decoder hidden states.
                For LSTM, it's (hidden_state, cell_state), each of shape (N, B, D_dec).
                For GRU, it's (hidden_state, None).
                N is the number of layers, B is the batch size, and D_dec is the decoder's hidden size.
            idx (int): The index of the state to extract.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The selected decoder hidden state.
                - For LSTM: (hidden_state, cell_state), each of shape (N, 1, D_dec).
                - For GRU: (hidden_state, None), where hidden_state has shape (N, 1, D_dec).

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
            >>> batch_size = 32
            >>> states = decoder.init_state(batch_size)
            >>> selected_state = decoder.select_state(states, 5)
            >>> print(selected_state[0].shape, selected_state[1].shape)
            torch.Size([2, 1, 320]) torch.Size([2, 1, 320])

        Note:
            This method is useful when you need to extract the state for a specific
            hypothesis from a batch of states, typically during beam search decoding.
        """
        return (
            states[0][:, idx : idx + 1, :],
            states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

    def create_batch_states(
        self,
        states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        new_states: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        check_list: Optional[List] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
                Create batch decoder hidden states from individual states.

        This method combines individual decoder states into a batch, which is useful
        for parallel processing of multiple hypotheses.

        Args:
            states (Tuple[torch.Tensor, Optional[torch.Tensor]]): Initial decoder hidden states.
                For LSTM, it's (hidden_state, cell_state), each of shape (N, B, D_dec).
                For GRU, it's (hidden_state, None).
            new_states (List[Tuple[torch.Tensor, Optional[torch.Tensor]]]): List of individual
                decoder hidden states to be combined.
                Each element is of shape (N, 1, D_dec) for LSTM hidden and cell states,
                or (N, 1, D_dec) and None for GRU.
            check_list (Optional[List]): Not used in the current implementation.
                Kept for potential future use or compatibility.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Combined batch of decoder hidden states.
                - For LSTM: (hidden_state, cell_state), each of shape (N, B, D_dec).
                - For GRU: (hidden_state, None), where hidden_state has shape (N, B, D_dec).
                N is the number of layers, B is the new batch size (length of new_states),
                and D_dec is the decoder's hidden size.

        Example:
            >>> decoder = TransducerDecoder(vocab_size=1000, rnn_type='lstm', num_layers=2)
            >>> individual_states = [decoder.init_state(1) for _ in range(5)]
            >>> batch_states = decoder.create_batch_states(decoder.init_state(5), individual_states)
            >>> print(batch_states[0].shape, batch_states[1].shape)
            torch.Size([2, 5, 320]) torch.Size([2, 5, 320])

        Note:
            This method is particularly useful in beam search decoding when combining
            states from different hypotheses into a single batch for efficient processing.
        """
        return (
            torch.cat([s[0] for s in new_states], dim=1),
            (
                torch.cat([s[1] for s in new_states], dim=1)
                if self.dtype == "lstm"
                else None
            ),
        )
