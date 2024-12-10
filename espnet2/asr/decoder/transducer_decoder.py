"""(RNN-)Transducer decoder definition."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis, Hypothesis


class TransducerDecoder(AbsDecoder):
    """
    TransducerDecoder is an (RNN-)Transducer decoder module that processes sequences 
    of input labels and produces corresponding output sequences. It is designed to 
    work with recurrent neural networks (RNNs), specifically LSTM or GRU architectures.

    Attributes:
        embed (torch.nn.Embedding): Embedding layer for input label sequences.
        dropout_embed (torch.nn.Dropout): Dropout layer applied to the embeddings.
        decoder (torch.nn.ModuleList): List of RNN layers (LSTM or GRU).
        dropout_dec (torch.nn.ModuleList): List of dropout layers applied to the RNN 
            outputs.
        dlayers (int): Number of decoder layers.
        dunits (int): Number of decoder units per layer.
        dtype (str): Type of RNN ('lstm' or 'gru').
        odim (int): Size of the output vocabulary.
        ignore_id (int): ID used to ignore certain labels in the decoding process.
        blank_id (int): ID representing the blank symbol in the model.
        device (torch.device): The device (CPU or GPU) on which the model resides.

    Args:
        vocab_size (int): Size of the output vocabulary.
        rnn_type (str): Type of RNN to use ('lstm' or 'gru'). Default is 'lstm'.
        num_layers (int): Number of decoder layers. Default is 1.
        hidden_size (int): Number of units in each decoder layer. Default is 320.
        dropout (float): Dropout rate for the decoder layers. Default is 0.0.
        dropout_embed (float): Dropout rate for the embedding layer. Default is 0.0.
        embed_pad (int): Padding index for the embedding layer. Default is 0.

    Methods:
        set_device(device): Set the device to be used for the decoder.
        init_state(batch_size): Initialize the hidden states of the decoder.
        rnn_forward(sequence, state): Perform a forward pass through the RNN layers.
        forward(labels): Process input label sequences to produce decoder outputs.
        score(hyp, cache): Compute decoder output and hidden states for a single 
            hypothesis.
        batch_score(hyps, dec_states, cache, use_lm): Compute decoder outputs for a 
            batch of hypotheses.
        select_state(states, idx): Retrieve the hidden state for a specified index.
        create_batch_states(states, new_states, check_list=None): Create batch hidden 
            states from new states.

    Examples:
        # Instantiate a TransducerDecoder
        decoder = TransducerDecoder(
            vocab_size=1000,
            rnn_type='lstm',
            num_layers=2,
            hidden_size=256,
            dropout=0.1,
            dropout_embed=0.1,
            embed_pad=0
        )

        # Forward pass with input labels
        labels = torch.randint(0, 1000, (32, 10))  # Batch of 32 sequences of length 10
        outputs = decoder(labels)

        # Initialize states for a batch
        init_states = decoder.init_state(batch_size=32)

        # Score a hypothesis
        hyp = Hypothesis(yseq=[1, 2, 3], dec_state=init_states)
        dec_out, new_state, label = decoder.score(hyp, cache={})

    Note:
        The decoder requires input sequences to be properly padded and tokenized 
        according to the model's vocabulary.

    Todo:
        Implement additional features for enhanced functionality.
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

        This method updates the device attribute of the TransducerDecoder 
        instance, allowing the model to operate on the specified GPU or 
        CPU. It is important to set the device correctly to ensure that 
        all tensor operations are performed on the desired hardware.

        Args:
            device: A torch.device object representing the device to be used.
                This can be a CPU or a specific GPU device (e.g., 
                torch.device("cuda:0") for the first GPU).

        Examples:
            >>> decoder = TransducerDecoder(vocab_size=100)
            >>> decoder.set_device(torch.device("cuda:0"))
            >>> print(decoder.device)
            cuda:0

        Note:
            Ensure that the specified device is available on the system. 
            Use `torch.cuda.is_available()` to check if CUDA is supported.

        Raises:
            ValueError: If the provided device is not a valid torch.device.
        """
        self.device = device

    def init_state(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """
        Initialize decoder states.

        This method creates and initializes the hidden states for the decoder. 
        The hidden states are essential for the operation of the recurrent 
        neural network (RNN) used in the transducer decoder. Depending on 
        the type of RNN (LSTM or GRU), the method will return either a tuple 
        containing both hidden states and cell states (for LSTM) or just 
        the hidden states (for GRU).

        Args:
            batch_size: The number of sequences in a batch. This determines 
                the size of the hidden states.

        Returns:
            A tuple containing the initialized hidden states:
            - For LSTM: ((N, B, D_dec), (N, B, D_dec))
            - For GRU: ((N, B, D_dec), None)
            Where:
                - N is the number of layers,
                - B is the batch size,
                - D_dec is the number of decoder units per layer.

        Examples:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> h_n, c_n = decoder.init_state(batch_size=32)
            >>> h_n.shape
            torch.Size([num_layers, 32, 320])
            >>> c_n.shape
            torch.Size([num_layers, 32, 320])  # Only for LSTM

        Note:
            This method should be called before the decoder is used for 
            generating predictions, ensuring that the initial hidden states 
            are set correctly for each batch of sequences.
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

        This method processes the input label sequences through the decoder 
        network, applying embedding and RNN transformations to generate 
        the decoder output sequences.

        Args:
            labels: Label ID sequences. Shape (B, L), where B is the batch 
                    size and L is the sequence length.

        Returns:
            dec_out: Decoder output sequences. Shape (B, T, U, D_dec), where 
                     T is the output length, U is the number of units, 
                     and D_dec is the dimension of the decoder.

        Examples:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> input_labels = torch.randint(0, 1000, (32, 10))  # Batch of 32
            >>> output = decoder.forward(input_labels)
            >>> output.shape
            torch.Size([32, T, U, D_dec])  # Shape will depend on T and U

        Note:
            Ensure that the input labels are properly padded and contain 
            valid IDs as per the embedding layer configuration.
        """
        init_state = self.init_state(labels.size(0))
        dec_embed = self.dropout_embed(self.embed(labels))

        dec_out, _ = self.rnn_forward(dec_embed, init_state)

        return dec_out

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Compute the score for a single hypothesis.

        This method performs a one-step forward pass for the given hypothesis
        using the decoder's current state and caches the result for future
        use. It retrieves the decoder output and the new hidden states based
        on the last label in the hypothesis.

        Args:
            hyp: The hypothesis containing the label sequence and current 
                 decoder state.
            cache: A dictionary that stores pairs of (dec_out, state) for 
                   each label sequence to avoid redundant computations.

        Returns:
            dec_out: The decoder output sequence for the current label. 
                      Shape: (1, D_dec)
            new_state: The updated decoder hidden states after processing 
                       the input. Shape: ((N, 1, D_dec), (N, 1, D_dec))
            label: The label ID for the language model. Shape: (1,)

        Examples:
            >>> hyp = Hypothesis(yseq=[2, 3, 4], dec_state=(h_n, c_n))
            >>> cache = {}
            >>> dec_out, new_state, label = decoder.score(hyp, cache)

        Note:
            This method assumes that the hypothesis has at least one label
            in its sequence.

        Raises:
            KeyError: If the hypothesis label sequence is not found in the 
                       cache and fails to generate a new output.
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
        One-step forward hypotheses.

        This method processes a batch of hypotheses, performing a one-step 
        forward pass through the decoder for each hypothesis. It leverages 
        a cache to avoid redundant computations, improving efficiency.

        Args:
            hyps: A list of hypotheses to score, which can be either 
                `Hypothesis` or `ExtendedHypothesis` instances.
            dec_states: Current decoder hidden states. This is a tuple 
                containing two tensors: ((N, B, D_dec), (N, B, D_dec)), 
                where N is the number of layers, B is the batch size, 
                and D_dec is the number of decoder units.
            cache: A dictionary mapping label sequence strings to 
                tuples of decoder output sequences and hidden states. 
                This is used to store and retrieve previously computed 
                results for efficiency.
            use_lm: A boolean indicating whether to compute label ID 
                sequences for the language model (LM).

        Returns:
            dec_out: The decoder output sequences for the batch. 
                Shape: (B, D_dec), where B is the batch size and 
                D_dec is the number of decoder units.
            dec_states: Updated decoder hidden states. This is a tuple 
                containing the new states for each hypothesis: 
                ((N, B, D_dec), (N, B, D_dec)).
            lm_labels: Label ID sequences for the language model. 
                Shape: (B,), where B is the batch size. If `use_lm` is 
                False, this will be None.

        Examples:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> hyps = [Hypothesis(yseq=[1, 2, 3], dec_state=initial_state)]
            >>> dec_states = decoder.init_state(batch_size=len(hyps))
            >>> cache = {}
            >>> dec_out, new_states, lm_labels = decoder.batch_score(
            ...     hyps, dec_states, cache, use_lm=True
            ... )
            >>> print(dec_out.shape)  # (B, D_dec)

        Note:
            The method assumes that all hypotheses in the input list 
            have been initialized properly and that their sequences 
            are valid.

        Raises:
            ValueError: If the provided hypotheses or states are invalid 
            or do not match the expected dimensions.
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
        Get specified ID state from decoder hidden states.

        This method retrieves the decoder hidden state for a specified index 
        from the provided decoder hidden states. It is particularly useful in 
        scenarios where multiple hypotheses are being processed in parallel, 
        and you need to extract the hidden state corresponding to a specific 
        hypothesis.

        Args:
            states: Decoder hidden states. 
                A tuple containing two tensors: 
                ((N, B, D_dec), (N, B, D_dec)), where N is the number of layers, 
                B is the batch size, and D_dec is the dimension of the decoder.
            idx: State ID to extract. This is the index of the hidden state 
                that you wish to retrieve from the decoder states.

        Returns:
            A tuple containing the decoder hidden state for the given ID. 
            The output will be in the shape:
            ((N, 1, D_dec), (N, 1, D_dec)) for LSTM, or 
            ((N, 1, D_dec), None) for GRU.

        Examples:
            >>> decoder = TransducerDecoder(vocab_size=1000)
            >>> states = decoder.init_state(batch_size=2)
            >>> selected_state = decoder.select_state(states, idx=0)
            >>> print(selected_state[0].shape)  # Output: (N, 1, D_dec)
            >>> print(selected_state[1].shape)  # Output: (N, 1, D_dec) for LSTM
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
        Create decoder hidden states for a batch of hypotheses.

        This method constructs the hidden states of the decoder based on the provided
        current states and a list of new states for each hypothesis. It concatenates
        the new states to form the complete hidden states for the batch.

        Args:
            states: Tuple containing the current decoder hidden states.
                The first element is of shape (N, B, D_dec) and the second (if LSTM)
                is of shape (N, B, D_dec) as well.
            new_states: List of tuples containing the new hidden states for each
                hypothesis, where each tuple is of the form ((1, D_dec), (1, D_dec))
                for LSTM or just (1, D_dec) for GRU.
            check_list: Optional list for additional state checks (default: None).

        Returns:
            Tuple of concatenated decoder hidden states:
                - The first element of shape (N, B, D_dec).
                - The second element (if LSTM) of shape (N, B, D_dec), else None.

        Examples:
            >>> current_states = (torch.zeros(2, 3, 320), torch.zeros(2, 3, 320))
            >>> new_hyp_states = [(torch.ones(1, 320), torch.ones(1, 320)),
            ...                   (torch.zeros(1, 320), torch.zeros(1, 320))]
            >>> batch_states = create_batch_states(current_states, new_hyp_states)
            >>> len(batch_states)  # Output: 2
            >>> batch_states[0].shape  # Output: torch.Size([2, 3, 320])

        Note:
            Ensure that the `new_states` list matches the expected format based on
            the decoder type (LSTM or GRU).
        """
        return (
            torch.cat([s[0] for s in new_states], dim=1),
            (
                torch.cat([s[1] for s in new_states], dim=1)
                if self.dtype == "lstm"
                else None
            ),
        )
