import torch

from espnet2.diar.attractor.abs_attractor import AbsAttractor


class RnnAttractor(AbsAttractor):
    """
    RnnAttractor is an encoder-decoder attractor model for speaker diarization.

    This class implements an RNN-based attractor that utilizes LSTM layers for 
    encoding and decoding speaker embeddings. It is designed to assist in 
    separating different speakers' voices in a mixed audio input.

    Attributes:
        attractor_encoder (torch.nn.LSTM): LSTM layer used for encoding the input.
        attractor_decoder (torch.nn.LSTM): LSTM layer used for decoding the output.
        dropout_layer (torch.nn.Dropout): Dropout layer for regularization.
        linear_projection (torch.nn.Linear): Linear layer for projecting the output.
        attractor_grad (bool): Flag to determine whether to allow gradient flow 
            through the attractor.

    Args:
        encoder_output_size (int): Size of the output features from the encoder.
        layer (int, optional): Number of LSTM layers. Defaults to 1.
        unit (int, optional): Number of units in each LSTM layer. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        attractor_grad (bool, optional): If True, allows gradient flow through 
            the attractor. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - attractor (torch.Tensor): Output attractor of shape 
            [Batch, num_spk + 1, F].
            - att_prob (torch.Tensor): Attention probabilities of shape 
            [Batch, num_spk + 1, 1].

    Examples:
        # Create an RnnAttractor instance
        rnn_attractor = RnnAttractor(encoder_output_size=128)

        # Forward pass through the attractor
        enc_input = torch.randn(10, 50, 128)  # Example input tensor
        ilens = torch.tensor([50] * 10)  # Input lengths
        dec_input = torch.zeros(10, 5, 128)  # Example decoder input (zeros)

        attractor, att_prob = rnn_attractor(enc_input, ilens, dec_input)

    Note:
        This class requires PyTorch to be installed in the environment.
    """

    def __init__(
        self,
        encoder_output_size: int,
        layer: int = 1,
        unit: int = 512,
        dropout: float = 0.1,
        attractor_grad: bool = True,
    ):
        super().__init__()
        self.attractor_encoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )
        self.attractor_decoder = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=unit,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.linear_projection = torch.nn.Linear(unit, 1)

        self.attractor_grad = attractor_grad

    def forward(
        self,
        enc_input: torch.Tensor,
        ilens: torch.Tensor,
        dec_input: torch.Tensor,
    ):
        """
        Perform the forward pass of the RnnAttractor model.

        This method takes the encoded input, input lengths, and decoder input to 
        compute the attractor outputs and attention probabilities. It processes 
        the inputs through the encoder and decoder LSTM layers, applying dropout 
        and linear projection to produce the final outputs.

        Args:
            enc_input (torch.Tensor): 
                Hidden space of shape [Batch, T, F] where T is the sequence length 
                and F is the feature dimension.
            ilens (torch.Tensor): 
                Input lengths of shape [Batch], indicating the actual lengths of 
                the sequences in the batch.
            dec_input (torch.Tensor): 
                Decoder input of shape [Batch, num_spk + 1, F] initialized to zeros, 
                where num_spk is the number of speakers.

        Returns:
            tuple: A tuple containing:
                - attractor (torch.Tensor): 
                    Output attractor of shape [Batch, num_spk + 1, F].
                - att_prob (torch.Tensor): 
                    Attention probabilities of shape [Batch, num_spk + 1, 1].
        
        Examples:
            >>> enc_input = torch.randn(4, 10, 128)  # Batch of 4, seq length 10, 128 features
            >>> ilens = torch.tensor([10, 8, 6, 4])  # Input lengths for each sequence
            >>> dec_input = torch.zeros(4, 3, 128)    # Decoder input for 3 speakers
            >>> attractor, att_prob = model.forward(enc_input, ilens, dec_input)
            >>> print(attractor.shape)  # Expected output: [4, 3, 128]
        >>> print(att_prob.shape)    # Expected output: [4, 3, 1]
        """
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            enc_input, lengths=ilens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hs = self.attractor_encoder(pack)
        attractor, _ = self.attractor_decoder(dec_input, hs)
        attractor = self.dropout_layer(attractor)
        if self.attractor_grad is True:
            att_prob = self.linear_projection(attractor)
        else:
            att_prob = self.linear_projection(attractor.detach())
        return attractor, att_prob
