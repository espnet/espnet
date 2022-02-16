import torch

from espnet2.diar.attractor.abs_attractor import AbsAttractor


class RnnAttractor(AbsAttractor):
    """encoder decoder attractor for speaker diarization"""

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
        """Forward.

        Args:
            enc_input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            dec_input (torch.Tensor): decoder input (zeros) [Batch, num_spk + 1, F]

        Returns:
            attractor: [Batch, num_spk + 1, F]
            att_prob: [Batch, num_spk + 1, 1]
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
