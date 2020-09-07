from collections import OrderedDict
from typing import Tuple

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
import torch
from torch_complex.tensor import ComplexTensor


class TFMaskingNet(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        nonlinear: str = "sigmoid",
        utt_mvn: bool = False,
        mask_type: str = "IRM",
        loss_type: str = "mask_mse",
        # RNN realted
        layer: int = 3,
        unit: int = 512,
        dropout: float = 0.0,
        # Transformer Related
        idim: int = 512,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        positionwise_layer_type: str = "conv1d",
        transformer_input_layer: str = "conv2d",
        positionwise_conv_kernel_size: int = 1,
        encoder_normalize_before: bool = False,
        encoder_concat_after: bool = False,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        use_scaled_pos_enc: bool = True,
        # conformer
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,

    ):
        super(TFMaskingNet, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        self.return_spec_in_training = False
        if loss_type not in ("mask_mse", "magnitude", "spectrum"):
            raise ValueError("Unsupported loss type: %s" % loss_type)
        self.rnn_type = rnn_type

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        if utt_mvn:
            self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)

        else:
            self.utt_mvn = None

        if rnn_type == "transformer":
            # get positional encoding class
            pos_enc_class = (
                ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
            )
            self.rnn = TransformerEncoder(
                idim=self.num_bin,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=transformer_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif rnn_type == "conformer":
            self.rnn = ConformerEncoder(
                idim=self.num_bin,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=transformer_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
            )
            self.linear = torch.nn.ModuleList(
                [torch.nn.Linear(adim, self.num_bin) for _ in range(self.num_spk)]
            )
        else:
            self.rnn = RNN(
                idim=self.num_bin,
                elayers=layer,
                cdim=unit,
                hdim=unit,
                dropout=dropout,
                typ=rnn_type,
            )

            self.linear = torch.nn.ModuleList(
                [torch.nn.Linear(unit, self.num_bin) for _ in range(self.num_spk)]
            )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            separated (List[ComplexTensor]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Freq),
                'spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)
        input_phase = input_spectrum / (input_magnitude + 10e-12)

        # apply utt mvn
        if self.utt_mvn:
            input_magnitude_mvn, fle = self.utt_mvn(input_magnitude, flens)
        else:
            input_magnitude_mvn = input_magnitude

        # predict masks for each speaker
        if self.rnn_type == 'blstm':
            x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        else: # transformer-based encoders
            pad_mask = make_non_pad_mask(flens).unsqueeze(1).to(input_magnitude_mvn.device)
            # print('pad_mask:', pad_mask.shape, pad_mask[:,:,-10:])
            x, flens = self.rnn(input_magnitude_mvn, pad_mask)
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.return_spec_in_training:
            predict_magnitude = [input_magnitude * m for m in masks]
            predicted_spectrums = [input_phase * pm for pm in predict_magnitude]
        else:
            if self.training and self.loss_type.startswith("mask"):
                predicted_spectrums = None
            else:
                # apply mask
                predict_magnitude = [input_magnitude * m for m in masks]
                predicted_spectrums = [input_phase * pm for pm in predict_magnitude]

        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        return predicted_spectrums, flens, masks

    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech [Batch, num_speaker, sample]
            output lengths
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Freq),
                'spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # predict spectrum for each speaker
        predicted_spectrums, flens, masks = self.forward(input, ilens)

        if predicted_spectrums is None:
            predicted_wavs = None
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(predicted_spectrums, ilens)[0]

        return predicted_wavs, ilens, masks
