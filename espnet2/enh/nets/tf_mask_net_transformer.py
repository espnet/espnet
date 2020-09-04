from collections import OrderedDict
from typing import Tuple
from typing import Optional
#from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
import torch
from torch_complex.tensor import ComplexTensor


class TFMaskingTransformer(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
        self,
        n_fft: int = 256, 
        win_length: int = None,
        hop_length: int = 128,
        dnn_type: str = "transformer",
        #layer: int = 3,
        #unit: int = 512,
        dropout: float = 0.0,
        num_spk: int = 2,
        nonlinear: str = "sigmoid",
        utt_mvn: bool = False,
        mask_type: str = "IRM",
        loss_type: str = "mask_mse",
        d_model: int = 256,
        nhead: int = 4,
        linear_units: int = 2048,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "linear",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    
    ):
        super(TFMaskingTransformer, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        if loss_type not in ("mask_mse", "magnitude", "spectrum"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.stft = Stft(n_fft=n_fft, win_length=win_length, hop_length=hop_length,)

        if utt_mvn:
            self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)

        else:
            self.utt_mvn = None

        #self.rnn = RNN(
        #    idim=self.num_bin,
        #    elayers=layer,
        #    cdim=unit,
        #    hdim=unit,
        #    dropout=dropout,
        #    typ=rnn_type,
        #)

        self.encoder = TransformerEncoder(
             input_size=self.num_bin,
             output_size=d_model,
             attention_heads=nhead,
             linear_units=linear_units,
             num_blocks=num_layers,
             positional_dropout_rate=positional_dropout_rate,
             attention_dropout_rate=attention_dropout_rate,
             input_layer=input_layer,
             normalize_before=normalize_before,
             concat_after=concat_after,
             positionwise_layer_type=positionwise_layer_type,
             positionwise_conv_kernel_size=positionwise_conv_kernel_size,
             padding_idx=padding_idx,
        )
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(d_model, self.num_bin) for _ in range(self.num_spk)]
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
            separated (list[ComplexTensor]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
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
        #x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        x, olens, _ = self.encoder(input_magnitude_mvn, flens)

        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

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
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
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
                                              
