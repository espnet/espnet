from collections import OrderedDict
from typing import Tuple

import torch
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
from torch_complex.tensor import ComplexTensor
import torchaudio
from espnet2.train.class_choices import ClassChoices


class TFMaskingNet(torch.nn.Module):
    """ TF Masking Speech Separation Net

    """

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            rnn_type: str = 'blstm',
            layer: int = 3,
            unit: int = 512,
            dropout: float = 0.0,
            num_spk: int = 2,
            none_linear: str = "sigmoid",
            utt_mvn: bool = False,
    ):
        super(TFMaskingNet, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        if utt_mvn:
            self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)

        else:
            self.utt_mvn = None

        self.rnn = RNN(
            idim=self.num_bin,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type)

        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(unit, self.num_bin)
                for _ in range(self.num_spk)
            ]
        )
        self.none_linear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[none_linear]

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
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
        x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        masks = OrderedDict()
        for linear in self.linear:
            y = linear(x)
            y = self.none_linear(y)
            masks.append(y)

        # apply mask
        predict_magnitude = [m * input_magnitude for m in masks]

        predicted_spectrums = [pm * input_phase for pm in predict_magnitude]

        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        return predicted_spectrums, flens, masks

    def forward_rawwav(
            self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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

        # complex spectrum -> raw wave
        predicted_wavs = [self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums]

        return predicted_wavs, ilens, masks
