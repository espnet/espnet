from typing import Tuple

import torch
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
from torch_complex.tensor import ComplexTensor
import torchaudio


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
        self.none_linear = torch.sigmoid

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            predcited magnitude masks [Batch, num_speaker, T, F]
            output lengths
        """

        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)

        # apply utt mvn
        if self.utt_mvn:
            input_magnitude_mvn, fle = self.utt_mvn(input_magnitude, flens)
        else:
            input_magnitude_mvn = input_magnitude

        # predict masks for each speaker
        x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.none_linear(y)
            masks.append(y)

        # apply mask
        predict_magnitude = [m * input_magnitude for m in masks]

        predict_magnitude = torch.stack(predict_magnitude, dim=1)

        return predict_magnitude, flens

    def forward_rawwav(
            self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            predcited speech [Batch, num_speaker, sample]
            output lengths
        """

        # compute phase spectrum from mixed speech
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)
        input_phase = input_spectrum / (input_magnitude + 10e-12)

        # predict magnitude spectrum for each speaker
        predcited_magnitudes, flens = self.forward(input, ilens)
        predcited_magnitudes = torch.unbind(predcited_magnitudes, dim=1)

        # magnitude spectrum -> complex spectrum -> raw wave
        predicted_spectrums = [pm * input_phase for pm in predcited_magnitudes]
        predicted_spectrums = [torch.stack([ps.real, ps.imag], dim=-1) for ps in predicted_spectrums]

        predicted_wavs = torch.stack([self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums], dim=1)

        return predicted_wavs, ilens
