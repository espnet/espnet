from collections import OrderedDict
from typing import Tuple

import torch
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.layers.stft import Stft
from espnet2.asr.frontend.nets.dnn_wpe import DNN_Beamformer
from espnet2.asr.frontend.nets.dnn_wpe import DNN_WPE
from torch_complex.tensor import ComplexTensor
import torchaudio


class BeamformerNet(torch.nn.Module):
    """ TF Masking based beamformer

    """

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            num_spk: int = 2,
            normalize_input: bool = False,
            # Dereverberation options
            use_wpe: bool = False,
            wnet_type: str = 'blstmp',
            wlayers: int = 3,
            wunits: int = 300,
            wprojs: int = 320,
            wdropout_rate: float = 0.0,
            taps: int = 5,
            delay: int = 3,
            use_dnn_mask_for_wpe: bool = True,

            # Beamformer options
            use_beamformer: bool = False,
            bnet_type: str = 'blstmp',
            blayers: int = 3,
            bunits: int = 300,
            bprojs: int = 320,
            badim: int = 320,
            ref_channel: int = -1,
            beamformer_type='mvdr',
            bdropout_rate=0.0,
    ):
        super(BeamformerNet, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.normalize_input = normalize_input
        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe

        if self.use_wpe:
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(
                wtype=wnet_type,
                widim=self.num_bin,
                wunits=wunits,
                wprojs=wprojs,
                wlayers=wlayers,
                num_spkr=num_spk,
                taps=taps,
                delay=delay,
                dropout_rate=wdropout_rate,
                iterations=iterations,
                use_dnn_mask=use_dnn_mask_for_wpe,
            )
        else:
            self.wpe = None

        if self.use_beamformer:
            if beamformer_type == 'mvdr':
                bnmask = num_spk + 1
            elif beamformer_type == 'mpdr':
                bnmask = num_spk
            else:
                raise ValueError(
                    "Not supporting beamformer_type={}".format(beamformer_type)
                )
            self.beamformer = DNN_Beamformer(
                btype=bnet_type,
                bidim=self.num_bin,
                bunits=bunits,
                bprojs=bprojs,
                blayers=blayers,
                bnmask=bnmask,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
                beamformer_type=beamformer_type,
            )
        else:
            self.beamformer = None

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            enhanced speech:
                ComplexTensor, or List[ComplexTensor, ComplexTensor]
            output lengths
            predcited masks: OrderedDict[
                'spk1': List[ComplexTensor(Batch, Frames, Channel, Freq)],
                'spk2': List[ComplexTensor(Batch, Frames, Channel, Freq)],
                ...
                'spkn': List[ComplexTensor(Batch, Frames, Channel, Freq)],
                'noise': List[ComplexTensor(Batch, Frames, Channel, Freq)],
            ]
        """
        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        # (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        if self.normalize_input:
            input_spectrum = input_spectrum / abs(input_spectrum).max()

        enhanced = input_spectrum
        masks = OrderedDict()

        if input_spectrum.dim() == 3:
            # single-channel input
            if self.use_wpe:
                # (B, T, F)
                enhanced, flens, masks_w = self.wpe(input_spectrum, flens)
                if masks_w is not None:
                    for spk in range(self.num_spk):
                        masks.setdefault('spk{}'.format(spk + 1), []).append(masks_w[spk])
            return enhanced, flens, masks

        # multi-channel input
        # 1. WPE
        if self.use_wpe:
            # (B, T, C, F)
            enhanced, flens, masks_w = self.wpe(input_spectrum, flens)
            if masks_w is not None:
                for spk in range(self.num_spk):
                    masks.setdefault('spk{}'.format(spk + 1), []).append(masks_w[spk])

        # 2. Beamformer
        if self.use_beamformer:
            # enhanced: (B, T, C, F) -> (B, T, F)
            enhanced, flens, masks_b = self.beamformer(enhanced, flens)
            for spk in range(self.num_spk):
                masks.setdefault('spk{}'.format(spk + 1), []).append(masks_b[spk])
            if len(masks_b) > self.num_spk:
                masks.setdefault('noise', []).append(masks_b[-1])

        return enhanced, flens, masks

    def forward_rawwav(
            self, input: torch.Tensor, ilens: torch.Tensor
    ):
        """
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech wavs:
                torch.Tensor(Batch, sample), or List[torch.Tensor(Batch, sample)]
            output lengths
            predcited masks: OrderedDict[
                'spk1': List[ComplexTensor],
                'spk2': List[ComplexTensor],
                ...
                'spkn': List[ComplexTensor],
                'noise': List[ComplexTensor],
            ]
        """
        enhanced, flens, masks = self.forward(input, ilens)
        if isinstance(enhanced, list):
            #  multi-speaker input
            predicted_spectrums = [torch.stack([enh.real, enh.imag], dim=-1) for enh in enhanced]
            predicted_wavs = torch.stack([self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums], dim=1)
        else:
            # single-speaker input
            predicted_spectrum = torch.stack([enhanced.real, enhanced.imag], dim=-1)
            predicted_wavs = self.stft.inverse(predicted_spectrum, ilens)[0]

        return predicted_wavs, ilens, masks
