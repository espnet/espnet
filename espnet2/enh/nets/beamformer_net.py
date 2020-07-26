from collections import OrderedDict
from typing import Optional

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.layers.dnn_beamformer import DNN_Beamformer
from espnet2.enh.layers.dnn_wpe import DNN_WPE
from espnet2.layers.stft import Stft


class BeamformerNet(AbsEnhancement):
    """TF Masking based beamformer

    """

    def __init__(
        self,
        num_spk: int = 1,
        normalize_input: bool = False,
        mask_type: str = "IPM^2",
        # STFT options
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        center: bool = True,
        window: Optional[str] = "hann",
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        # Dereverberation options
        use_wpe: bool = False,
        wnet_type: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        # Beamformer options
        use_beamformer: bool = True,
        bnet_type: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        badim: int = 320,
        ref_channel: int = -1,
        use_noise_mask: bool = True,
        beamformer_type="mvdr",
        bdropout_rate=0.0,
    ):
        super(BeamformerNet, self).__init__()

        self.mask_type = mask_type

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            window=window,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        )

        self.normalize_input = normalize_input
        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe

        if self.use_wpe:
            if use_dnn_mask_for_wpe:
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
                taps=taps,
                delay=delay,
                dropout_rate=wdropout_rate,
                iterations=iterations,
                use_dnn_mask=use_dnn_mask_for_wpe,
            )
        else:
            self.wpe = None

        self.ref_channel = ref_channel
        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(
                btype=bnet_type,
                bidim=self.num_bin,
                bunits=bunits,
                bprojs=bprojs,
                blayers=blayers,
                num_spk=num_spk,
                use_noise_mask=use_noise_mask,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
                beamformer_type=beamformer_type,
                btaps=taps,
                bdelay=delay,
            )
        else:
            self.beamformer = None

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, Nsample, Channel]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            enhanced speech  (single-channel):
                torch.Tensor or List[torch.Tensor]
            output lengths
            predcited masks: OrderedDict[
                'dereverb': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
                'noise1': torch.Tensor(Batch, Frames, Channel, Freq),
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
                enhanced, flens, mask_w = self.wpe(input_spectrum.unsqueeze(-2), flens)
                enhanced = enhanced.squeeze(-2)
                if mask_w is not None:
                    masks["dereverb"] = mask_w.squeeze(-2)

        elif input_spectrum.dim() == 4:
            # multi-channel input
            # 1. WPE
            if self.use_wpe:
                # (B, T, C, F)
                enhanced, flens, mask_w = self.wpe(input_spectrum, flens)
                if mask_w is not None:
                    masks["dereverb"] = mask_w

            # 2. Beamformer
            if self.use_beamformer:
                # enhanced: (B, T, C, F) -> (B, T, F)
                enhanced, flens, masks_b = self.beamformer(enhanced, flens)
                for spk in range(self.num_spk):
                    masks["spk{}".format(spk + 1)] = masks_b[spk]
                if len(masks_b) > self.num_spk:
                    masks["noise1"] = masks_b[self.num_spk]

        else:
            raise ValueError(
                "Invalid spectrum dimension: {}".format(input_spectrum.shape)
            )

        # Convert ComplexTensor to torch.Tensor
        # (B, T, F) -> (B, T, F, 2)
        if isinstance(enhanced, list):
            # multi-speaker output
            enhanced = [torch.stack([enh.real, enh.imag], dim=-1) for enh in enhanced]
        else:
            # single-speaker output
            enhanced = torch.stack([enhanced.real, enhanced.imag], dim=-1).float()
        return enhanced, flens, masks

    def forward_rawwav(self, input: torch.Tensor, ilens: torch.Tensor):
        """Output with wavformes.

        Args:
            input (torch.Tensor): mixed speech [Batch, Nsample, Channel]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech wavs (single-channel):
                torch.Tensor(Batch, Nsamples), or List[torch.Tensor(Batch, Nsamples)]
            output lengths
            predcited masks: OrderedDict[
                'dereverb': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
                'noise1': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """
        enhanced, flens, masks = self.forward(input, ilens)
        if isinstance(enhanced, list):
            # multi-speaker input
            predicted_wavs = [self.stft.inverse(ps, ilens)[0] for ps in enhanced]
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(enhanced, ilens)[0]

        return predicted_wavs, ilens, masks
