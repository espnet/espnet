from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder


class SEANetEnhDecoder(AbsDecoder):
    """Codec decoder for speech enhancement and separation"""

    def __init__(
        self,
        codec_rate: int = 16000,
        sample_rate: int = 8000,
        hidden_dim: int = 128,
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[int] = [8, 5, 4, 2],
        encdec_activation: str = "ELU",
        encdec_activation_params: Dict[str, Any] = {"alpha": 1.0},
        encdec_norm: str = "weight_norm",
        encdec_norm_params: Dict[str, Any] = {},
        encdec_kernel_size: int = 7,
        encdec_residual_kernel_size: int = 7,
        encdec_last_kernel_size: int = 7,
        encdec_dilation_base: int = 2,
        encdec_causal: bool = False,
        encdec_pad_mode: str = "reflect",
        encdec_true_skip: bool = False,
        encdec_compress: int = 2,
        encdec_lstm: int = 2,
        decoder_trim_right_ratio: float = 1.0,
        decoder_final_activation: Optional[str] = None,
        decoder_final_activation_params: Optional[dict] = None,
    ):
        super().__init__()
        self.codec_decoder = SEANetDecoder(
            channels=encdec_channels,
            dimension=hidden_dim,
            n_filters=encdec_n_filters,
            n_residual_layers=encdec_n_residual_layers,
            ratios=encdec_ratios,
            activation=encdec_activation,
            activation_params=encdec_activation_params,
            norm=encdec_norm,
            norm_params=encdec_norm_params,
            kernel_size=encdec_kernel_size,
            residual_kernel_size=encdec_residual_kernel_size,
            last_kernel_size=encdec_last_kernel_size,
            dilation_base=encdec_dilation_base,
            causal=encdec_causal,
            pad_mode=encdec_pad_mode,
            true_skip=encdec_true_skip,
            compress=encdec_compress,
            lstm=encdec_lstm,
            trim_right_ratio=decoder_trim_right_ratio,
            final_activation=decoder_final_activation,
            final_activation_params=decoder_final_activation_params,
        )
        self.codec_fs = codec_rate
        self.sample_fs = sample_rate
        if self.codec_fs != self.sample_fs:
            self.dac_sampler = T.Resample(self.codec_fs, self.sample_fs)
        self._output_dim = hidden_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch] (not used)
            fs (int): sampling rate in Hz (Not used)
        """
        wav = self.codec_decoder(input.transpose(1, 2))[:, 0, :]

        # Resampling back to original sampling rate if needed
        wav = wav.unsqueeze(1)
        if self.codec_fs != self.sample_fs:
            self.dac_sampler = self.dac_sampler.to(wav.device)
            wav = self.resample_audio(wav)
        wav = wav.squeeze(1)

        # T might have changed due to model. If so, fix it here
        T_origin = ilens.max()
        if wav.shape[-1] != T_origin:

            T_est = wav.shape[-1]
            if T_origin > T_est:
                wav = F.pad(wav, (0, T_origin - T_est))
            else:
                wav = wav[:, :T_origin]
        return wav, ilens

    def resample_audio(self, x):
        """
        torchaudio resample function used here only requires last dimension to be time.
        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        """
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1  # model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate

        x_resamp = self.dac_sampler(x)

        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0]

        return x_resamp.to(device)
