from functools import reduce
import operator
import torch
import torchaudio.transforms as T
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from typing import Any, Dict, List, Optional, Tuple


class SEANetEnhEncoder(AbsEncoder):
    """Codec encoder for speech enhancement and separation"""

    def __init__(
        self,
        codec_rate: int = 16000,
        sample_rate: int = 8000,
        encdec_channels: int = 1,
        hidden_dim: int = 128,
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
    ):
        super().__init__()
        self.codec_encoder = SEANetEncoder(
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
        )
        self.codec_fs = codec_rate
        self.sample_fs = sample_rate
        if self.codec_fs != self.sample_fs:
            self.dac_sampler = T.Resample(self.sample_fs, self.codec_fs)
        self._output_dim = hidden_dim
        self.subsample = reduce(operator.mul, encdec_ratios)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = torch.unsqueeze(input, 1)
        if self.codec_fs != self.sample_fs:
            self.dac_sampler = self.dac_sampler.to(input.device)
            input = self.resample_audio(input)
            
        feature = self.codec_encoder(input).transpose(1, 2)

        flens = ilens.clone().apply_(lambda x: ((x//self.subsample) +1 ))
        return feature, flens

    def resample_audio(self, x):
        '''
        torchaudio resample function used here only requires last dimension to be time.
        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        '''
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1 #model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate
        
        x_resamp = self.dac_sampler(x)
        
        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(),dim=2,keepdim=True)[0]

        return x_resamp.to(device)