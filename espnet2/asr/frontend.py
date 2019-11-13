import copy
from typing import Optional, Tuple

import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.feature_transform import \
    FeatureTransform
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_values import get_defaut_values


class Frontend1(torch.nn.Module):
    """Conventional frontend structure for ASR

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """
    def __init__(
        self,
        stft_conf: dict = get_defaut_values(Stft),
        frontend_conf: Optional[dict] = get_defaut_values(Frontend),
        feature_transform_conf: dict = get_defaut_values(FeatureTransform)
    ):
        super().__init__()

        # Deepcopy (In general, dict shouldn't be used as default arg)
        stft_conf = copy.deepcopy(stft_conf)
        frontend_conf = copy.deepcopy(frontend_conf)
        feature_transform_conf = copy.deepcopy(feature_transform_conf)

        self.stft = Stft(**stft_conf)
        if frontend_conf is not None:
            self.frontend = Frontend(**frontend_conf)
        else:
            self.frontend = None

        self.feature_transform = \
            FeatureTransform(**feature_transform_conf)

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape
        # input_stft: (..., F, T, 2) -> (..., F, T)
        input_stft = \
            ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        # input_stft: (..., F, T) -> (..., T, F)
        input_stft = input_stft.transpose(-1, -2)

        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, [Channel,] Length, Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. Feature transform e.g. Stft -> Mel-Fbank
        # input_stft: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.feature_transform(input_stft, feats_lens)

        return input_feats, feats_lens
