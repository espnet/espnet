from typing import Tuple
from typing import Union
from typing import Dict

import humanfriendly
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.frontend.nets.tf_mask_net import TFMaskingNet
from espnet2.train.class_choices import ClassChoices
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.stft import Stft

frontend_choices = ClassChoices(
    name="enh_model",
    classes=dict(tf_masking=TFMaskingNet),
    type_check=torch.nn.Module,
    default="enh",
)


class EnhFrontend(AbsFrontend):
    """Speech separation frontend 

    STFT -> T-F masking -> [STFT_0, ... , STFT_S]
    """

    def __init__(
            self,
            enh_type: str = 'tf_maksing',
            fs: Union[int, str] = 16000,
            enh_conf: Dict = None,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.enh_type = enh_type

        self.enh_model = frontend_choices.get_class(enh_type)(**enh_conf)

    def output_size(self) -> int:
        return self.bins

    def forward_rawwav(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_wavs, ilens = self.enh_model.forward_rawwav(input, input_lengths)

        return predicted_wavs, ilens

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): raw wave input [batch, samples]
            input_lengths (torch.Tensor): [batch]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            predcited magnitude spectrum [Batch, num_speaker, T, F]
        """
        # 1. Domain-conversion: e.g. Stft: time -> time-freq

        predicted_magnitude, flens = self.enh_model(input, input_lengths)

        return predicted_magnitude, flens
