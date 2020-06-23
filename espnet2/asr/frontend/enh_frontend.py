from typing import Tuple
from typing import Union
from typing import Dict

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.frontend.nets.tf_mask_net import BeamformerNet
from espnet2.asr.frontend.nets.tf_mask_net import TFMaskingNet
from espnet2.asr.frontend.nets.tasnet import TasNet
from espnet2.train.class_choices import ClassChoices
from espnet2.asr.frontend.abs_frontend import AbsFrontend

frontend_choices = ClassChoices(
    name="enh_model",
    classes=dict(tf_masking=TFMaskingNet, tasnet=TasNet, wpe_beamformer=BeamformerNet),
    type_check=torch.nn.Module,
    default="tf_masking",
)


class EnhFrontend(AbsFrontend):
    """Speech separation frontend 

    STFT -> T-F masking -> [STFT_0, ... , STFT_S]
    """

    def __init__(
            self,
            enh_type: str = 'tf_maksing',
            mask_type="IRM",
            fs: int = 16000,
            tf_factor: float = 0.5,
            enh_conf: Dict = None,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        assert (tf_factor <= 1.0) and (tf_factor >= 0), "tf_factor must in 0~1"
        self.tf_factor = tf_factor

        self.enh_type = enh_type
        self.mask_type = mask_type
        self.enh_model = frontend_choices.get_class(enh_type)(**enh_conf)
        self.num_spk = self.enh_model.num_spk
        self.stft = self.enh_model.stft

    def output_size(self) -> int:
        return self.bins

    def forward_rawwav(
            self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor
    ):
        predicted_wavs, ilens, masks = self.enh_model.forward_rawwav(speech_mix, speech_mix_lengths)

        return predicted_wavs, ilens, masks

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        """
        Args:
            input (torch.Tensor): raw wave input [batch, samples]
            input_lengths (torch.Tensor): [batch]

        Returns:
            enhanced spectrum, or predicted magnitude spectrum:
                torch.Tensor or List[torch.Tensor]
            output lengths
            predcited masks: OrderedDict[
                'spk1': List[torch.Tensor(Batch, Frames, Channel, Freq)],
                'spk2': List[torch.Tensor(Batch, Frames, Channel, Freq)],
                ...
                'spkn': List[torch.Tensor(Batch, Frames, Channel, Freq)],
                'noise': List[torch.Tensor(Batch, Frames, Channel, Freq)],
            ]
        """
        # 1. Domain-conversion: e.g. Stft: time -> time-freq

        predicted_spectrums, flens, masks = self.enh_model(input, input_lengths)

        return predicted_spectrums, flens, masks
