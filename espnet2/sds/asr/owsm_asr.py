from espnet2.bin.s2t_inference import Speech2Text
from espnet2.sds.asr.abs_asr import AbsASR
import os
import numpy as np
import torch
from typeguard import typechecked
import librosa

class OWSMModel(AbsASR):
    """OWSM ASR"""

    @typechecked
    def __init__(
        self,
        tag = "espnet/owsm_v3.1_ebf",
        device="cuda",
    ): 
        super().__init__()
        self.s2t = Speech2Text.from_pretrained(
            model_tag=tag,
            device=device,
            task_sym="<asr>",
            beam_size=1,
            predict_time=False,
        )
        self.device=device
    
    def warmup(self):
        with torch.no_grad():
            dummy_input = torch.randn(
                    (3000),
                    dtype=getattr(torch, "float16"),
                    device="cpu",
            ).cpu().numpy()
            speech = librosa.util.fix_length(dummy_input, size=(16000 * 30))
            res = self.s2t(speech)
    
    def forward(self,array):
        with torch.no_grad():
            array = librosa.util.fix_length(array, size=(16000 * 30))
            prompt=" ".join(self.s2t(array)[0][0].split()[1:])
            return prompt