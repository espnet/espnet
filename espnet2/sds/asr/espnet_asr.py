from espnet2.bin.asr_inference import Speech2Text
from espnet2.sds.asr.abs_asr import AbsASR
import os
import numpy as np
import torch
from typeguard import typechecked
import librosa

class ESPnetASRModel(AbsASR):
    """ESPnet ASR"""

    @typechecked
    def __init__(
        self,
        tag = "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp",
        device="cuda",
    ): 
        super().__init__()
        self.s2t = Speech2Text.from_pretrained(
            model_tag=tag,
            device=device,
            beam_size=1,
        )
        self.device=device
    
    def warmup(self):
        with torch.no_grad():
            dummy_input = torch.randn(
                    (3000),
                    dtype=getattr(torch, "float16"),
                    device="cpu",
            ).cpu().numpy()
            res = self.s2t(dummy_input)[0][0]
    
    def forward(self,array):
        with torch.no_grad():
            prompt=self.s2t(array)[0][0]
            return prompt