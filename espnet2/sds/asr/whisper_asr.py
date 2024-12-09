from espnet2.sds.asr.abs_asr import AbsASR
import os
import numpy as np
import torch
from typeguard import typechecked
import librosa

class WhisperASRModel(AbsASR):
    """Whisper ASR"""

    @typechecked
    def __init__(
        self,
        tag = "large",
        device="cuda",
    ): 
        super().__init__()
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e
        self.s2t = whisper.load_model(tag, device=device)
        self.device=device
    
    def warmup(self):
        with torch.no_grad():
            dummy_input = torch.randn(
                    (3000),
                    dtype=getattr(torch, "float16"),
                    device="cpu",
            ).cpu().numpy()
            res = self.s2t.transcribe(torch.tensor(dummy_input).float(), beam_size=1)["text"]
    
    def forward(self,array):
        with torch.no_grad():
            prompt=self.s2t.transcribe(torch.tensor(array).float(), beam_size=1)["text"]
            return prompt