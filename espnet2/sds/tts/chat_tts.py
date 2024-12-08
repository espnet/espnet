from espnet2.sds.tts.abs_tts import AbsTTS
import os
import numpy as np
from typeguard import typechecked
import torch

class ChatTTSModel(AbsTTS):
    """ChaTTS Model"""

    @typechecked
    def __init__(
        self,
        device="cuda",
    ): 
        super().__init__()
        try:
            import ChatTTS
        except Exception as e:
            print("Error: ChatTTS is not properly installed.")
            raise e
        self.text2speech = ChatTTS.Chat()
        self.text2speech.load(compile=False)
    
    def warmup(self):
        with torch.no_grad():
            wav=self.text2speech.infer(["Sid"])[0]
    
    def forward(self,transcript):
        with torch.no_grad():
            audio_chunk=self.text2speech.infer([transcript])[0]
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            return (24000, audio_chunk)